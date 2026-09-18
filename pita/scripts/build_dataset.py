"""PairRM scores -> per-response rewards -> the parquet the trainer consumes.

Replaces ``SPPO/scripts/compute_prob.py``. SPPO needed a chosen/rejected pair plus three
probability columns for its regression loss; PITA needs a scalar reward per response,
because the classifier is trained pointwise to predict whether a continuation will win.

Reward is the response's **win rate** against its siblings::

    p_i = mean_j sigmoid(s_i - s_j)

which is the Bradley-Terry probability that response i beats a randomly drawn sibling. A
soft target in [0, 1] is exactly what the guidance later consumes -- the offset
``eta * z`` tilts logits by the log-odds of winning -- and BCE with soft targets is a
proper scoring rule for it. ``--binarize`` falls back to SPPO's argmax/argmin pair.

No Hub push: upstream uploaded every round to a private dataset repo, which is an
unnecessary dependency for a local pipeline.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument("--ranking_dir", type=str, default=None,
                        help="Defaults to ranking/<output_dir>.")
    parser.add_argument("--train_file", type=str, required=True, help="Destination parquet.")
    parser.add_argument("--prompts", type=str, default="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1")
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--num_gpu", type=int, default=8)
    parser.add_argument("--binarize", action="store_true",
                        help="Reward 1 for the best response and 0 for the worst, dropping the rest.")
    parser.add_argument("--drop_no_variation", action="store_true",
                        help="Drop prompts whose responses all score alike; they carry no signal.")
    parser.add_argument("--variation_eps", type=float, default=1e-3)
    return parser.parse_args()


def load_scores(args, num_prompts):
    """Stitch the per-worker PairRM score shards back into prompt order."""
    ranking_dir = args.ranking_dir or os.path.join("ranking", args.output_dir)
    gpus = args.gpu_ids.strip("()").split(",") if args.gpu_ids else list(range(args.num_gpu))
    scores = [None] * num_prompts
    for data_frac, gpu in enumerate(gpus):
        shard = np.load(os.path.join(ranking_dir, f"{gpu}_{data_frac}.npy"))
        for local_index, row in enumerate(shard):
            index = data_frac * args.frac_len + local_index
            if index < num_prompts:
                scores[index] = row
    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise ValueError(f"no PairRM score for {len(missing)} prompts, e.g. {missing[:5]}")
    return scores


def win_rates(scores):
    """Bradley-Terry win rate of each response against its siblings."""
    scores = np.asarray(scores, dtype=np.float64)
    diff = scores[:, None] - scores[None, :]
    probabilities = 1.0 / (1.0 + np.exp(-diff))
    np.fill_diagonal(probabilities, 0.5)
    return probabilities.mean(axis=1)


def main():
    args = parse_arguments()
    data = load_dataset(args.prompts, split=args.prompt_split)
    prompts = [data[i]["prompt"] for i in range(len(data))]

    responses = []
    for pair in range(args.pairs):
        with open(os.path.join(args.output_dir, f"responses_{pair}.json")) as f:
            responses.append(json.load(f))
    for pair, column in enumerate(responses):
        if len(column) != len(prompts):
            raise ValueError(f"responses_{pair}.json has {len(column)} rows, expected {len(prompts)}")

    scores = load_scores(args, len(prompts))

    rows, dropped = [], 0
    for index, prompt in enumerate(prompts):
        rates = win_rates(scores[index])
        if args.drop_no_variation and float(rates.max() - rates.min()) < args.variation_eps:
            dropped += 1
            continue

        if args.binarize:
            picks = [(int(rates.argmax()), 1.0), (int(rates.argmin()), 0.0)]
        else:
            picks = [(j, float(rates[j])) for j in range(args.pairs)]

        for sample_index, reward in picks:
            rows.append({
                "prompt_id": index,
                "prompt": prompt,
                "response": responses[sample_index][index],
                "reward": reward,
                "win_rate": float(rates[sample_index]),
                "rm_score": float(scores[index][sample_index]),
                "sample_index": sample_index,
            })

    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.train_file)), exist_ok=True)
    frame.to_parquet(args.train_file, index=False)
    print(
        f"wrote {len(frame)} rows from {len(prompts) - dropped} prompts to {args.train_file}"
        + (f" ({dropped} dropped for no variation)" if dropped else "")
    )
    print(f"reward mean {frame['reward'].mean():.4f}, std {frame['reward'].std():.4f}")


if __name__ == "__main__":
    main()
