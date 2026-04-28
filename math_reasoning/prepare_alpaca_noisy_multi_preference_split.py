import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AlpacaFarm noisy multi-preference split metadata."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca_farm",
        help="HuggingFace dataset name to load.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="alpaca_noisy_multi_preference",
        help="Subset/config name in the HuggingFace dataset.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="preference",
        help="HuggingFace split to convert (for this subset, use preference).",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=int,
        default=1,
        help="Whether to allow dataset loading code from the dataset repo (1/0).",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of kept rows assigned to eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
        help="Random seed used for deterministic train/eval assignment.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="math_reasoning/dataset/alpaca_noisy_multi_preference_train_eval.json",
        help="Output JSON path for mapping: instruction -> {id, split, output_1, output_2, preference}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.eval_ratio <= 1.0:
        raise ValueError(f"eval_ratio must be in [0, 1], got {args.eval_ratio}")

    dataset_dict = load_dataset(
        args.dataset_name,
        args.dataset_config,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if args.hf_split not in dataset_dict:
        raise ValueError(
            f"split '{args.hf_split}' not found. Available splits: {list(dataset_dict.keys())}"
        )
    data = dataset_dict[args.hf_split]

    kept_rows = []
    discarded_non_empty_input = 0
    for row in tqdm(data, desc="Filtering rows"):
        current_input = (row["input"] or "").strip()
        if current_input:
            discarded_non_empty_input += 1
            continue
        kept_rows.append(row)

    kept_size = len(kept_rows)
    eval_size = int(kept_size * args.eval_ratio)
    rng = random.Random(args.seed)
    shuffled_indices = list(range(kept_size))
    rng.shuffle(shuffled_indices)
    eval_indices = set(shuffled_indices[:eval_size])

    output = {}
    for idx, row in enumerate(tqdm(kept_rows, desc="Building output JSON")):
        instruction = row["instruction"]
        if instruction in output:
            raise ValueError("Duplicate instruction found after filtering.")
        output[instruction] = {
            "id": idx,
            "split": "eval" if idx in eval_indices else "train",
            "output_1": row["output_1"],
            "output_2": row["output_2"],
            "preference": row["preference"],
        }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"saved to: {args.output_path}")
    print(f"total rows in source split: {len(data)}")
    print(f"discarded rows (non-empty input): {discarded_non_empty_input}")
    print(f"kept rows: {kept_size}")
    print(f"train rows: {kept_size - eval_size}")
    print(f"eval rows: {eval_size}")


if __name__ == "__main__":
    main()
