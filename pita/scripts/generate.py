"""Sample K responses per prompt from the frozen policy, optionally PITA-guided.

Derived from ``SPPO/scripts/generate.py``. Same sharding contract, so
``combine_generate.py`` and ``rank.py`` work unchanged: each of the 8 workers writes
``responses_{data_frac}_{pair}.json``.

Two changes. The tokenizer is resolved with ``AutoTokenizer`` rather than matched against
substrings of the model name (upstream raised ``ValueError("Model not supported")`` for
anything outside mistral/llama-3/gemma-2). And the engine carries the PITA logits
processor, which tilts each step's logits toward the value classifier.

``--eta 0`` skips the classifier entirely -- that is round 1, and the unguided baseline.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

from pita.guidance import PITAGuidedLogitsProcessor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Frozen reference policy.")
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument("--prompts", type=str, default="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1")
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--maxlen", type=int, default=2048, help="Max tokens to generate.")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Engine context window: must hold prompt + generation.")
    parser.add_argument("--pairs", type=int, default=5, help="Responses sampled per prompt.")
    parser.add_argument("--frac_len", type=int, default=0, help="Prompts per shard.")
    parser.add_argument("--data_frac", type=int, default=0, help="Which shard this worker takes.")
    parser.add_argument("--world_size", type=int, default=1, help="Tensor-parallel size.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--classifier_path", type=str, default=None, help="Round r-1 checkpoint.")
    parser.add_argument("--eta", type=float, default=0.0, help="Guidance strength; 0 is unguided.")
    parser.add_argument("--guide_top_k", type=int, default=20,
                        help="Candidates scored per step; -1 scores the whole vocabulary.")
    parser.add_argument("--inference_mode", type=str, default="expectation",
                        choices=["expectation", "bernoulli"])
    parser.add_argument("--cd_baseline", action="store_true",
                        help="Contrastive-decoding ablation: offset is eta*sigmoid(z).")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80,
                        help="Leave headroom for the classifier and its KV bank.")
    parser.add_argument("--max_num_seqs", type=int, default=64,
                        help="Also sizes the classifier KV bank: rows x max_model_len.")
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def apply_template(text, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def split_prompts(prompts, frac_len, data_frac):
    if frac_len <= 0:
        return prompts[:]
    start = frac_len * data_frac
    end = frac_len * (data_frac + 1)
    return prompts[start:] if end > len(prompts) else prompts[start:end]


def build_engine(args):
    """Engine plus the per-request guidance settings, if guidance is on."""
    guided = args.eta != 0.0
    if guided and not args.classifier_path:
        raise ValueError("--eta != 0 requires --classifier_path")

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.world_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    if guided:
        kwargs["logits_processors"] = [PITAGuidedLogitsProcessor]
        kwargs["additional_config"] = {
            "pita": {
                "classifier_path": args.classifier_path,
                "eta": args.eta,
                "top_k": args.guide_top_k,
                "inference_mode": args.inference_mode,
                "cd_baseline": args.cd_baseline,
                "dtype": args.dtype,
            }
        }
    return LLM(**kwargs), ({"eta": args.eta} if guided else None)


def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.prompts, split=args.prompt_split)
    llm, extra_args = build_engine(args)
    tokenizer = llm.get_tokenizer()

    prompts = [apply_template(data[i]["prompt"], tokenizer) for i in range(len(data))]
    prompts = split_prompts(prompts, args.frac_len, args.data_frac)
    print(f"shard {args.data_frac}: {len(prompts)} prompts, eta={args.eta}")

    for pair in range(args.pairs):
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.maxlen,
            seed=pair * 50,
            extra_args=extra_args,
        )
        responses = llm.generate(prompts, sampling_params)
        outputs = [r.outputs[0].text for r in responses]
        path = os.path.join(args.output_dir, f"responses_{args.data_frac}_{pair}.json")
        with open(path, "w") as f:
            json.dump(outputs, f)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
