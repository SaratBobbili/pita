"""Generate AlpacaEval 2 responses under PITA guidance.

Derived from ``math_reasoning/my_alpaca_eval_code/eval_ckpt.py``: the dataset load and the
``model_outputs.json`` writer are kept (that file is the only interface AlpacaEval needs),
and its per-example HF ``generate()`` loop is replaced by the same vLLM path used during
training, so eval decoding and round decoding cannot drift apart.

Judging is a separate step and needs an OpenAI key::

    export OPENAI_API_KEY=...
    alpaca_eval --model_outputs <output_dir>/model_outputs.json \\
                --annotators_config weighted_alpaca_eval_gpt4_turbo

``--eta_sweep`` writes one subdirectory per value, since guidance strength is the knob
that moves the number most.
"""

import argparse
import json
import os
import sys

from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pita.guidance import PITAGuidedLogitsProcessor  # noqa: E402

ALPACA_EVAL_SIZE = 805


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Frozen reference policy.")
    parser.add_argument("--classifier_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--generator", type=str, default=None,
                        help="Name recorded in model_outputs.json; defaults to the checkpoint name.")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--eta_sweep", type=float, nargs="*", default=None,
                        help="Run several guidance strengths, one output directory each.")
    parser.add_argument("--guide_top_k", type=int, default=20)
    parser.add_argument("--inference_mode", type=str, default="expectation",
                        choices=["expectation", "bernoulli"])
    parser.add_argument("--cd_baseline", action="store_true")
    # SPPO's AlpacaEval settings, kept so the win rate is comparable to theirs.
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Engine context window: must hold prompt + generation.")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def build_engine(args, etas):
    """One engine serves the whole sweep; eta varies per request via extra_args."""
    guided = any(e != 0.0 for e in etas)
    if guided and not args.classifier_path:
        raise ValueError("guided evaluation requires --classifier_path")

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
                "eta": max(etas),
                "top_k": args.guide_top_k,
                "inference_mode": args.inference_mode,
                "cd_baseline": args.cd_baseline,
                "dtype": args.dtype,
            }
        }
    return LLM(**kwargs), guided


def main():
    args = parse_arguments()
    etas = args.eta_sweep if args.eta_sweep else [args.eta]

    data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
    if len(data) != ALPACA_EVAL_SIZE:
        print(f"warning: expected {ALPACA_EVAL_SIZE} instructions, got {len(data)}")

    llm, guided = build_engine(args, etas)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["instruction"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for row in data
    ]

    default_name = os.path.basename(os.path.normpath(args.classifier_path or args.model))
    for eta in etas:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            extra_args={"eta": eta} if guided else None,
        )
        responses = llm.generate(prompts, sampling_params)

        generator = args.generator or f"{default_name}_eta{eta}"
        outputs = [
            {
                "dataset": row["dataset"],
                "instruction": row["instruction"],
                "output": response.outputs[0].text,
                "generator": generator,
            }
            for row, response in zip(data, responses)
        ]

        out_dir = args.output_dir if len(etas) == 1 else os.path.join(args.output_dir, f"eta_{eta}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "model_outputs.json")
        with open(path, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"eta={eta}: wrote {len(outputs)} responses to {path}")

    print("\nJudge with:\n  export OPENAI_API_KEY=...\n"
          f"  alpaca_eval --model_outputs {args.output_dir}/model_outputs.json "
          "--annotators_config weighted_alpaca_eval_gpt4_turbo")


if __name__ == "__main__":
    main()
