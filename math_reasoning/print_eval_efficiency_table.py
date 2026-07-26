import argparse
import csv
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_root",
    type=Path,
    default=Path(__file__).parent / "checkpoints" / "llama_3_8b_instruct_gsm8k",
)
parser.add_argument("--guided_samples", type=int, default=8)
parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
args = parser.parse_args()


def load_json(path):
    with path.open() as f:
        return json.load(f)


def load_run(name, run_dir, samples, total_divisor, accuracy_key):
    report = load_json(run_dir / "efficiency" / "efficiency_report.json")
    stats = report["inference_stats"]
    inference = report["inference"]
    reward = load_json(next(run_dir.glob("reward_stats_*.json")))
    return {
        "name": name,
        "samples": samples,
        "accuracy": reward[accuracy_key],
        "wall": stats["inference_wall_time_sec"] / total_divisor,
        "gpu_hours": stats["inference_wall_time_sec"] / total_divisor / 3600,
        "total_tokens": stats["total_generated_tokens"] / total_divisor,
        "latency": inference["total_wall_sec_per_example"],
        "generation_latency": inference["generate_wall_sec_per_example"],
        "tokens_per_example": inference["generated_tokens"],
        "throughput": stats["tokens_per_sec"],
        "guidance_calls": inference["guidance_calls_per_example"],
        "ref_flops": stats["total_ref_only_flops_est"] / total_divisor,
        "guided_flops": stats["total_guided_flops_est"] / total_divisor,
    }


root = args.checkpoint_root
training_costs = root / "training_costs"
runs = [
    load_run("Ref@1", root / "ref_pass1", 1, 1, "pass_k_accuracy_mean"),
    load_run("Ref@2", root / "ref_pass2", 2, 1, "pass_k_accuracy_mean"),
    load_run(
        "PITA",
        training_costs / "pita" / "ckpt_10000",
        1,
        args.guided_samples,
        "single_sample_accuracy_mean",
    ),
    load_run(
        "Q#",
        training_costs / "q_sharp" / "ckpt_10000",
        1,
        args.guided_samples,
        "single_sample_accuracy_mean",
    ),
    load_run(
        "CD",
        training_costs / "cd" / "ckpt_10000",
        1,
        args.guided_samples,
        "single_sample_accuracy_mean",
    ),
]


def mean_std(values, precision=4):
    return f"{values['mean']:.{precision}f} ± {values['std']:.{precision}f}"


rows = [
    ("Rollouts per problem", [str(run["samples"]) for run in runs]),
    ("Accuracy", [f"{run['accuracy']:.4f}" for run in runs]),
    ("Overall inference wall-clock time (s)", [f"{run['wall']:,.2f}" for run in runs]),
    ("Overall inference cost (GPU-hours)", [f"{run['gpu_hours']:.4f}" for run in runs]),
    ("Total generated tokens", [f"{run['total_tokens']:,.0f}" for run in runs]),
    ("End-to-end latency per rollout (s)", [mean_std(run["latency"]) for run in runs]),
    (
        "Generation-only latency per rollout (s)",
        [mean_std(run["generation_latency"]) for run in runs],
    ),
    (
        "Generated tokens per rollout",
        [mean_std(run["tokens_per_example"], 2) for run in runs],
    ),
    ("Run-level throughput (tokens/s)", [f"{run['throughput']:.2f}" for run in runs]),
    (
        "Guidance calls per rollout",
        [mean_std(run["guidance_calls"], 2) for run in runs],
    ),
    ("Reference-only compute estimate (FLOPs)", [f"{run['ref_flops']:.3e}" for run in runs]),
    ("Guided compute estimate (FLOPs)", [f"{run['guided_flops']:.3e}" for run in runs]),
]

header = ["Metric"] + [run["name"] for run in runs]
if args.format == "csv":
    writer = csv.writer(sys.stdout)
    writer.writerow(header)
    for metric, values in rows:
        writer.writerow([metric] + values)
else:
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for metric, values in rows:
        print("| " + " | ".join([metric] + values) + " |")
