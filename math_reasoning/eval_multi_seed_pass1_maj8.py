import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from accuracy_utils import (
    compute_majority_vote_correct,
    equivalence_partition,
    numeric_or_symbolic_correctness,
    process_sample,
)
from utils import read_jsonl


# Resolve all paths relative to this script so execution cwd does not matter.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Root directory that contains one folder per parent seed.
INFERENCE_RESULTS_DIR = os.path.join(BASE_DIR, "inference_results")
# Ground-truth problem file used for evaluation answers.
DATA_PATH = os.path.join(BASE_DIR, "dataset", "gsm8k_test.jsonl")
# Split map used to keep only eval problems and preserve eval indexing.
TRAIN_EVAL_SAVE_PATH = os.path.join(BASE_DIR, "dataset", "gsm8k_test_eval.json")
# Number of samples in each parent-seed folder for each problem.
NUM_SAMPLES = 8
# GSM8K matching setup from eval_ckpt.py.
EXTRACT_LAST_OCCURRENCE = True


def load_eval_examples():
    with open(TRAIN_EVAL_SAVE_PATH, "r") as f:
        train_eval_problems_d = json.load(f)
    original_examples = read_jsonl(DATA_PATH)
    eval_examples = []
    for example in original_examples:
        if train_eval_problems_d[example["problem"]]["split"] == "eval":
            eval_examples.append(example)
    return eval_examples


def evaluate_parent_seed(parent_dir, eval_examples):
    maj8_correct = []
    pass1_by_sample = [[] for _ in range(NUM_SAMPLES)]

    for i in tqdm(range(len(eval_examples)), desc=f"eval {os.path.basename(parent_dir)}"):
        answer_processed = str(eval_examples[i]["answer"])
        sample_predictions = []

        for j in range(NUM_SAMPLES):
            sample_path = os.path.join(parent_dir, f"{i}_r{j}.json")
            assert os.path.exists(sample_path), f"missing {sample_path}"
            with open(sample_path, "r") as f:
                sample_predictions.append(json.load(f)["prediction"])

        processed_predictions = [
            process_sample(pred, None, EXTRACT_LAST_OCCURRENCE) for pred in sample_predictions
        ]
        correctness = [
            numeric_or_symbolic_correctness(pred, answer_processed) if pred is not None else False
            for pred in processed_predictions
        ]

        sample_partition = equivalence_partition(processed_predictions, numeric_or_symbolic_correctness)
        maj8_correct.append(
            compute_majority_vote_correct(
                processed_predictions,
                correctness,
                sample_partition,
                strict_tie_breaking=False,
            )
        )
        for j in range(NUM_SAMPLES):
            pass1_by_sample[j].append(correctness[j])

    maj8_acc = float(np.mean(maj8_correct))
    pass1_acc_by_sample = [float(np.mean(v)) for v in pass1_by_sample]
    return maj8_acc, pass1_acc_by_sample


def main():
    eval_examples = load_eval_examples()

    parent_seed_dirs = [
        os.path.join(INFERENCE_RESULTS_DIR, d)
        for d in sorted(os.listdir(INFERENCE_RESULTS_DIR))
        if d.startswith("individual_eval_inference_")
        and os.path.isdir(os.path.join(INFERENCE_RESULTS_DIR, d))
    ]
    assert len(parent_seed_dirs) > 0, f"no parent-seed folders in {INFERENCE_RESULTS_DIR}"

    maj8_scores = []
    pass1_scores = []
    per_parent_report = defaultdict(dict)

    for parent_dir in parent_seed_dirs:
        maj8_acc, pass1_acc_by_sample = evaluate_parent_seed(parent_dir, eval_examples)
        maj8_scores.append(maj8_acc)
        pass1_scores.extend(pass1_acc_by_sample)
        per_parent_report[os.path.basename(parent_dir)]["maj@8"] = maj8_acc
        per_parent_report[os.path.basename(parent_dir)]["pass@1_samples"] = pass1_acc_by_sample

    summary = {
        "num_parent_seeds": len(parent_seed_dirs),
        "num_pass1_seeds": len(pass1_scores),
        "maj@8_mean": float(np.mean(maj8_scores)),
        "maj@8_std": float(np.std(maj8_scores)),
        "pass@1_mean": float(np.mean(pass1_scores)),
        "pass@1_std": float(np.std(pass1_scores)),
        "per_parent_seed": dict(per_parent_report),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
