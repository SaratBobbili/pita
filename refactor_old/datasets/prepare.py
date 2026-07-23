import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def prepare_alpaca_split(dataset_name="tatsu-lab/alpaca_farm",
                         dataset_config="alpaca_noisy_multi_preference",
                         hf_split="preference",
                         eval_ratio=0.1,
                         seed=47,
                         output_path="math_reasoning/dataset/alpaca_noisy_multi_preference_train_eval.json",
                         trust_remote_code=True):
    dataset_dict = load_dataset(dataset_name, dataset_config, trust_remote_code=trust_remote_code)
    data = dataset_dict[hf_split]

    kept_rows = []
    for row in tqdm(data, desc="Filtering rows"):
        current_input = (row["input"] or "").strip()
        if current_input:
            continue
        kept_rows.append(row)

    kept_size = len(kept_rows)
    eval_size = int(kept_size * eval_ratio)
    rng = random.Random(seed)
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

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved to: {output_path}")
    print(f"Kept rows: {kept_size}, train: {kept_size - eval_size}, eval: {eval_size}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="tatsu-lab/alpaca_farm")
    parser.add_argument("--dataset_config", default="alpaca_noisy_multi_preference")
    parser.add_argument("--hf_split", default="preference")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output_path", default="math_reasoning/dataset/alpaca_noisy_multi_preference_train_eval.json")
    args = parser.parse_args()
    prepare_alpaca_split(**vars(args))
