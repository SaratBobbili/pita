"""Convert PITA preference JSONL into HF datasets with prompt/chosen/rejected."""
import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset, DatasetDict
from tqdm import tqdm

_MR = Path(__file__).resolve().parents[2]
if str(_MR) not in sys.path:
    sys.path.insert(0, str(_MR))
from utils import read_jsonl


MERGE_KEYS = [
    "fully_guided_predictions",
    "fully_guided_predictions_correctness",
    "partial_guided_prompts",
    "partial_guided_prompts_tokenized",
    "num_response_tokens_in_partial_guided_prompts",
    "partial_guided_responses_tokenized",
    "partial_guided_predictions",
    "partial_guided_predictions_correctness",
]


def resolve_root(cfg: DictConfig) -> Path:
    if cfg.math_reasoning_root:
        return Path(cfg.math_reasoning_root)
    return _MR


def load_and_merge(data_paths, root: Path):
    problem_position = {}
    all_data = []
    for data_path in data_paths:
        path = root / data_path if not os.path.isabs(data_path) else Path(data_path)
        current = read_jsonl(str(path))
        for ex in tqdm(current, desc=f"Loading {path.name}"):
            problem = ex["problem"]
            if problem not in problem_position:
                problem_position[problem] = len(all_data)
                all_data.append(ex)
            else:
                idx = problem_position[problem]
                for k in MERGE_KEYS:
                    all_data[idx][k].extend(ex[k])
    return all_data


def to_preference_rows(examples, split_map):
    train_rows, eval_rows = [], []
    for ex in tqdm(examples, desc="Building pairs"):
        split = split_map[ex["problem"]]["split"]
        prompt = ex["prompt"]
        partials = ex["partial_guided_predictions"]
        fulls = ex["fully_guided_predictions"]
        correctness = ex["partial_guided_predictions_correctness"]
        for j in range(len(correctness)):
            if correctness[j]:
                chosen, rejected = partials[j], fulls[j]
            else:
                chosen, rejected = fulls[j], partials[j]
            row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            if split == "train":
                train_rows.append(row)
            elif split == "eval":
                eval_rows.append(row)
            else:
                raise ValueError(f"Unknown split: {split}")
    return train_rows, eval_rows


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    root = resolve_root(cfg)
    os.chdir(root)
    print(OmegaConf.to_yaml(cfg))

    with open(root / cfg.train_eval_save_path) as f:
        split_map = json.load(f)

    all_data = load_and_merge(cfg.data_paths, root)
    train_rows, eval_rows = to_preference_rows(all_data, split_map)
    print(f"train pairs: {len(train_rows)}  eval pairs: {len(eval_rows)}")

    out_dir = root / cfg.prepared_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    )
    ds.save_to_disk(str(out_dir))
    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "num_train": len(train_rows),
                "num_eval": len(eval_rows),
                "data_paths": list(cfg.data_paths),
            },
            f,
            indent=2,
        )
    print(f"Saved dataset to {out_dir}")


if __name__ == "__main__":
    main()
