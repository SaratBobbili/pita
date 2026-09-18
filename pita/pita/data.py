"""Turn ranked generations into classifier training tensors.

Derived from ``refactor_old/training/dataset.py`` and ``training/builder.py``, with the
arithmetic/GSM8K branch dropped and two upstream problems fixed:

*Left-pad leakage.* Upstream stored the already-padded prompt tensor as
``partial_guided_prompts_tokenized`` and the collator then marked those pads as attended,
so the model read ``128002`` pad tokens as real context. Prompts are tokenized unpadded
here, and padding is added only at collation.

*Left padding.* Upstream left-padded training batches, which silently shifts positions
for a causal model unless ``position_ids`` is supplied (it was not). Batches are
right-padded instead, so the default ``arange`` positions are correct.

The training signal is one scalar reward per response, broadcast to every response token:
the classifier learns to predict, from any prefix, whether the continuation will be
preferred.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED_COLUMNS = ("prompt", "response", "reward")


def read_pairs(path: str) -> list[dict]:
    """Load the parquet written by ``scripts/build_dataset.py``."""
    frame = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; got {list(frame.columns)}")
    return frame.to_dict("records")


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Chat-templated prompt ids, unpadded.

    ``apply_chat_template`` is the architecture-neutral path; SPPO's generate script
    instead matched on substrings of the model name and raised for anything unrecognised.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )


def build_examples(records, tokenizer, max_length: int = -1, use_all_response_tokens: bool = True):
    """Tokenize ``(prompt, response, reward)`` records into flat training examples.

    ``input_ids`` is the context and ``target_ids`` the tokens whose value is supervised.
    The split puts the prompt's final token at the head of ``target_ids`` so that the Q
    head, which scores the token *following* each position, is supervised from the very
    first generated token onward.

    With ``use_all_response_tokens`` false only that first decision is supervised, which
    is the upstream ``use_all_ref_tokens=0`` ablation.
    """
    data = {"input_ids": [], "target_ids": [], "rewards": [], "loss_weights": [], "prompt": []}
    for record in records:
        prompt_ids = encode_prompt(tokenizer, record["prompt"])
        if len(prompt_ids) < 2:
            continue
        response_ids = tokenizer(record["response"], add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids[:-1]
        target_ids = [prompt_ids[-1]]
        if use_all_response_tokens:
            target_ids = target_ids + list(response_ids)

        if max_length != -1:
            if len(input_ids) >= max_length - 1:
                continue
            target_ids = target_ids[: max_length - len(input_ids)]
        if not target_ids:
            continue

        data["input_ids"].append(input_ids)
        data["target_ids"].append(target_ids)
        data["rewards"].append(float(record["reward"]))
        data["loss_weights"].append(1.0)
        data["prompt"].append(record["prompt"])
    return data


def split_by_prompt(data, eval_ratio: float, eval_max_size: int, seed: int = 47):
    """Hold out whole prompts, never individual responses.

    Splitting on responses would leak: several responses share a prompt, so the classifier
    would be scored on prompts it trained on.
    """
    prompts = sorted(set(data["prompt"]))
    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)
    n_eval = max(1, int(len(prompts) * eval_ratio)) if eval_ratio > 0 else 0
    eval_prompts = set(prompts[:n_eval])

    keys = [k for k in data if k != "prompt"]
    train = {k: [] for k in keys}
    held = {k: [] for k in keys}
    for i, prompt in enumerate(data["prompt"]):
        target = held if prompt in eval_prompts else train
        for k in keys:
            target[k].append(data[k][i])

    if eval_max_size != -1 and len(held["input_ids"]) > eval_max_size:
        pick = rng.choice(len(held["input_ids"]), eval_max_size, replace=False)
        held = {k: [held[k][i] for i in pick] for k in keys}
    return train, held


class ClassifierDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}


def collate(batch, pad_token_id: int):
    """Right-pad a batch; ``loss_mask`` selects the supervised target tokens."""
    width = max(len(x["input_ids"]) + len(x["target_ids"]) for x in batch)
    input_ids, attention_mask, loss_mask = [], [], []
    for x in batch:
        prefix, target = x["input_ids"], x["target_ids"]
        pad = width - len(prefix) - len(target)
        input_ids.append(torch.tensor(list(prefix) + list(target) + [pad_token_id] * pad, dtype=torch.long))
        attention_mask.append(torch.tensor([1] * (len(prefix) + len(target)) + [0] * pad, dtype=torch.bool))
        loss_mask.append(torch.tensor([0] * len(prefix) + [1] * len(target) + [0] * pad, dtype=torch.bool))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "loss_mask": torch.stack(loss_mask),
        "rewards": torch.tensor([x["rewards"] for x in batch], dtype=torch.float),
        "loss_weights": torch.tensor([x["loss_weights"] for x in batch], dtype=torch.float),
    }


def explained_variance(predictions, labels):
    return 1 - torch.var(predictions - labels) / torch.var(labels).clamp(min=1e-8)


def r_squared(predictions, labels):
    ss_res = torch.sum(torch.square(labels - predictions))
    ss_tot = torch.sum(torch.square(labels - torch.mean(labels))).clamp(min=1e-8)
    return 1 - ss_res / ss_tot


def roc_auc(predictions, labels):
    """ROC-AUC via the Mann-Whitney U identity, so sklearn is not a dependency.

    AUC is the probability a random positive outranks a random negative, which is the
    normalised rank sum of the positives. ``labels`` are thresholded at 0.5 because
    rewards are soft win rates rather than hard classes. Returns NaN when one class is
    absent and AUC is undefined.
    """
    positive = labels > 0.5
    n_pos = int(positive.sum())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average_rank handles ties, which are common when many tokens share a prediction.
    order = predictions.float().argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, predictions.numel() + 1, device=predictions.device, dtype=torch.float)
    unique, inverse = torch.unique(predictions.float(), return_inverse=True)
    tie_sum = torch.zeros(unique.numel(), device=predictions.device).index_add_(0, inverse, ranks)
    tie_count = torch.zeros(unique.numel(), device=predictions.device).index_add_(
        0, inverse, torch.ones_like(ranks)
    )
    ranks = (tie_sum / tie_count)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
