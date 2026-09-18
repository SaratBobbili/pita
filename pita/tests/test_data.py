"""Data assembly: tokenization, prompt-level splitting, collation, reward labelling."""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

from pita.data import build_examples, collate, roc_auc, split_by_prompt

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubTokenizer:
    """Deterministic stand-in so these tests need no model download."""

    BOS, GEN = 1, 2

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        ids = [self.BOS] + [ord(c) % 50 + 3 for c in messages[0]["content"]] + [self.GEN]
        return ids if tokenize else "".join(map(str, ids))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 + 3 for c in text]}


def records(n=4, per_prompt=2):
    return [
        {"prompt": f"prompt-{i}", "response": f"response-{i}-{j}", "reward": 0.1 * j}
        for i in range(n)
        for j in range(per_prompt)
    ]


# ---------------------------------------------------------------- tokenization

def test_prompt_ids_are_unpadded():
    """Upstream stored the already-padded prompt row, so pads leaked in as context."""
    data = build_examples(records(1, 1), StubTokenizer())
    assert 0 not in data["input_ids"][0], "no pad ids should appear before collation"


def test_target_ids_start_at_the_last_prompt_token():
    """The Q head scores the token *after* each position, so supervision must start at
    the final prompt token for the first generated token to be covered."""
    tok = StubTokenizer()
    data = build_examples([{"prompt": "ab", "response": "cd", "reward": 1.0}], tok)
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": "ab"}])
    assert data["input_ids"][0] == prompt_ids[:-1]
    assert data["target_ids"][0][0] == prompt_ids[-1]
    assert data["target_ids"][0][1:] == tok("cd")["input_ids"]


def test_first_decision_only_mode():
    data = build_examples(
        [{"prompt": "ab", "response": "cdef", "reward": 1.0}],
        StubTokenizer(), use_all_response_tokens=False,
    )
    assert len(data["target_ids"][0]) == 1


def test_max_length_truncates_targets():
    data = build_examples(
        [{"prompt": "abc", "response": "d" * 50, "reward": 1.0}],
        StubTokenizer(), max_length=10,
    )
    assert len(data["input_ids"][0]) + len(data["target_ids"][0]) <= 10


# ---------------------------------------------------------------- splitting

def test_split_holds_out_whole_prompts():
    """A response-level split would leak: siblings share a prompt."""
    data = build_examples(records(10, 3), StubTokenizer())
    train, held = split_by_prompt(data, eval_ratio=0.3, eval_max_size=-1, seed=0)
    assert len(train["input_ids"]) + len(held["input_ids"]) == len(data["input_ids"])
    assert len(held["input_ids"]) > 0
    assert "prompt" not in train, "the grouping key should not reach the collator"


def test_split_is_deterministic():
    data = build_examples(records(10, 2), StubTokenizer())
    a, _ = split_by_prompt(data, 0.2, -1, seed=7)
    b, _ = split_by_prompt(data, 0.2, -1, seed=7)
    assert a["input_ids"] == b["input_ids"]


def test_eval_max_size_caps_holdout():
    data = build_examples(records(20, 3), StubTokenizer())
    _, held = split_by_prompt(data, 0.5, eval_max_size=4, seed=0)
    assert len(held["input_ids"]) == 4


# ---------------------------------------------------------------- collation

def test_collate_right_pads_and_masks():
    items = [
        {"input_ids": [5, 6], "target_ids": [7], "rewards": 1.0, "loss_weights": 1.0},
        {"input_ids": [1, 2, 3], "target_ids": [4, 8], "rewards": 0.0, "loss_weights": 1.0},
    ]
    out = collate(items, pad_token_id=99)

    assert out["input_ids"].shape == (2, 5)
    # Short row pads on the right, so unpadded positions keep their natural indices.
    assert out["input_ids"][0].tolist() == [5, 6, 7, 99, 99]
    assert out["attention_mask"][0].tolist() == [True, True, True, False, False]
    assert out["loss_mask"][0].tolist() == [False, False, True, False, False]
    assert out["loss_mask"][1].tolist() == [False, False, False, True, True]
    assert not (out["loss_mask"] & ~out["attention_mask"]).any(), "pads must never be supervised"


# ---------------------------------------------------------------- metrics

def test_roc_auc_matches_known_values():
    perfect = roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert perfect == pytest.approx(1.0)
    inverted = roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert inverted == pytest.approx(0.0)
    tied = roc_auc(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert tied == pytest.approx(0.5)


def test_roc_auc_undefined_for_one_class():
    assert np.isnan(roc_auc(torch.tensor([0.1, 0.9]), torch.tensor([1.0, 1.0])))


# ---------------------------------------------------------------- reward labelling

def test_win_rate_ordering_and_range():
    win_rates = _load("build_dataset").win_rates
    rates = win_rates([3.0, 1.0, -2.0])
    assert ((rates >= 0) & (rates <= 1)).all()
    assert rates[0] > rates[1] > rates[2], "higher PairRM score must mean higher reward"


def test_win_rate_is_half_when_all_tie():
    win_rates = _load("build_dataset").win_rates
    assert np.allclose(win_rates([2.0, 2.0, 2.0]), 0.5)
