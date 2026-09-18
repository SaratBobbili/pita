"""The classifier cache must be indistinguishable from a plain full forward.

These run on CPU with a tiny random model, so they are a fast guard on the part of the
system with no obvious failure signal: a desynced cache does not crash, it silently
degrades guidance quality.
"""

import pytest
import torch
from transformers import AutoModel, LlamaConfig
from vllm.v1.sample.logits_processor import MoveDirectionality

from pita.guidance import BankCache, PITAGuidedLogitsProcessor
from pita.masking import build_4d_mask, position_ids_for

DTYPE = torch.float32
TOL = 1e-4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    m = AutoModel.from_config(config).eval()
    m.set_attn_implementation("sdpa")
    return m


def _bank(model, max_rows=4, capacity=32):
    c = model.config
    return BankCache(
        num_layers=c.num_hidden_layers, max_rows=max_rows,
        num_kv_heads=c.num_key_value_heads, capacity=capacity,
        head_dim=c.hidden_size // c.num_attention_heads, device="cpu", dtype=DTYPE,
    )


def _prefill(model, bank, seqs, lengths):
    """Replay `lengths[i]` tokens of each sequence into the bank."""
    widest = max(lengths)
    padded = [s[:n] + [0] * (widest - n) for s, n in zip(seqs, lengths)]
    rows = torch.arange(len(seqs))
    starts = torch.zeros(len(seqs), dtype=torch.long)
    bank.rows, bank.write_pos, bank.key_len, bank.detached = rows, starts, widest, True
    with torch.inference_mode():
        model(
            input_ids=torch.tensor(padded),
            attention_mask=build_4d_mask(starts, widest, widest, DTYPE),
            position_ids=position_ids_for(starts, widest),
            past_key_values=bank, use_cache=True,
        )
    bank.detached = False
    for i, n in enumerate(lengths):
        bank.lengths[i] = n


def _full_forward(model, ids):
    t = torch.tensor([ids])
    with torch.inference_mode():
        return model(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state[0, -1]


def test_decode_step_matches_full_forward(model):
    """Ragged prefill plus one cached decode step == one uncached forward."""
    seqs = [[5, 9, 3, 7, 2, 11], [8, 1, 4, 6, 6, 6]]
    lengths = [4, 2]
    bank = _bank(model)
    _prefill(model, bank, seqs, lengths)

    starts = bank.lengths[:2]
    key_len = int(starts.max()) + 1
    bank.rows, bank.write_pos, bank.key_len, bank.detached = torch.arange(2), starts, key_len, False
    with torch.inference_mode():
        got = model(
            input_ids=torch.tensor([[s[n]] for s, n in zip(seqs, lengths)]),
            attention_mask=build_4d_mask(starts, 1, key_len, DTYPE),
            position_ids=position_ids_for(starts, 1),
            past_key_values=bank, use_cache=True,
        ).last_hidden_state[:, -1]

    for i, (s, n) in enumerate(zip(seqs, lengths)):
        assert torch.allclose(got[i], _full_forward(model, s[: n + 1]), atol=TOL)


def test_candidates_see_prefix_but_not_each_other(model):
    """V head: each candidate must score as if it alone followed the prefix."""
    seqs = [[5, 9, 3, 7], [8, 1, 4, 2]]
    lengths = [4, 3]
    bank = _bank(model)
    _prefill(model, bank, seqs, lengths)

    candidates = torch.tensor([[11, 12, 13], [21, 22, 23]])
    k = candidates.shape[1]
    starts = bank.lengths[:2]
    key_len = int(starts.max()) + k
    bank.rows, bank.write_pos, bank.key_len, bank.detached = torch.arange(2), starts, key_len, False
    with torch.inference_mode():
        got = model(
            input_ids=candidates,
            attention_mask=build_4d_mask(starts, k, key_len, DTYPE, block="diagonal"),
            position_ids=position_ids_for(starts, k, same_position=True),
            past_key_values=bank, use_cache=True,
        ).last_hidden_state

    for b in range(2):
        for j in range(k):
            expected = _full_forward(model, seqs[b][: lengths[b]] + [int(candidates[b, j])])
            assert torch.allclose(got[b, j], expected, atol=TOL), f"candidate {b},{j}"

    # Candidates are hypothetical: they must not advance the rows.
    assert bank.lengths[:2].tolist() == lengths


def test_reorder_swap(model):
    seqs, lengths = [[5, 9, 3, 7], [8, 1, 4, 2]], [4, 3]
    bank = _bank(model)
    _prefill(model, bank, seqs, lengths)
    before = [bank.keys[0][0].clone(), bank.keys[0][1].clone()]

    bank.reorder(0, 1, MoveDirectionality.SWAP)

    assert torch.equal(bank.keys[0][0], before[1])
    assert torch.equal(bank.keys[0][1], before[0])
    assert bank.lengths[:2].tolist() == [lengths[1], lengths[0]]


def test_reorder_unidirectional(model):
    seqs, lengths = [[5, 9, 3, 7], [8, 1, 4, 2]], [4, 3]
    bank = _bank(model)
    _prefill(model, bank, seqs, lengths)
    source = bank.keys[0][0].clone()

    bank.reorder(0, 1, MoveDirectionality.UNIDIRECTIONAL)

    assert torch.equal(bank.keys[0][1], source)
    assert int(bank.lengths[1]) == lengths[0]


@pytest.mark.parametrize("eta", [0.5, 2.0])
def test_expectation_offset_can_boost(eta):
    """Regression: the upstream odds-ratio clamp forced every offset non-positive.

    See refactor_old/models/guidance.py:84-86.
    """
    processor = PITAGuidedLogitsProcessor.__new__(PITAGuidedLogitsProcessor)
    processor.inference_mode = "expectation"
    processor.cd_baseline = False
    processor.classifier = type("C", (), {"loss_type": "bce"})()

    z = torch.tensor([[-1.0, 0.0, 1.0, 3.0]])
    offsets = processor._offsets(z, torch.tensor([[eta]]))

    assert torch.allclose(offsets, eta * z)
    assert (offsets[0, 2:] > 0).all(), "positive values must raise a token's logit"
    assert (offsets[0, 0] < 0).item(), "negative values must lower it"
