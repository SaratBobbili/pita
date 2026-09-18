"""ValueClassifier: head sizing, loss, gradients, and checkpoint round-trips."""

import pytest
import torch
from transformers import AutoModel, LlamaConfig, Qwen2Config

from pita.classifier import ValueClassifier
from pita.data import collate

REF_VOCAB = 96


def backbone(kind="llama"):
    torch.manual_seed(0)
    cls = LlamaConfig if kind == "llama" else Qwen2Config
    config = cls(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return AutoModel.from_config(config)


def make(kind="llama", **kwargs):
    return ValueClassifier(backbone(kind), ref_vocab_size=REF_VOCAB, **kwargs)


def batch(n=3, prompt_len=4, response_len=5):
    torch.manual_seed(1)
    items = [
        {
            "input_ids": torch.randint(0, REF_VOCAB, (prompt_len,)).tolist(),
            "target_ids": torch.randint(0, REF_VOCAB, (response_len - i,)).tolist(),
            "rewards": 0.25 * i,
            "loss_weights": 1.0,
        }
        for i in range(n)
    ]
    return collate(items, pad_token_id=0)


# ---------------------------------------------------------------- head geometry

@pytest.mark.parametrize("kind", ["llama", "qwen"])
def test_backbone_is_composed_not_subclassed(kind):
    """Any AutoModel works; there is no per-architecture registry to extend."""
    model = make(kind)
    assert model.backbone.config.model_type in ("llama", "qwen2")
    assert model.score.out_features == REF_VOCAB


def test_q_head_width_follows_reference_vocab():
    """Head width must be the *reference* config.vocab_size, not len(tokenizer).

    Guidance indexes this head with reference top-k token ids, so a narrower head would
    index out of bounds on rare tokens.
    """
    model = make()
    assert model.backbone.config.vocab_size == 128
    assert model.score.out_features == REF_VOCAB


def test_v_head_is_scalar():
    assert make(head_type="V").score.out_features == 1


def test_mle_head_allocates_atoms():
    model = make(loss_type="mle", num_atoms=7)
    assert model.score.out_features == REF_VOCAB * 7


def test_rejects_unknown_head_and_loss():
    with pytest.raises(ValueError, match="head_type"):
        make(head_type="Z")
    with pytest.raises(ValueError, match="loss_type"):
        make(loss_type="hinge")


# ---------------------------------------------------------------- training step

@pytest.mark.parametrize("head_type,loss_type", [("Q", "bce"), ("Q", "mse"), ("Q", "mle"), ("V", "bce")])
def test_forward_produces_finite_loss_and_gradients(head_type, loss_type):
    model = make(head_type=head_type, loss_type=loss_type)
    data = batch()
    loss, logits = model(
        input_ids=data["input_ids"], attention_mask=data["attention_mask"],
        labels=data["rewards"], loss_mask=data["loss_mask"], loss_weights=data["loss_weights"],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.score.weight.grad is not None
    assert torch.isfinite(model.score.weight.grad).all()
    expected_len = data["input_ids"].shape[1] - (1 if head_type == "Q" else 0)
    assert logits.shape[:2] == (data["input_ids"].shape[0], expected_len)


def test_padding_does_not_change_the_loss():
    """Right padding plus loss_mask must make batch composition irrelevant.

    Upstream left-padded without supplying position_ids, so a causal model silently saw
    shifted positions.
    """
    model = make().eval()
    item = {
        "input_ids": [5, 9, 3, 7], "target_ids": [2, 11, 4],
        "rewards": 0.75, "loss_weights": 1.0,
    }
    longer = {
        "input_ids": [1, 2, 3, 4, 5, 6], "target_ids": [7, 8, 9, 10],
        "rewards": 0.25, "loss_weights": 1.0,
    }

    def loss_of(items, index):
        data = collate(items, pad_token_id=0)
        with torch.no_grad():
            _, logits = model(
                input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                labels=data["rewards"], loss_mask=data["loss_mask"],
                loss_weights=data["loss_weights"],
            )
        mask = data["loss_mask"][index, 1:]
        return logits[index][mask]

    alone = loss_of([item], 0)
    padded = loss_of([item, longer], 0)
    assert torch.allclose(alone, padded, atol=1e-4)


def test_predictions_are_probabilities():
    for loss_type in ("bce", "mse", "mle"):
        model = make(loss_type=loss_type)
        data = batch()
        with torch.no_grad():
            _, logits = model(
                input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                labels=data["rewards"], loss_mask=data["loss_mask"],
                loss_weights=data["loss_weights"],
            )
        predictions = model.calculate_predictions(logits)
        assert ((predictions >= 0) & (predictions <= 1)).all(), loss_type


# ---------------------------------------------------------------- checkpointing

@pytest.mark.parametrize("head_type", ["Q", "V"])
def test_save_load_round_trip(tmp_path, head_type):
    model = make(head_type=head_type).eval()
    torch.nn.init.normal_(model.score.weight, std=0.02)
    data = batch()
    with torch.no_grad():
        _, before = model(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"],
            labels=data["rewards"], loss_mask=data["loss_mask"], loss_weights=data["loss_weights"],
        )

    model.save_pretrained(str(tmp_path))
    reloaded = ValueClassifier.from_pretrained(str(tmp_path)).eval()

    assert reloaded.head_type == head_type
    assert reloaded.ref_vocab_size == REF_VOCAB
    with torch.no_grad():
        _, after = reloaded(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"],
            labels=data["rewards"], loss_mask=data["loss_mask"], loss_weights=data["loss_weights"],
        )
    assert torch.allclose(before, after, atol=1e-5)


def test_score_candidates_matches_dense_head():
    """The gathered Q lookup must equal materialising the whole head."""
    model = make().eval()
    torch.nn.init.normal_(model.score.weight, std=0.02)
    hidden = torch.randn(3, model.backbone.config.hidden_size)
    candidates = torch.randint(0, REF_VOCAB, (3, 5))

    got = model.score_candidates(hidden, candidates)
    dense = model.score(hidden)
    expected = torch.gather(dense, 1, candidates)
    assert torch.allclose(got, expected, atol=1e-4)
