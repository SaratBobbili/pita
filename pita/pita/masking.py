"""4D attention masks built in plain torch.

Transformers exposes per-architecture helpers such as
``_prepare_4d_causal_attention_mask_with_cache_position``, but they are private, differ
per model file, and were removed in transformers 5.x. Everything the classifier needs is
a handful of comparisons, so we build the masks ourselves and stay model-agnostic.

Masks are additive float masks: ``0.0`` means attend, ``finfo.min`` means masked.

Cache layout assumed throughout: the classifier's KV bank is **left-aligned**. Row ``r``
holding ``lengths[r]`` valid tokens occupies key positions ``[0, lengths[r])``, and the
``q_len`` freshly appended queries land at ``[lengths[r], lengths[r] + q_len)``. Every
per-row quantity is therefore just that row's length.
"""

import torch

QUERY_BLOCKS = ("causal", "diagonal")


def _neg(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def build_4d_mask(
    lengths: torch.Tensor,
    q_len: int,
    key_len: int,
    dtype: torch.dtype,
    block: str = "causal",
) -> torch.Tensor:
    """``[B, 1, q_len, key_len]`` additive mask.

    Args:
        lengths: ``[B]`` valid cached tokens per row, i.e. where this row's new queries
            start.
        q_len: number of queries appended in this forward.
        key_len: width of the key axis the attention will see (cached region plus the
            new queries; rows shorter than the widest one are masked off).
        block: how the appended queries relate to each other.

            - ``"causal"``: they are consecutive real tokens, so query ``i`` sees keys
              ``[0, lengths[b] + i]``. Used for prefill and catch-up.
            - ``"diagonal"``: they are competing candidates for the *same* next slot, so
              query ``i`` sees the cached prefix plus only its own key. Used by the V
              head, which must score k alternatives without letting them see each other.
    """
    if block not in QUERY_BLOCKS:
        raise ValueError(f"query block must be one of {QUERY_BLOCKS}, got {block!r}")

    device = lengths.device
    keys = torch.arange(key_len, device=device).view(1, 1, key_len)
    queries = torch.arange(q_len, device=device).view(1, q_len, 1)
    starts = lengths.view(-1, 1, 1)

    cached = keys < starts
    if block == "causal":
        visible = keys <= starts + queries
    else:
        visible = cached | (keys == starts + queries)

    mask = torch.where(visible, 0.0, _neg(dtype)).to(dtype)
    return mask.unsqueeze(1)


def position_ids_for(lengths: torch.Tensor, q_len: int, same_position: bool = False) -> torch.Tensor:
    """``[B, q_len]`` positions for the appended queries.

    Row ``r`` has consumed ``lengths[r]`` tokens, so its next token sits at position
    ``lengths[r]``. With ``same_position`` every query shares that position — correct for
    V-head candidates, which are competing alternatives for one slot rather than a
    sequence.
    """
    base = lengths.unsqueeze(1)
    if same_position:
        return base.expand(-1, q_len)
    return base + torch.arange(q_len, device=lengths.device).unsqueeze(0)
