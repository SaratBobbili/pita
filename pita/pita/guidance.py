"""PITA guidance as a vLLM V1 logits processor.

The offset formulas come from ``refactor_old/models/guidance.py``; the surrounding
machinery is new, because that file drove HF ``generate()`` one sequence at a time while
vLLM owns a continuously-rebatched decode loop.

Shape of the problem: the frozen reference model produces logits, and for each candidate
next token we need a value from a *second*, much smaller model that has consumed the same
token prefix. So the classifier must run in lockstep with vLLM's batch, keeping its own
KV cache across steps while requests join, finish, and get shuffled between slots.

The design that makes this tractable is to treat the cache as **pure optimization**.
Every step each row recomputes the context it *should* have consumed --
``len(prompt_tok_ids) + len(output_tok_ids)``, both handed to us by vLLM, the latter as a
live reference -- and replays whatever it is missing. A new request, a preempted and
recomputed request, and an ordinary decode step all reduce to the same "bring this row up
to date" path, so nothing here depends on vLLM's internal scheduling decisions.

Wire it up from the caller as::

    LLM(model=ref_model,
        logits_processors=[PITAGuidedLogitsProcessor],
        additional_config={"pita": {"classifier_path": ..., "eta": 1.0}})

Omit ``additional_config["pita"]`` and the processor stays inert, which is how the
unguided ``eta=0`` reference baseline runs with no classifier loaded at all.
"""

import torch
import torch.nn.functional as F
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

from pita.classifier import ValueClassifier
from pita.masking import build_4d_mask, position_ids_for

INFERENCE_MODES = ("expectation", "bernoulli")


def log1p_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, torch.zeros_like(x))


class BankCache:
    """Batched KV cache for the classifier, one row per vLLM batch slot.

    Left-aligned: row ``r`` occupies key positions ``[0, lengths[r])``. Rows are
    permuted to mirror vLLM's own slot moves, so the active rows always form the
    contiguous prefix ``[0, num_reqs)`` and the steady-state decode step reads a plain
    slice -- a view, never a copy.

    Every forward writes its keys into the bank, including the V head's hypothetical
    candidates: attention can only see a key that is actually present, and candidates
    land in scratch slots at ``[lengths[r], lengths[r] + k)`` that no mask exposes once
    the pass ends. Leaving ``lengths`` untouched is what makes them hypothetical -- the
    next real decode step overwrites the first such slot, and later steps the rest.
    Capacity therefore reserves ``top_k`` slots beyond ``max_model_len``.

    Implements just enough of the transformers ``Cache`` protocol for the backbone to
    call ``update()``; it deliberately places keys at per-row positions itself rather
    than relying on a batch-wide ``cache_position``.
    """

    def __init__(self, num_layers, max_rows, num_kv_heads, capacity, head_dim, device, dtype):
        self.capacity = capacity
        self.max_rows = max_rows
        shape = (max_rows, num_kv_heads, capacity, head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.lengths = torch.zeros(max_rows, dtype=torch.long, device=device)
        # Set per forward by the processor.
        self.rows = None        # [B] bank row indices this forward writes to
        self.write_pos = None   # [B] first key position each row writes at
        self.key_len = 0        # width of the key axis returned to attention
        self.detached = False   # True during prefill: return only the new keys

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.key_len)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        q_len = key_states.shape[2]
        rows = self.rows.view(-1, 1).expand(-1, q_len)
        cols = self.write_pos.view(-1, 1) + torch.arange(q_len, device=key_states.device)
        # bank[row, :, col] <- [B, q, H, D]
        self.keys[layer_idx][rows, :, cols] = key_states.permute(0, 2, 1, 3)
        self.values[layer_idx][rows, :, cols] = value_states.permute(0, 2, 1, 3)
        if self.detached:
            # Prefill: these rows had no prior content, so the new keys are the whole
            # visible context and no gather off the bank is needed.
            return key_states, value_states
        n = self.rows.shape[0]
        return self.keys[layer_idx][:n, :, : self.key_len], self.values[layer_idx][:n, :, : self.key_len]

    def reorder(self, src: int, dst: int, directionality) -> None:
        for tensors in (self.keys, self.values):
            for t in tensors:
                if directionality == MoveDirectionality.SWAP:
                    tmp = t[src].clone()
                    t[src] = t[dst]
                    t[dst] = tmp
                else:
                    t[dst] = t[src]
        if directionality == MoveDirectionality.SWAP:
            self.lengths[src], self.lengths[dst] = self.lengths[dst].clone(), self.lengths[src].clone()
        else:
            self.lengths[dst] = self.lengths[src]


class _Row:
    __slots__ = ("prompt_ids", "output_ids", "eta")

    def __init__(self, prompt_ids, output_ids, eta):
        self.prompt_ids = prompt_ids or []
        self.output_ids = output_ids
        self.eta = eta

    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def token_at(self, i: int) -> int:
        n = len(self.prompt_ids)
        return self.prompt_ids[i] if i < n else self.output_ids[i - n]

    def tokens(self, stop: int):
        n = len(self.prompt_ids)
        if stop <= n:
            return list(self.prompt_ids[:stop])
        return list(self.prompt_ids) + list(self.output_ids[: stop - n])


class PITAGuidedLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        extra = sampling_params.extra_args or {}
        eta = extra.get("eta")
        if eta is not None and not isinstance(eta, (int, float)):
            raise ValueError(f"pita eta must be a number, got {eta!r}")

    def __init__(self, vllm_config, device, is_pin_memory):
        cfg = (vllm_config.additional_config or {}).get("pita")
        self.enabled = bool(cfg)
        self.rows: dict[int, _Row] = {}
        if not self.enabled:
            return

        self.device = device
        self.default_eta = float(cfg.get("eta", 1.0))
        self.top_k = int(cfg.get("top_k", 20))
        self.inference_mode = cfg.get("inference_mode", "expectation")
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"inference_mode must be one of {INFERENCE_MODES}")
        self.cd_baseline = bool(cfg.get("cd_baseline", False))

        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        self.classifier = ValueClassifier.from_pretrained(cfg["classifier_path"], dtype=dtype)
        # SDPA accepts the arbitrary additive masks we build; flash-attn does not.
        self.classifier.backbone.set_attn_implementation("sdpa")
        self.classifier.to(device).eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

        config = self.classifier.config
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.bank = BankCache(
            num_layers=config.num_hidden_layers,
            max_rows=vllm_config.scheduler_config.max_num_seqs,
            num_kv_heads=num_kv_heads,
            capacity=vllm_config.model_config.max_model_len + max(self.top_k, 1),
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    def is_argmax_invariant(self) -> bool:
        return False

    # ------------------------------------------------------------------ batch state

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not self.enabled or batch_update is None:
            return
        # Order is mandated by vLLM: removed, then added, then moved.
        for index in batch_update.removed:
            self.rows.pop(index, None)

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            extra = (params.extra_args or {}) if params is not None else {}
            eta = float(extra.get("eta", self.default_eta))
            self.rows[index] = _Row(prompt_tok_ids, output_tok_ids, eta)
            self.bank.lengths[index] = 0

        for src, dst, directionality in batch_update.moved:
            a, b = self.rows.pop(src, None), self.rows.pop(dst, None)
            if a is not None:
                self.rows[dst] = a
            if directionality == MoveDirectionality.SWAP and b is not None:
                self.rows[src] = b
            self.bank.reorder(src, dst, directionality)

    # ------------------------------------------------------------------ classifier

    def _prefill(self, indices, targets):
        """Bring rows up to ``targets`` tokens from scratch. Returns nothing."""
        widest = max(targets)
        pad = 0
        padded = []
        for row_index, target in zip(indices, targets):
            toks = self.rows[row_index].tokens(target)
            padded.append(toks + [pad] * (widest - target))
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        rows = torch.tensor(indices, dtype=torch.long, device=self.device)
        starts = torch.zeros(len(indices), dtype=torch.long, device=self.device)

        self.bank.rows, self.bank.write_pos = rows, starts
        self.bank.key_len, self.bank.detached = widest, True
        mask = build_4d_mask(starts, widest, widest, self.classifier.backbone.dtype)
        self.classifier.backbone(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids_for(starts, widest),
            past_key_values=self.bank,
            use_cache=True,
        )
        self.bank.detached = False
        for row_index, target in zip(indices, targets):
            self.bank.lengths[row_index] = target

    def _decode_step(self, num_reqs, tokens):
        """Append one token to every active row; return ``[num_reqs, H]`` last hidden."""
        rows = torch.arange(num_reqs, device=self.device)
        starts = self.bank.lengths[:num_reqs]
        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)

        key_len = int(starts.max().item()) + 1
        self.bank.rows, self.bank.write_pos = rows, starts
        self.bank.key_len, self.bank.detached = key_len, False
        mask = build_4d_mask(starts, 1, key_len, self.classifier.backbone.dtype)
        out = self.classifier.backbone(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids_for(starts, 1),
            past_key_values=self.bank,
            use_cache=True,
        )
        self.bank.lengths[:num_reqs] += 1
        return out.last_hidden_state[:, -1]

    def _candidate_hidden(self, num_reqs, candidate_ids):
        """V head: hidden states for k hypothetical next tokens, without caching them."""
        k = candidate_ids.shape[1]
        rows = torch.arange(num_reqs, device=self.device)
        starts = self.bank.lengths[:num_reqs]
        key_len = int(starts.max().item()) + k

        self.bank.rows, self.bank.write_pos = rows, starts
        self.bank.key_len, self.bank.detached = key_len, False
        mask = build_4d_mask(starts, k, key_len, self.classifier.backbone.dtype, block="diagonal")
        out = self.classifier.backbone(
            input_ids=candidate_ids,
            attention_mask=mask,
            position_ids=position_ids_for(starts, k, same_position=True),
            past_key_values=self.bank,
            use_cache=True,
        )
        return out.last_hidden_state

    def _offsets(self, z, eta):
        """Logit offsets from raw classifier values ``z``; ``eta`` is ``[B, 1]``."""
        if self.classifier.loss_type == "mle":
            log_pmfs = F.log_softmax(z, dim=-1)
            atoms = self.classifier.atoms.to(log_pmfs.device)
            offset = torch.logsumexp(log_pmfs + eta.unsqueeze(-1) * atoms, dim=-1)
            return offset - offset.min(dim=-1, keepdim=True).values
        if self.cd_baseline:
            return eta * torch.sigmoid(z)
        if self.inference_mode == "expectation":
            # log(sigmoid(z) / (1 - sigmoid(z))) == z, exactly. The upstream code spelled
            # this out and then clamped the *odds ratio* to <= 1-1e-6, which forced every
            # offset non-positive so guidance could only ever suppress tokens.
            return eta * z
        return log1p_exp(eta + z) - log1p_exp(z)

    # ------------------------------------------------------------------ entry point

    @torch.inference_mode()
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.rows:
            return logits
        num_reqs = logits.shape[0]
        missing = [i for i in range(num_reqs) if i not in self.rows]
        if missing:
            raise RuntimeError(f"pita: untracked batch slots {missing[:8]}")

        # Each row should have consumed every token but the newest; the shared decode
        # step below appends that one. Anything further behind is replayed from scratch,
        # which covers fresh requests and post-preemption recompute alike.
        contexts = [self.rows[i].context_len() for i in range(num_reqs)]
        stale = [i for i in range(num_reqs) if int(self.bank.lengths[i]) != contexts[i] - 1]
        if stale:
            self._prefill(stale, [contexts[i] - 1 for i in stale])

        hidden = self._decode_step(num_reqs, [self.rows[i].token_at(contexts[i] - 1) for i in range(num_reqs)])

        if self.top_k > 0:
            candidates = torch.topk(logits, min(self.top_k, logits.shape[-1]), dim=-1).indices
        else:
            candidates = torch.arange(logits.shape[-1], device=logits.device).expand(num_reqs, -1)

        if self.classifier.head_type == "V":
            z = self.classifier.score_candidates(self._candidate_hidden(num_reqs, candidates))
        else:
            z = self.classifier.score_candidates(hidden, candidates)

        eta = torch.tensor(
            [self.rows[i].eta for i in range(num_reqs)], device=logits.device, dtype=torch.float32
        ).unsqueeze(1)
        offsets = self._offsets(z.float(), eta)
        return logits.scatter_add_(1, candidates, offsets.to(logits.dtype))
