"""PITA value classifier.

Derived from ``refactor_old/models/classifier.py``. The original defined one
``Custom<Arch>ForSequenceClassification`` subclass per model family behind an
``_ARCH_TO_CLS`` registry, so every new family needed new code and the V head reached
into a private, per-architecture attention-mask helper.

Here the backbone is *composed* rather than subclassed: any ``AutoModel`` works, masks
come from :mod:`pita.masking`, and adding a model family is a YAML file.

Two head types, both predicting "will this continuation be preferred":

``Q`` (default)
    ``score: Linear(hidden, ref_vocab)``. One forward gives a value for every candidate
    next token, so guidance costs a single 1-token step and top-k is a gather.
``V``
    ``score: Linear(hidden, 1)``. Scoring k candidates needs the backbone run over k
    appended positions, i.e. ~k times the compute per decode step.

Three losses: ``bce`` (default), ``mse``, and ``mle`` (distributional, over ``num_atoms``
atoms spanning ``[V_min, V_max]``).
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

HEAD_TYPES = ("Q", "V")
LOSS_TYPES = ("bce", "mse", "mle")
HEAD_CONFIG_FILE = "value_head.json"
HEAD_WEIGHTS_FILE = "value_head.safetensors"


def validate_pair(ref_model_id: str, classifier_model_id: str) -> int:
    """Assert ref and classifier share a tokenizer; return the ref vocab width.

    Guidance adds offsets to reference logits indexed by *reference* token ids, and the
    Q head is indexed by those same ids, so the two models must agree on the vocabulary.
    The upstream check compared only ``len(tokenizer)``, which passes for genuinely
    different vocabularies of equal size; compare the actual token->id maps instead.

    The returned width is the reference model's ``config.vocab_size``, not
    ``len(tokenizer)``. They differ (Qwen2.5: 151936 vs 151665) and the logits the
    processor indexes into are ``config.vocab_size`` wide.
    """
    ref_vocab = AutoTokenizer.from_pretrained(ref_model_id).get_vocab()
    clf_vocab = AutoTokenizer.from_pretrained(classifier_model_id).get_vocab()
    if ref_vocab != clf_vocab:
        only_ref = set(ref_vocab) - set(clf_vocab)
        only_clf = set(clf_vocab) - set(ref_vocab)
        raise ValueError(
            f"{ref_model_id} and {classifier_model_id} do not share a tokenizer "
            f"({len(ref_vocab)} vs {len(clf_vocab)} tokens; "
            f"{len(only_ref)} only in ref, {len(only_clf)} only in classifier). "
            "PITA guidance requires a shared vocabulary."
        )

    ref_width = AutoConfig.from_pretrained(ref_model_id).vocab_size
    clf_width = AutoConfig.from_pretrained(classifier_model_id).vocab_size
    if clf_width < ref_width:
        raise ValueError(
            f"classifier vocab_size ({clf_width}) is narrower than reference "
            f"({ref_width}); the value head could not cover every reference token."
        )
    return ref_width


class ValueClassifier(nn.Module):
    def __init__(
        self,
        backbone,
        ref_vocab_size: int,
        head_type: str = "Q",
        loss_type: str = "bce",
        use_bias: bool = False,
        num_atoms: int = 11,
        V_min: float = 0.0,
        V_max: float = 1.0,
    ):
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")

        self.backbone = backbone
        self.head_type = head_type
        self.loss_type = loss_type
        self.use_bias = bool(use_bias)
        self.ref_vocab_size = ref_vocab_size
        self.num_atoms = num_atoms if loss_type == "mle" else 1
        self.V_min = V_min
        self.V_max = V_max

        slots = ref_vocab_size if head_type == "Q" else 1
        hidden = backbone.config.hidden_size
        self.score = nn.Linear(hidden, slots * self.num_atoms, bias=self.use_bias)
        self.score.to(dtype=backbone.dtype)

        if loss_type == "mse":
            self.loss_fct = MSELoss(reduction="none")
        elif loss_type == "bce":
            self.loss_fct = BCEWithLogitsLoss(reduction="none")
        else:
            self.loss_fct = None
        self.register_buffer(
            "atoms", torch.linspace(V_min, V_max, self.num_atoms).float(), persistent=False
        )

    @property
    def config(self):
        return self.backbone.config

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_backbone(cls, model_id: str, ref_vocab_size: int, dtype=None, **kwargs) -> "ValueClassifier":
        """Fresh classifier from a pretrained backbone; the value head is untrained."""
        backbone = AutoModel.from_pretrained(model_id, dtype=dtype)
        return cls(backbone, ref_vocab_size, **kwargs)

    @classmethod
    def from_pretrained(cls, path: str, dtype=None) -> "ValueClassifier":
        with open(os.path.join(path, HEAD_CONFIG_FILE)) as f:
            head_cfg = json.load(f)
        backbone = AutoModel.from_pretrained(path, dtype=dtype)
        model = cls(
            backbone,
            ref_vocab_size=head_cfg["ref_vocab_size"],
            head_type=head_cfg["head_type"],
            loss_type=head_cfg["loss_type"],
            use_bias=head_cfg["use_bias"],
            num_atoms=head_cfg["num_atoms"],
            V_min=head_cfg["V_min"],
            V_max=head_cfg["V_max"],
        )
        head_state = load_file(os.path.join(path, HEAD_WEIGHTS_FILE))
        model.score.load_state_dict(head_state)
        return model

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        save_file(
            {k: v.contiguous() for k, v in self.score.state_dict().items()},
            os.path.join(path, HEAD_WEIGHTS_FILE),
        )
        with open(os.path.join(path, HEAD_CONFIG_FILE), "w") as f:
            json.dump(
                {
                    "head_type": self.head_type,
                    "loss_type": self.loss_type,
                    "use_bias": self.use_bias,
                    "ref_vocab_size": self.ref_vocab_size,
                    "num_atoms": self.num_atoms,
                    "V_min": self.V_min,
                    "V_max": self.V_max,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------ head init

    def zero_init_head(self) -> None:
        nn.init.zeros_(self.score.weight)
        if self.use_bias:
            nn.init.zeros_(self.score.bias)

    def reuse_init_head(self, model_id: str) -> None:
        """Warm-start a Q head from the backbone's own LM head.

        ``get_output_embeddings()`` is the architecture-neutral accessor; the upstream
        code reached for ``.lm_head`` directly.
        """
        if self.head_type != "Q":
            raise ValueError("reuse init only applies to Q heads")
        donor = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.score.weight.dtype)
        output_embeddings = donor.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError(f"{model_id} exposes no output embeddings to reuse")
        weight = output_embeddings.weight.data[: self.ref_vocab_size]
        if self.loss_type == "mle":
            # Every atom of a token starts from that token's LM-head row.
            weight = weight.repeat(1, self.num_atoms).view(self.ref_vocab_size * self.num_atoms, -1)
        self.score.weight.data.copy_(weight.to(self.score.weight.device))
        del donor

    # ------------------------------------------------------------------ scoring

    def _gather_head(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Head output for selected tokens only.

        ``hidden`` is ``[..., H]`` and ``token_ids`` ``[...]`` with matching leading dims.
        Returns ``[...]`` (``[..., num_atoms]`` for ``mle``).

        Materialising the full ``[B, L, vocab]`` head output, as the upstream code did,
        costs ~2 GB at batch 8 / length 1024 / 128k vocab. Only the selected rows are
        ever used, so gather the weight rows instead; autograd scatter-adds gradients
        back to exactly those rows.
        """
        if self.loss_type == "mle":
            atom_offsets = torch.arange(self.num_atoms, device=token_ids.device)
            rows = token_ids.unsqueeze(-1) * self.num_atoms + atom_offsets
            weight = self.score.weight[rows]                       # [..., A, H]
            out = (weight * hidden.unsqueeze(-2)).sum(-1)          # [..., A]
            if self.use_bias:
                out = out + self.score.bias[rows]
            return out
        weight = self.score.weight[token_ids]                      # [..., H]
        out = (weight * hidden).sum(-1)                            # [...]
        if self.use_bias:
            out = out + self.score.bias[token_ids]
        return out

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full head output for a V head: ``[..., ]`` or ``[..., num_atoms]``."""
        out = self.score(hidden)
        if self.loss_type == "mle":
            return out
        return out.squeeze(-1)

    def score_candidates(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Raw value logits ``z`` for candidate next tokens.

        Q head: ``hidden`` is ``[B, H]`` (the last position) and ``candidate_ids``
        ``[B, k]``; the head is indexed directly.
        V head: ``hidden`` is ``[B, k, H]``, already computed by running the backbone
        over the k candidate positions; ``candidate_ids`` is unused.

        Returns ``[B, k]``, or ``[B, k, num_atoms]`` for ``mle``.
        """
        if self.head_type == "Q":
            return self._gather_head(hidden.unsqueeze(1).expand(-1, candidate_ids.shape[1], -1), candidate_ids)
        return self._head(hidden)

    # ------------------------------------------------------------------ training

    def calculate_loss(self, logits, labels, loss_weights, loss_mask):
        """Token-level loss against a single sequence-level label.

        ``logits`` ``[B, L]`` (``[B, L, A]`` for ``mle``), ``labels`` ``[B]`` in
        ``[0, 1]``, ``loss_mask`` ``[B, L]`` selecting response tokens.
        """
        seqlen = logits.shape[1]
        labels_expanded = labels.unsqueeze(1).expand(-1, seqlen)

        if self.loss_type == "mse":
            loss = self.loss_fct(torch.sigmoid(logits), labels_expanded.to(logits.dtype))
        elif self.loss_type == "bce":
            loss = self.loss_fct(logits, labels_expanded.to(logits.dtype))
        else:
            log_pmfs = F.log_softmax(logits, dim=-1)
            label_indices = torch.round(labels * (self.num_atoms - 1)).long()
            label_indices = torch.clamp(label_indices, 0, self.num_atoms - 1)
            loss = -log_pmfs[torch.arange(logits.shape[0]), :, label_indices]

        loss = (loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
        return (loss * loss_weights).mean()

    def calculate_predictions(self, logits):
        """Map raw head output to a value in ``[0, 1]``."""
        if self.loss_type in ("mse", "bce"):
            return torch.sigmoid(logits)
        pmfs = torch.softmax(logits, dim=-1)
        return (pmfs * self.atoms.to(pmfs.device)).sum(dim=-1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        loss_mask=None,
        loss_weights=None,
    ):
        """Training forward. Returns ``(loss, logits)`` with logits aligned to ``loss_mask``.

        Q head: the value of the token actually taken, i.e. ``Q(s_t, a_t)`` where
        ``a_t = input_ids[t+1]``, so logits and mask are both shifted by one.
        V head: the value of each state, ``V(s_t)``, unshifted.
        """
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if self.head_type == "Q":
            logits = self._gather_head(hidden[:, :-1], input_ids[:, 1:]).float()
            mask = loss_mask[:, 1:] if loss_mask is not None else None
        else:
            logits = self._head(hidden).float()
            mask = loss_mask

        loss = None
        if labels is not None:
            loss = self.calculate_loss(logits, labels, loss_weights, mask)
        return loss, logits
