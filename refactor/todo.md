# PITA Refactor — Status & Continuation Guide

**STATUS: Tasks 0–10 done** (GPU end-to-end parity test passed: `generate → train → eval`).
**Task 11 (Qwen/Llama classifier generalization) — DONE in code, GPU re-verify pending.**
Test artifacts under `refactor/outputs/` are gitignored.

## What Has Been Done (Tasks 0–9 Complete)

All new code lives inside `/scratch/user/saratb_tamu.edu/research/pita/refactor/`.

### Files Created

```
refactor/
├── pyproject.toml
├── configs/
│   ├── generate.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── dataset/
│       ├── gsm8k.yaml
│       ├── math.yaml
│       ├── hh_rlhf.yaml
│       └── alpaca_pref.yaml
├── datasets/
│   ├── __init__.py
│   ├── base.py          — Example, GenerationTask, DatasetSpec protocol
│   ├── arithmetic.py    — GSM8K/MATH loader (from collect_training_data.py)
│   ├── preference.py    — HH-RLHF/AlpacaEval loader (from collect_training_data_alpaca.py)
│   └── prepare.py       — AlpacaFarm split prep (from prepare_alpaca_noisy_multi_preference_split.py)
├── generation/
│   ├── __init__.py
│   ├── collector.py     — Model loading + shared helpers
│   ├── strategies.py    — GuidedGenerate, GuidedPairwise, OfflinePairs
│   ├── worker.py        — Ray GPU actor
│   └── driver.py        — Orchestrator (Ray dispatch, parquet merge)
├── models/
│   ├── __init__.py
│   ├── classifier.py    — CustomLlamaForSequenceClassification (from my_alpaca_eval_code/classifier.py)
│   └── guidance.py      — CustomValueGuidedLogitProcessor + generate_with_classifier_guidance
├── training/
│   ├── __init__.py
│   ├── dataset.py       — All data utils, collate, metrics, I/O, perplexity_with_classifier_guidance
│   ├── builder.py       — Unified example builder (arithmetic + preference paths)
│   └── trainer.py       — Accelerate training loop
├── eval/
│   ├── __init__.py
│   ├── arithmetic.py    — GSM8K/MATH eval (generation + scoring + KL)
│   └── preference.py    — HH-RLHF PPL eval + AlpacaEval gen eval
├── scoring/
│   ├── __init__.py
│   └── arithmetic.py    — Math answer verification (from accuracy_utils.py)
└── cli/
    ├── __init__.py
    ├── generate.py      — python -m cli.generate dataset=gsm8k
    ├── train.py         — python -m cli.train dataset=hh_rlhf
    └── eval.py          — python -m cli.eval dataset=hh_rlhf
```

### Smoke Tests Passed (with conda env `qsharp` active)

- `from scoring.arithmetic import quick_evaluate_single` ✓
- `from models.classifier import CustomLlamaForSequenceClassification` ✓
- `from models.guidance import CustomValueGuidedLogitProcessor` ✓
- `from training.dataset import create_classifier_data, CustomClassifierDataset` ✓
- `from datasets.base import DatasetSpec, GenerationTask` ✓

### Full Import Smoke Tests — PASSED (deps installed, `pip install -e .` done)

- `from generation.driver import run_generation` ✓
- `from training.trainer import run_training` ✓
- `from eval.arithmetic import run_eval` ✓
- `from eval.preference import run_eval` ✓

### CLI Hydra Wiring — VERIFIED

- `python -m cli.generate --help`, `cli.train --help`, `cli.eval --help` all resolve configs ✓
- `dataset={gsm8k,math,hh_rlhf,alpaca_pref}` overrides resolve correctly via `--cfg job`:
  - gsm8k/math → family=arithmetic, eval_type=arithmetic, strategy=guided
  - hh_rlhf → family=preference, eval_type=preference_ppl, strategy=offline_pairs
  - alpaca_pref → family=preference, eval_type=preference_gen, strategy=offline_pairs

---

## Task 10 — COMPLETE (GPU end-to-end parity test PASSED)

Ran on GPU node `dgx020` (H200), env `qsharp`. Full `generate → train → eval`
pipeline verified on a tiny HH-RLHF subset (`outputs/tiny_hh.json`):

```bash
# generate (offline pairs; num_repeats=1 for preference data)
python -m cli.generate dataset=hh_rlhf dataset.data_path=$PWD/outputs/tiny_hh.json \
  generation.num_repeats=1 trainer.n_gpus_per_node=1 trainer.output_dir=outputs/test_hh
# train (1 epoch, tiny batch)
python -m cli.train dataset=hh_rlhf data.train_file=$PWD/outputs/test_hh/train_data.parquet \
  data.train_eval_save_path=$PWD/outputs/tiny_hh.json training.batch_size=4 \
  training.num_epochs=1 training.eval_freq=500 trainer.output_dir=outputs/test_hh/train
# eval (preference PPL win rate)
python -m cli.eval dataset=hh_rlhf dataset.data_path=$PWD/outputs/tiny_hh_eval.json \
  eval.classifier_ckpt=$PWD/outputs/test_hh/train/ckpt_0 models.eta=1.0 \
  eval.num_samples=1 eval.batch_size=4 trainer.output_dir=outputs/test_hh/eval
```

### Runtime fixes applied during parity test

- `generation/worker.py`: Ray passes cfg as a plain dict; restore it to a
  `DictConfig` (`OmegaConf.create`) so `cfg.dataset.*` attribute access works.
- `datasets/preference.py`: add `'problem'` (= prompt text) to `Example.extra`;
  the training builder keys on it.
- `training/dataset.py` + `training/builder.py`: generation writes parquet, so
  read it via new `read_parquet_records` (converts numpy arrays back to native
  lists) instead of `read_jsonl`.
- `training/trainer.py` + `eval/{preference,arithmetic}.py`: classifier
  `num_labels` must equal the model vocab size (`config.vocab_size`, 151936),
  not `len(tokenizer)` (151665) — the guidance processor indexes ref-model
  top-k token ids into `score`. Eval also now reads model params from the
  nested `models`/`dataset` sections of the saved `args.json`.

Test artifacts live under `refactor/outputs/` (gitignored).

---

## Task 11 — Qwen/Llama classifier generalization

The value classifier now supports both Llama and Qwen backbones, selected by a
new Hydra field `models.classifier_arch` (`llama` | `qwen`). Previously the only
class was `CustomLlamaForSequenceClassification` (a `LlamaModel`), so the default
Qwen configs silently dropped Qwen's attention q/k/v biases.

- [x] **Stage 1 — `models/classifier.py`**: shared `_ValueClassifierMixin`
  (head/score/atoms init, embeddings, `zero_init_classifier`, `calculate_loss`,
  `calculate_predictions`, `forward`) with backbone built via
  `AutoModel.from_config(config)`. Two thin subclasses,
  `CustomLlamaForSequenceClassification` and `CustomQwen2ForSequenceClassification`,
  differ only in base class and the per-class `_prepare_mask` staticmethod (the
  llama vs qwen2 `_prepare_4d_causal_attention_mask_with_cache_position`, used only
  in the `V`-type branch). Factory `get_classifier_class(arch)` maps
  `llama`→Llama, `qwen`/`qwen2`→Qwen.
- [x] **Stage 2 — configs**: `models.classifier_arch: qwen` in `generate.yaml` and
  `train.yaml`; `classifier_arch: null` in `eval.yaml` (resolved from saved
  `args.json`).
- [x] **Stage 3 — wiring**: `generation/collector.py`, `training/trainer.py`,
  `eval/preference.py`, `eval/arithmetic.py` now call
  `get_classifier_class(arch).from_pretrained(...)`. The two eval files resolve
  `classifier_arch` from `args.json` via `resolve_dict_value` like the other model
  params.
- [x] Import smoke test passed (`get_classifier_class('qwen'|'llama')`).
- [ ] **GPU re-verify** (env `qsharp`): re-run the Task 10 tiny HH-RLHF
  train/eval with `models.classifier_arch=qwen` and confirm the Qwen classifier
  loads with no unexpected-key warnings about dropped `*.bias` weights. Also try
  `classifier_arch=llama` with a Llama checkpoint.

---

## System Prompt for New Chat Session

```
You are working on the PITA standalone package refactor.

CONTEXT:
- All code lives in /scratch/user/saratb_tamu.edu/research/pita/refactor/
- The status file is at /scratch/user/saratb_tamu.edu/research/pita/refactor/todo.md
- Legacy code (untouched) is in /scratch/user/saratb_tamu.edu/research/pita/math_reasoning/
- Conda env: qsharp (Python 3.12). Deps installed; `pip install -e .` done (run from refactor/).
- Tasks 0-10 done (GPU end-to-end parity passed on a tiny HH-RLHF subset).
- Task 11 (Qwen/Llama classifier generalization) is DONE in code, import-smoke-tested,
  but NOT yet GPU re-verified. Changes are not committed. outputs/ is gitignored.

TASK 11 RECAP (see the "Task 11" section above for details):
- models/classifier.py: shared _ValueClassifierMixin + thin
  CustomLlamaForSequenceClassification / CustomQwen2ForSequenceClassification
  subclasses + get_classifier_class(arch) factory. Backbone via AutoModel.from_config.
- New Hydra field models.classifier_arch (qwen|llama): set in generate.yaml/train.yaml
  (qwen), null in eval.yaml (resolved from saved args.json).
- All 4 classifier load sites (collector/trainer/eval.preference/eval.arithmetic)
  use get_classifier_class(arch).from_pretrained(...).

RUNNING (need a GPU node, NOT a login node `slogin-*`; activate qsharp):
- An interactive GPU allocation may already exist (check `squeue`); run commands on
  it via `srun --jobid=<ID> --overlap bash -lc '...'`.
- Relative dataset paths in configs resolve against the repo root
  (math_reasoning/...), so either run from the repo root or override
  dataset.data_path / data.train_eval_save_path to absolute paths.
- CLIs: `python -m cli.generate dataset=<gsm8k|math|hh_rlhf|alpaca_pref> ...`,
  `python -m cli.train ...`, `python -m cli.eval ...` (run from refactor/).
- See the "Task 10" section above for the exact verified parity commands.

LIKELY NEXT STEPS (only if asked):
- GPU re-verify Task 11: re-run the Task 10 tiny HH-RLHF train/eval with
  models.classifier_arch=qwen; confirm no unexpected-key (*.bias) warnings when the
  Qwen classifier loads. Then sanity-check classifier_arch=llama with a Llama ckpt.
- Commit the staged changes.
- Run full-scale generate/train/eval on real (non-subset) data.
- Wire up the arithmetic (GSM8K/MATH) path end-to-end on GPU (only the preference
  path was parity-tested).

RULES:
- All new code goes inside refactor/ only. Legacy code stays untouched.
- Minimize code changes. Reuse existing patterns.
- No placeholder code. No unnecessary comments.
- Use tqdm for long operations.
- All important parameters flow through Hydra configs.
```
