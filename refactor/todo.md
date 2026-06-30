# PITA Refactor — Status & Continuation Guide

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

## System Prompt for New Chat Session

```
You are continuing implementation of the PITA standalone package refactor.

CONTEXT:
- All code lives in /scratch/user/saratb_tamu.edu/research/pita/refactor/
- The assignment spec is at /scratch/user/saratb_tamu.edu/research/pita/assignment.md
- The status file is at /scratch/user/saratb_tamu.edu/research/pita/refactor/todo.md
- Legacy code (untouched) is in /scratch/user/saratb_tamu.edu/research/pita/math_reasoning/
- Conda env: qsharp (Python 3.12). Deps (hydra-core, omegaconf, ray) installed; `pip install -e .` done.
- Tasks 0-9 are COMPLETE. All source files are written.
- Task 10 validation: import smoke tests + CLI Hydra wiring are VERIFIED/PASSED.
- Remaining: GPU small-shard end-to-end parity test, then git add.

YOUR JOB:
1. Read refactor/todo.md for full status.
2. Ensure you are on a GPU node (NOT a login node). Activate conda env qsharp.
3. Run the small-shard parity test (cli.generate, then a tiny train + eval).
4. Fix any runtime errors that arise.
5. Stage changes with git add refactor/ when the parity test passes.

RULES:
- All new code goes inside refactor/ only. Legacy code stays untouched.
- Minimize code changes. Reuse existing patterns.
- No placeholder code. No unnecessary comments.
- Use tqdm for long operations.
- All important parameters flow through Hydra configs.
```
