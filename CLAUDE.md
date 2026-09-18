# Working rules for this repo

PITA: preference-guided inference-time alignment. The language model is **frozen**; all
that trains is a small value classifier that tilts its logits during decoding.

Current goal, status, and subtasks live in `Project_plan.md` — read it before starting
work. This file is the durable *how*; that file is the *what*.

## 1. `pita/` is the only writable tree

| Path | Role |
|------|------|
| `pita/` | **Product.** All new code goes here. |
| `SPPO/` | Read-only reference — infra shape (generate/rank/pipeline, accelerate configs, arg parser) |
| `refactor_old/` | Read-only reference — previous refactor: classifier, guidance math, trainer |
| `math_reasoning/` | Read-only reference — original research code; `my_alpaca_eval_code/` has the AlpacaEval writer |
| `verl-tool-lens/` | Unrelated |

Never edit a file in a reference tree. Not to fix a bug, not to add a feature, not
temporarily.

## 2. Reuse by copy, never by regeneration

To reuse functionality from a reference tree: **`cp` the file into `pita/` first, then edit
the copy.** Do not retype or regenerate a known-good module from scratch — it is slower and
it reintroduces bugs the original already solved.

Name the source in the new file's module docstring, and say what changed. Every module in
`pita/pita/` does this; follow the pattern.

## 3. Never install anything

Conda environments and package installs are the user's, always. When code needs a new
dependency, declare it (in `setup.py`, or just say so) and stop. Do not run `conda create`,
`pip install`, or equivalent.

Inspecting environments read-only is fine and often useful:

```bash
/scratch/user/saratb_tamu.edu/miniconda3/envs/pita/bin/python -c "import vllm; print(vllm.__version__)"
```

**Use the `pita` env** (`/scratch/user/saratb_tamu.edu/miniconda3/envs/pita`): vLLM 0.11.0,
torch 2.8.0, transformers 5.12.1, accelerate 1.14.0. Not SPPO's env — its `torch==2.1.2` /
`transformers==4.42.4` pins predate the vLLM V1 logits-processor API this project needs.

## 4. Keep it one designed codebase

Clean configs, one trainer, one generation path, no dead placeholders, no speculative
abstraction layers "for later". Complexity is the thing most likely to break this project:
prefer the design with fewer moving parts even when it is slightly less general.

Concretely, for this repo:

- **Model-agnostic, always.** No per-architecture subclasses, registries, or substring
  matching on model names. Compose over `AutoModel`, use `apply_chat_template`, build
  attention masks locally rather than importing private per-model helpers. Adding a model
  family must stay *one YAML file, zero code*.
- Parameters flow through YAML recipes + CLI overrides (SPPO-shaped), not Hydra.
- `tqdm` on anything long-running.

## 5. One plan doc per repo

`Project_plan.md` only. When a plan goes stale, rewrite it — do not leave a second one
beside it.

## 6. Testing

```bash
cd pita && python -m pytest tests/ -q     # 33 tests, CPU only, no GPU or network
```

Tests must stay runnable on a login node: tiny random models built from config, stub
tokenizers, no downloads.

`tests/test_guidance_cache.py` is the one that matters most. The classifier's incremental
KV cache must be numerically identical to a plain full forward — a desynced cache **does
not crash**, it silently degrades guidance and shows up only as a worse AlpacaEval number.
Any change to `guidance.py` or `masking.py` must keep it passing.

## 7. Two upstream bugs — do not reintroduce

1. **`expectation` guidance offset.** `refactor_old/models/guidance.py:84-86` (and all
   three `math_reasoning` classifier variants) clamp the *odds ratio* to `<= 1-1e-6`,
   forcing every offset non-positive so guidance can only suppress tokens, never boost
   them. The correct offset is just `eta * z`, since `sigmoid(z)/(1-sigmoid(z)) == exp(z)`.
   Every shipped `expectation` number upstream is suspect.
2. **Pad leakage.** Upstream stored the already-padded prompt row and the collator marked
   those pads as attended. Tokenize prompts unpadded; right-pad batches (left padding also
   shifts position ids silently).

## 8. Hardware

Slurm, partition `def`, 8x H200 per node. **The login node has no GPU** — anything touching
CUDA needs `srun`/`sbatch`. Check for an existing allocation with `squeue` and reuse it:

```bash
srun --jobid=<ID> --overlap bash -lc '...'
```
