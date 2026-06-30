# PITA Standalone Package — Implementation Assignment

## Context

**PITA** (Preference Guided Inference Time Alignment) is a research framework for training classifiers that guide language model decoding toward preferred outputs. The current implementation lives as scattered scripts under `math_reasoning/` with duplicated code across `Jeremy/`, `my_alpaca_eval_code/`, and the root. This assignment builds a clean, flat Python package at the repo root that unifies all four datasets (GSM8K, MATH, HH-RLHF, AlpacaEval) behind three CLI entrypoints:

```bash
python -m cli.generate dataset=gsm8k        # Ray-sharded generation → parquet
python -m cli.train   dataset=hh_rlhf ...   # Accelerate DDP training
python -m cli.eval    dataset=hh_rlhf ...   # Dataset-specific evaluation
```

**Legacy code stays untouched** — new modules sit alongside `math_reasoning/`, `imdb_gen/`, `star_graph/`.

### Target layout

```
/scratch/user/saratb_tamu.edu/research/pita/
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
│   ├── base.py
│   ├── arithmetic.py
│   ├── preference.py
│   └── prepare.py
├── generation/
│   ├── __init__.py
│   ├── collector.py
│   ├── strategies.py
│   ├── worker.py
│   └── driver.py
├── models/
│   ├── __init__.py
│   ├── classifier.py
│   └── guidance.py
├── training/
│   ├── __init__.py
│   ├── builder.py
│   ├── dataset.py
│   └── trainer.py
├── eval/
│   ├── __init__.py
│   ├── arithmetic.py
│   └── preference.py
├── scoring/
│   ├── __init__.py
│   └── arithmetic.py
├── cli/
│   ├── __init__.py
│   ├── generate.py
│   ├── train.py
│   └── eval.py
├── math_reasoning/   # legacy, untouched
├── imdb_gen/
└── star_graph/
```

---

## Task 0: Environment Setup

### 0.1 — Base conda environment

The existing env is defined in `pita.yml` (env name `qsharp`, Python 3.12). Activate it or recreate:

```bash
conda env create -f /scratch/user/saratb_tamu.edu/research/pita/pita.yml
conda activate qsharp
```

Key packages already present: `torch 2.4.1`, `transformers 4.45.2`, `accelerate 1.6.0`, `datasets 3.0.2`, `pandas 2.2.3`, `pyarrow 19.0.1`, `wandb 0.19.9`, `math-verify 0.7.0`, `scikit-learn 1.6.1`, `tqdm`.

### 0.2 — Install new dependencies

The standalone package needs three additional libraries not in `pita.yml`:

```bash
pip install hydra-core omegaconf "ray[default]"
```

- **Hydra** — composable YAML config system. Docs: https://hydra.cc/docs/intro/
- **OmegaConf** — config container used by Hydra. Docs: https://omegaconf.readthedocs.io/
- **Ray** — distributed task execution for generation. Docs: https://docs.ray.io/en/latest/

### 0.3 — Create `pyproject.toml`

Create `pyproject.toml` at repo root. Use `setuptools` with explicit package includes so legacy dirs are excluded:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pita"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.4",
    "transformers>=4.45",
    "accelerate>=1.6",
    "datasets>=3.0",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "ray[default]>=2.9",
    "pyarrow>=19.0",
    "pandas>=2.2",
    "wandb>=0.19",
    "tqdm",
    "numpy>=2.2",
    "scikit-learn>=1.6",
    "math-verify>=0.7",
]

[tool.setuptools.packages.find]
include = ["datasets*", "generation*", "models*", "training*", "eval*", "scoring*", "cli*", "configs*"]
exclude = ["math_reasoning*", "imdb_gen*", "star_graph*", "refactor*"]
```

Then install in dev mode:

```bash
cd /scratch/user/saratb_tamu.edu/research/pita
pip install -e .
```

### 0.4 — Create all `__init__.py` files

Create empty `__init__.py` in each new package directory: `datasets/`, `generation/`, `models/`, `training/`, `eval/`, `scoring/`, `cli/`.

---

## Task 1: Scoring Utilities — `scoring/arithmetic.py`

**Port from:** `math_reasoning/accuracy_utils.py`

This is a leaf module with no internal dependencies — build it first.

### What to port

Copy over all functions from `math_reasoning/accuracy_utils.py`:

| Function | Purpose |
|----------|---------|
| `numeric_or_symbolic_correctness(prediction, answer)` | Check if prediction matches answer numerically or symbolically |
| `find_boxed_content(s, last_occurrence)` | Extract content from `\boxed{}` LaTeX |
| `extract_between_and_with_boxes(x, last_occurrence=False)` | Extract answer from LaTeX-formatted string |
| `split_answer_separator(text, separator=None)` | Split answer at separator token |
| `equivalence_partition(iterable, relation)` | Partition items by equivalence relation |
| `compute_majority_vote_correct(processed_predictions, predictions_correctness, predictions_partition, strict_tie_breaking=True, partition_weights=None)` | Majority vote accuracy |
| `process_sample(sample, few_shot_separator, extract_last_occurrence)` | Extract prediction from raw model output |
| `sample_match_strict(sample, reference)` | Strict string match |
| `quick_evaluate_single(dataset_type, solution_or_answer, few_shot_separator, extract_last_occurrence, match_fn, raw_prediction)` | Single-sample correctness check |
| `evaluate_preference(dataset_type, solution_or_answer, few_shot_separator, extract_last_occurrence, match_fn, raw_prediction1, raw_prediction2, soft_pref)` | Pairwise preference evaluation |

**External deps:** `math_verify` (for symbolic math equivalence checks).

### Verification

```python
from scoring.arithmetic import quick_evaluate_single, sample_match_strict
```

---

## Task 2: Model Layer — `models/classifier.py` and `models/guidance.py`

**Port from:** `math_reasoning/my_alpaca_eval_code/classifier.py` (271 lines)

This module has two classes and one helper. Split them across two files for clarity.

### 2.1 — `models/classifier.py`: The classifier model

Port `CustomLlamaForSequenceClassification(LlamaPreTrainedModel)`.

**Key signatures:**

```python
class CustomLlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config, loss_type, use_bias, classifier_type, *,
                 num_atoms=11, V_min=0.0, V_max=1.0):
        # classifier_type: "Q" or "V"
        # loss_type: "mse", "bce", or "mle"
        # Head sizing:
        #   mse/bce + Q → Linear(hidden_size, num_labels)
        #   mse/bce + V → Linear(hidden_size, 1)
        #   mle + Q → Linear(hidden_size, num_labels * num_atoms)
        #   mle + V → Linear(hidden_size, num_atoms)

    def zero_init_classifier(self): ...
    def calculate_loss(self, logits, labels, loss_weights, loss_mask): ...
    def calculate_predictions(self, logits): ...

    def forward(self,
        input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None,
        loss_weights=None, logit_indices=None, loss_mask=None,
        use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ) -> SequenceClassifierOutputWithPast: ...
```

**Key imports from transformers:**
- `LlamaPreTrainedModel`, `LlamaModel`
- `SequenceClassifierOutputWithPast`
- `Cache` (from `transformers.cache_utils`)
- `LogitsProcessor`

**Forward behavior branches:**
- Train Q: `labels` provided → indexed token logits → loss
- Infer Q: `labels=None` → logits + past_key_values
- Train V: `labels` provided, `loss_mask` required, `logit_indices=None`
- Infer V: `labels=None`, `logit_indices` required for top-k expansion

### 2.2 — `models/guidance.py`: Logit processor + generation helper

Port `CustomValueGuidedLogitProcessor(LogitsProcessor)` and the standalone `log1p_exp` helper.

**Key signatures:**

```python
def log1p_exp(x):
    return torch.logaddexp(x, torch.tensor(0.0).to(x.device))

class CustomValueGuidedLogitProcessor(LogitsProcessor):
    def __init__(self, eta, ref_model, ref_model_tokenizer,
                 value_classifier, inference_mode, top_k,
                 cd_baseline=0, use_cache=True):
        # inference_mode: "expectation", "bernoulli", or "disabled"
        # Maintains classifier_state dict with keys:
        #   input_ids, attention_mask, use_cache, past_key_values, first_pass

    def reset_classifier_state(self): ...

    def get_classifier_values(self, input_ids, top_k_indices):
        # Calls value_classifier(input_ids=..., logit_indices=top_k_indices, ...)

    def modify_top_k_logits(self, ref_model_logits, logit_offset, top_k_indices):
        # torch.scatter_add on ref_model_logits

    def __call__(self, input_ids, ref_model_logits):
        # 1. Pick top_k_indices (all vocab if top_k == -1)
        # 2. Get classifier values
        # 3. Compute logit_offset based on inference_mode:
        #    - mle: log-sum-exp over atoms with eta * atoms
        #    - expectation: sigmoid / log-odds offset
        #    - bernoulli: sigmoid or log1p_exp ratio
        # 4. Apply via modify_top_k_logits
```

Also port `generate_with_classifier_guidance` into this file (from `math_reasoning/my_alpaca_eval_code/utils.py`):

```python
def generate_with_classifier_guidance(ref_model, tokenizer, logit_processor,
                                       inputs, generate_kwargs,
                                       return_output_only, return_text, eta):
    # If eta != 0: reset logit_processor, generate with processor list
    # If eta == 0: generate without processor
    # Optionally strip prompt tokens / decode to text
```

### Verification

```python
from models.classifier import CustomLlamaForSequenceClassification
from models.guidance import CustomValueGuidedLogitProcessor, generate_with_classifier_guidance
```

---

## Task 3: Training Data Utilities — `training/dataset.py`

**Port from:** `math_reasoning/my_alpaca_eval_code/utils.py` (data-handling parts)

### What to port

**Functions:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `tokenize_with_chat_template` | `(tokenizer, prompts, use_chat_template, device)` | Chat-template tokenization; returns `(inputs, formatted_prompts)` |
| `get_output_indices` | `(outputs, eos_token_id)` | Find first EOS position per sequence |
| `create_classifier_data` | `(all_data, use_all_ref_tokens, max_length=None)` | Flatten roll-in/roll-out token pairs into classifier training dicts |
| `custom_collate_fn` | `(batch: list[dict], pad_token_id: int)` | Left-pad variable-length sequences into batch with attention/loss masks |

**Classes:**

| Class | Purpose |
|-------|---------|
| `CustomClassifierDataset(Dataset)` | Wraps `{input_ids, target_ids, rewards, loss_weights}` dict |
| `DynamicBatchSampler(Sampler)` | Groups by max batch size and max padded tokens |

**`create_classifier_data` input contract** (`all_data` is a list of dicts):

| Key | Type |
|-----|------|
| `prompt_tokenized` | `list[list[int]]` — roll-in prefixes |
| `response_tokenized` | `list[list[int]]` — roll-out continuations |
| `reward` | `list[scalar]` — per-pair reward |

**`create_classifier_data` output:**

```python
{
    'input_ids': list[list[int]],
    'target_ids': list[list[int]],
    'rewards': list[float],
    'loss_weights': list[float],  # always 1.0 currently
}
```

**`custom_collate_fn` output:**

```python
{
    'input_ids': Tensor[B, max_len],       # left-padded
    'attention_mask': Tensor[B, max_len],  # bool
    'loss_mask': Tensor[B, max_len],       # bool, True only on target tokens
    'rewards': Tensor[B],
    'loss_weights': Tensor[B],
}
```

### Also port these metric helpers (used by trainer)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `calculate_explained_variance` | `(predictions, labels)` | `1 - var(pred-label)/var(label)` |
| `calculate_r2` | `(predictions, labels)` | Standard R² |
| `calculate_mle_stats` | `(logits, atoms)` | Expected value, variance, entropy from distributional logits |
| `kl_divergence` | `(logits1, logits2)` | Per-(batch,seq) KL divergence |
| `get_average_reward` | `(all_data, eval_key, simulation_rounds)` | Monte Carlo reward sampling |

**Class used by `calculate_mle_stats`:**

```python
class CategoricalDistributionRL:
    def __init__(self, atoms, logits):  # applies softmax to logits
    def expected_value(self): ...       # sum(pmf * atoms)
    def variance(self): ...             # E[Z²] - E[Z]²
    def entropy(self): ...              # -sum(pmf * log_pmf)
```

### Also port these I/O + misc helpers

| Function | Purpose |
|----------|---------|
| `read_jsonl(path)` | Read JSONL file → list of dicts |
| `write_jsonl(results, path)` | Write list of dicts → JSONL |
| `write_json_array(results, path)` | Write list → JSON array file |
| `get_message(instruction)` | Wrap string → `[{"role":"user","content":...}]` |
| `get_parent_directory(path)` | `os.path.dirname` with strip |
| `resolve_dict_value(d1, d2, key1, key2=None)` | Fallback dict lookup |
| `save_model(model, tokenizer, optimizer, lr_scheduler, accelerator, save_dir, push_to_hub=False, repo_id=None)` | Accelerate-aware model save |

### Also port `perplexity_with_classifier_guidance` (from `utils_hhrlhf.py`)

This is the only function `utils_hhrlhf.py` adds over `utils.py`:

```python
def perplexity_with_classifier_guidance(ref_model, tokenizer, logit_processor,
                                         inputs, response_inputs, eta):
    # Teacher-forced perplexity of continuation tokens under classifier-guided logits.
    # For each batch item: reset processor, for each response token:
    #   build prefix, forward model, apply logit_processor, accumulate -log p(token)
    # PPL = exp(mean NLL). Returns Tensor[B].
```

---

## Task 4: Dataset Layer — `datasets/`

### 4.1 — `datasets/base.py`: Protocol + shared types

Define the `DatasetSpec` protocol and shared data types:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Example:
    example_id: int
    prompt: str
    split: str  # "train" or "eval"
    # Family-specific fields stored as extra dict
    extra: dict

@dataclass
class GenerationTask:
    example_id: int
    repeat_id: int        # 0 .. num_repeats-1
    context_id: int       # 0 .. num_context-1
    seed: int             # derived from base_seed + repeat/context offsets

class DatasetSpec(Protocol):
    name: str
    def load(self, cfg) -> list[Example]: ...
    def build_tasks(self, examples, num_repeats, num_context, base_seed) -> list[GenerationTask]: ...
    def generation_strategy(self) -> str: ...  # "guided" | "offline_pairs" | "guided_pairwise"
```

### 4.2 — `datasets/arithmetic.py`: GSM8K + MATH

**Port from:** data loading in `math_reasoning/collect_training_data.py` and `collect_training_data_pref.py`

- **Input:** JSONL with `prompt` + `answer` (GSM8K) or `solution` (MATH); plus a train/eval split JSON keyed by problem text → `{id, split}`
- **`generation_strategy()`:** returns `"guided"`
- **Scoring:** delegates to `scoring.arithmetic` functions
- Config flag `dataset.name: gsm8k | math` switches only the answer key and match function

**Data paths (existing):**
- GSM8K: `math_reasoning/dataset/gsm8k_*.jsonl` + split JSON
- MATH: `math_reasoning/dataset/math_*.jsonl` + split JSON

### 4.3 — `datasets/preference.py`: HH-RLHF + AlpacaEval

**Port from:** data loading in `math_reasoning/my_alpaca_eval_code/collect_training_data_alpaca.py`

- **Input schema** (shared for both):
  ```json
  {"prompt": str, "output_1": str, "output_2": str, "preference": 1|2, "id": int, "split": str}
  ```
- **`generation_strategy()`:** returns `"offline_pairs"` (round 1) or `"guided"` (round 2+ with classifier ckpt)
- **HH-RLHF:** loads from `math_reasoning/anthropic_hh_train_eval.json` (134 MB, keyed by prompt → `{id, split, output_1, output_2, preference}`)
- **AlpacaEval:** loads from `math_reasoning/dataset/alpaca_noisy_multi_preference_train_eval.json` (same schema, keyed by instruction)

### 4.4 — `datasets/prepare.py`: Alpaca split preparation

**Port from:** `math_reasoning/prepare_alpaca_noisy_multi_preference_split.py`

What it does:
1. `datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_noisy_multi_preference", split="preference")`
2. Filter rows where `input` is empty
3. Shuffle with seed, assign `train`/`eval` splits (90/10)
4. Write JSON dict keyed by `instruction` → `{id, split, output_1, output_2, preference}`

**HF datasets library reference:** https://huggingface.co/docs/datasets/

---

## Task 5: Hydra Configuration — `configs/`

**Reference:** https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/

### 5.1 — `configs/generate.yaml`

```yaml
defaults:
  - dataset: gsm8k

trainer:
  n_gpus_per_node: 8
  output_dir: outputs/${dataset.name}/gen

generation:
  num_repeats: 16
  num_context: 1
  batch_size: 8
  temperature: 0.8
  top_p: 0.9
  max_new_tokens: 1024

models:
  ref_model_id: Qwen/Qwen2.5-7B-Instruct
  classifier_ckpt: null
  classifier_model_id: Qwen/Qwen2.5-1.5B-Instruct
  classifier_type: Q
  inference_mode: expectation
  eta: 1.0
  top_k: 20
  cd_baseline: 0
  dtype: bfloat16
```

### 5.2 — `configs/train.yaml`

Map hyperparams from `math_reasoning/Jeremy/train_hh_rlhf.sh` and the argparse defaults in `my_alpaca_eval_code/train_classifier.py`:

```yaml
defaults:
  - dataset: hh_rlhf

trainer:
  n_gpus_per_node: 8
  output_dir: outputs/${dataset.name}/train

data:
  train_file: null  # path to parquet from generate step
  train_eval_save_path: null
  use_all_ref_tokens: 1
  drop_no_variation: 1
  max_length: -1
  shift_reward: 0
  scale_reward: 1

models:
  ref_model_id: Qwen/Qwen2.5-7B-Instruct
  classifier_model_id: Qwen/Qwen2.5-1.5B-Instruct
  classifier_type: Q
  classifier_ckpt_path: null
  init_mode: reuse
  inference_mode: expectation
  loss_type: bce
  use_bias: 0
  dtype: bfloat16

training:
  batch_size: 8
  gradient_accumulation_steps: 1
  num_epochs: 100
  lr: 2e-5
  warmup_steps: -1
  weight_decay: 0.01
  eval_freq: 500
  ckpt_freq: 5000
  eval_max_size: 1000
  seed: 47

wandb:
  enabled: false
  entity: null
  project: PITA
  run_name: ${models.inference_mode}_${dataset.name}
```

### 5.3 — `configs/eval.yaml`

```yaml
defaults:
  - dataset: gsm8k

trainer:
  output_dir: null  # defaults to classifier_ckpt dir

eval:
  classifier_ckpt: null  # required
  batch_size: 8
  num_samples: 8
  max_new_tokens: 1024
  seed: 47

models:
  # resolved from ckpt args.json, overridable:
  ref_model_id: null
  classifier_model_id: null
  classifier_type: null
  inference_mode: null
  loss_type: null
  eta: null
  top_k: 20
  dtype: null
```

### 5.4 — Dataset YAML overrides (`configs/dataset/`)

**`gsm8k.yaml`:**
```yaml
name: gsm8k
family: arithmetic
answer_key: answer
match_fn: symbolic
data_path: math_reasoning/dataset/gsm8k_train.jsonl
train_eval_save_path: math_reasoning/dataset/gsm8k_train_eval.json
eval_type: arithmetic
generation_strategy: guided
```

**`math.yaml`:** same structure, `answer_key: solution`, different paths.

**`hh_rlhf.yaml`:**
```yaml
name: hh_rlhf
family: preference
data_path: math_reasoning/anthropic_hh_train_eval.json
train_eval_save_path: math_reasoning/anthropic_hh_train_eval.json
eval_type: preference_ppl
generation_strategy: offline_pairs
```

**`alpaca_pref.yaml`:**
```yaml
name: alpaca_pref
family: preference
data_path: math_reasoning/dataset/alpaca_noisy_multi_preference_train_eval.json
train_eval_save_path: math_reasoning/dataset/alpaca_noisy_multi_preference_train_eval.json
eval_type: preference_gen
generation_strategy: offline_pairs
```

---

## Task 6: Generation Pipeline — `generation/`

This is the most complex module. It replaces manual `--start_index` / `--end_index` sharding with Ray-based distributed task execution.

### References

- **Ray core:** https://docs.ray.io/en/latest/ray-core/walkthrough.html
- **Ray remote actors:** https://docs.ray.io/en/latest/ray-core/actors.html
- **Parquet I/O:** https://arrow.apache.org/docs/python/parquet.html

### 6.1 — `generation/collector.py`: Core collection loop

**Port from:** the inner loops of three legacy collect scripts. This is the shared logic that runs on each worker.

Port these helpers (originally in `utils.py`):
- `tokenize_with_chat_template` — already in `training/dataset.py`, import from there
- `generate_with_classifier_guidance` — already in `models/guidance.py`, import from there
- `get_output_indices` — already in `training/dataset.py`, import from there

The collector receives a batch of `GenerationTask` objects and the loaded models, then returns result dicts.

### 6.2 — `generation/strategies.py`: Strategy implementations

Three strategies, each mapping to a legacy collect script:

| Strategy class | Port from | When used |
|----------------|-----------|-----------|
| `GuidedGenerate` | `math_reasoning/collect_training_data.py` inner loop | GSM8K/MATH, preference round 2+ |
| `GuidedPairwise` | `math_reasoning/collect_training_data_pref.py` inner loop | Math pref variant (`num_context > 1`) |
| `OfflinePairs` | `math_reasoning/my_alpaca_eval_code/collect_training_data_alpaca.py` inner loop | HH/Alpaca round 1 |

**`GuidedGenerate` flow** (from `collect_training_data.py`):
1. Generate fully-guided completion (classifier active)
2. Evaluate correctness via `scoring.arithmetic`
3. Random cut the guided response at a random token position
4. Generate **two** unguided continuations from the cut point (classifier disabled, different seeds)
5. Score the pair with DeBERTa reward model → soft preference
6. `evaluate_preference` to determine correctness label

**`GuidedPairwise` flow** (from `collect_training_data_pref.py`):
1. Generate fully-guided completion
2. Random cut
3. Generate **one** unguided continuation
4. Compare partial vs full using OpenMath2 loss as soft preference
5. `evaluate_preference` for label

**`OfflinePairs` flow** (from `collect_training_data_alpaca.py`):
1. No LM generation at all
2. Read `output_1` and `output_2` from dataset
3. Tokenize both responses
4. Labels from dataset `preference` field (1 = output_1 wins, 2 = output_2 wins)
5. Write tokenized prompts/responses with correctness flags

**Key output JSON keys written by each strategy** (these become parquet columns):

Math strategies write:
- `fully_guided_predictions`, `fully_guided_predictions_correctness`
- `partial_guided_prompts_tokenized`, `partial_guided_prompts`
- `num_response_tokens_in_partial_guided_prompts`
- `partial_guided_responses_tokenized`, `partial_guided_predictions`
- `partial_guided_predictions_correctness`

Alpaca/HH OfflinePairs additionally writes:
- `fully_guided_responses_tokenized`
- Does NOT write `num_response_tokens_in_partial_guided_prompts`

### 6.3 — `generation/worker.py`: Ray GPU actor

**Reference:** https://docs.ray.io/en/latest/ray-core/actors.html#actors-with-gpus

```python
@ray.remote(num_gpus=1)
class GenerationWorker:
    def __init__(self, cfg):
        # Load ref_model, classifier, reward_model (if strategy requires) once
        # All on the single GPU assigned by Ray

    def run(self, tasks: list[GenerationTask], examples: dict[int, Example]) -> list[dict]:
        # For each batch of tasks:
        #   strategy.collect(batch, models) → result dicts
        # Write shard parquet to {output_dir}/shards/shard_{rank}.parquet
        # Return path to shard file
```

### 6.4 — `generation/driver.py`: Orchestrator

**Port from:** the outer loop + sharding logic of the collect scripts (replacing `--start_index`/`--end_index`).

```python
def run_generation(cfg):
    ray.init(num_gpus=cfg.trainer.n_gpus_per_node)

    # 1. Load dataset via DatasetSpec
    dataset_spec = instantiate(cfg.dataset)
    examples = dataset_spec.load(cfg)

    # 2. Build all tasks
    tasks = dataset_spec.build_tasks(examples, cfg.generation.num_repeats, cfg.generation.num_context, cfg.generation.seed)

    # 3. Filter already-done tasks via parquet manifest (resume support)
    if os.path.exists(f"{cfg.trainer.output_dir}/shards/"):
        done_keys = read_existing_parquet_keys(...)
        tasks = [t for t in tasks if (t.example_id, t.repeat_id, t.context_id) not in done_keys]

    # 4. Shard tasks across workers
    n_workers = cfg.trainer.n_gpus_per_node
    shards = np.array_split(tasks, n_workers)

    # 5. Dispatch to Ray actors
    workers = [GenerationWorker.remote(cfg) for _ in range(n_workers)]
    shard_paths = ray.get([w.run.remote(s, examples_dict) for w, s in zip(workers, shards)])

    # 6. Merge shard parquets → {output_dir}/train_data.parquet
    merge_parquets(shard_paths, f"{cfg.trainer.output_dir}/train_data.parquet")
```

**Resume logic:** before dispatching, read any existing shard parquets and extract `(example_id, repeat_id, context_id)` tuples. Skip tasks already completed. This replaces the legacy per-file existence check (`{id}_r{i}.json` already exists → skip).

---

## Task 7: Training Pipeline — `training/`

### 7.1 — `training/builder.py`: Unified example builder

**Port from:** the data-prep sections of both `train_classifier.py` files (root + my_alpaca_eval_code)

This merges two different data-prep paths into one:

**Arithmetic path** (from root `train_classifier.py`):
- Read parquet/jsonl with math collect keys
- Compute reward per sample from `partial_guided_predictions_correctness` (all entries, not just [0])
- Each problem → one set of `(prompt_tokenized, response_tokenized, reward)` entries
- Supports `drop_no_variation` flag (skip examples where all rewards are identical)

**Preference path** (from `my_alpaca_eval_code/train_classifier.py`):
- Split each problem into **2 samples**: partial_guided + fully_guided
- Binary reward from `correctness[0]`
- Uses `fully_guided_responses_tokenized` key (Alpaca-specific)

Both paths produce the same `all_data` format for `create_classifier_data`:
```python
[{"prompt_tokenized": [...], "response_tokenized": [...], "reward": [...]}, ...]
```

### 7.2 — `training/trainer.py`: Accelerate training loop

**Port from:** `math_reasoning/my_alpaca_eval_code/train_classifier.py`

**Reference:** https://huggingface.co/docs/accelerate/

Key training loop structure:
```
1. Load classifier model via CustomLlamaForSequenceClassification.from_pretrained
2. Apply init_mode (reuse / zero_init)
3. Build train + eval data via builder.py → create_classifier_data → CustomClassifierDataset
4. Wrap with Accelerate: accelerator.prepare(model, optimizer, scheduler, dataloader)
5. Training loop:
   FOR epoch in num_epochs:
     FOR batch in train_loader:
       global_step += accelerator.num_processes  # FIX: legacy uses hardcoded world_size=1
       loss = classifier_model(input_ids, attention_mask, labels, loss_weights, loss_mask).loss
       loss /= gradient_accumulation_steps
       accelerator.backward(loss)
       every grad_accum steps: clip_grad (V only), optimizer.step, scheduler.step
       every eval_freq: run eval (loss, R², explained variance, MLE stats, accuracy/ROC-AUC)
       every ckpt_freq: save_model(...)
```

**Important fix while porting:** replace `global_step += world_size` (argparse default 1) with `global_step += accelerator.num_processes`.

**Eval metrics computed:**
- Loss, Explained Variance, R², Prediction Min/Max/Mean
- MLE stats (expected value, variance, entropy) if `loss_type=mle`
- Accuracy + ROC-AUC if `inference_mode=bernoulli`
- Root version also does OOD eval on held-out problems

**Outputs:**
- `{output_dir}/args.json` — saved config for eval to reload
- `{output_dir}/ckpt_{step}/` — model checkpoints
- wandb logging (optional)

---

## Task 8: Eval Pipeline — `eval/`

### 8.1 — `eval/arithmetic.py`: GSM8K / MATH evaluation

**Port from:** `math_reasoning/eval_ckpt.py`

Flow:
1. Load training args from `{classifier_ckpt_parent}/args.json`
2. Load eval-split problems from data jsonl, filtered by train_eval split JSON
3. For each repeat (num_samples):
   - Tokenize prompts via `tokenize_with_chat_template`
   - `generate_with_classifier_guidance` (classifier-guided generation)
   - Compute per-token KL(pi_aligned || pi_ref) via `kl_divergence`
   - Save per-example results
4. Post-generation:
   - `process_sample` + `match_fn` → `predictions_correctness`
   - `pass@k`, `majority_vote_correct` via `compute_majority_vote_correct`
5. Write `inference_eval_results_*.jsonl` + `reward_stats_*.json`

**Metrics:** `single_sample_accuracy_mean`, `majority_vote_accuracy_mean`, `pass_k_accuracy_mean`, per-example KL

### 8.2 — `eval/preference.py`: HH-RLHF + AlpacaEval evaluation

Two sub-modes dispatched by `cfg.dataset.eval_type`:

**`preference_ppl`** (HH-RLHF) — Port from `eval_ckpt_hhrlhf.py`:
1. Load preference data (JSON dict or HF dataset)
2. For each batch:
   - Tokenize prompts + two candidate responses
   - `ppl_1 = perplexity_with_classifier_guidance(response_1 | prompt)`
   - `ppl_2 = perplexity_with_classifier_guidance(response_2 | prompt)`
   - `predicted_preference = 1 if ppl_1 <= ppl_2 else 2`
   - `is_success = (predicted == true_preference)`
3. Write results + `reward_stats` with `win_rate`

**`preference_gen`** (AlpacaEval) — Port from `my_alpaca_eval_code/eval_ckpt.py`:
1. Load `tatsu-lab/alpaca_eval` eval split from HF
2. For each batch:
   - `generate_with_classifier_guidance`
   - Compute KL
3. Write `inference_eval_results_*.jsonl` + `model_outputs.json` (Alpaca-Eval submission format)
4. No correctness eval — outputs are for external Alpaca-Eval benchmark

---

## Task 9: CLI Entrypoints — `cli/`

### Reference

- **Hydra main decorator:** https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/
- **`hydra.utils.instantiate`:** https://hydra.cc/docs/advanced/instantiate_objects/overview/

### 9.1 — `cli/generate.py`

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig):
    from generation.driver import run_generation
    run_generation(cfg)

if __name__ == "__main__":
    main()
```

Run: `python -m cli.generate dataset=gsm8k trainer.n_gpus_per_node=4`

### 9.2 — `cli/train.py`

```python
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    from training.trainer import run_training
    run_training(cfg)
```

Run: `python -m cli.train dataset=hh_rlhf data.train_file=outputs/hh_rlhf/gen/train_data.parquet`

For multi-GPU, wrap with accelerate:
```bash
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
python -m accelerate.commands.launch --num_processes $NUM_GPUS -m cli.train dataset=hh_rlhf ...
```

### 9.3 — `cli/eval.py`

```python
@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    if cfg.dataset.eval_type == "arithmetic":
        from eval.arithmetic import run_eval
    else:
        from eval.preference import run_eval
    run_eval(cfg)
```

Run: `python -m cli.eval dataset=hh_rlhf eval.classifier_ckpt=outputs/hh_rlhf/train/ckpt_5000/`

---

## Task 10: Validation

### 10.1 — Smoke test each module imports

```bash
python -c "from scoring.arithmetic import quick_evaluate_single; print('scoring OK')"
python -c "from models.classifier import CustomLlamaForSequenceClassification; print('models OK')"
python -c "from models.guidance import CustomValueGuidedLogitProcessor; print('guidance OK')"
python -c "from training.dataset import create_classifier_data, CustomClassifierDataset; print('training data OK')"
python -c "from datasets.base import DatasetSpec, GenerationTask; print('datasets OK')"
python -c "from generation.driver import run_generation; print('generation OK')"
```

### 10.2 — Small-shard parity test

Run the new pipeline on a small subset and compare against legacy output:

```bash
# Generate (10 examples, 2 repeats, 1 GPU)
python -m cli.generate dataset=hh_rlhf \
    generation.num_repeats=2 \
    trainer.n_gpus_per_node=1 \
    trainer.output_dir=outputs/test_hh

# Compare parquet keys against legacy JSON shard keys
python -c "
import pandas as pd, json
df = pd.read_parquet('outputs/test_hh/train_data.parquet')
print('Parquet columns:', list(df.columns))
print('Rows:', len(df))
"
```

### 10.3 — Training parity

```bash
python -m cli.train dataset=hh_rlhf \
    data.train_file=outputs/test_hh/train_data.parquet \
    training.num_epochs=1 \
    training.eval_freq=10 \
    trainer.output_dir=outputs/test_hh_train
```

Verify `args.json` is written and checkpoint directories are created.

### 10.4 — Eval parity

```bash
python -m cli.eval dataset=hh_rlhf \
    eval.classifier_ckpt=outputs/test_hh_train/ckpt_10/ \
    eval.batch_size=4 \
    eval.num_samples=2
```

---

## Summary: Dependency Order

Build modules in this order — each step only depends on previously completed steps:

```
Task 0: Environment + pyproject.toml + __init__.py files
  ↓
Task 1: scoring/arithmetic.py          (no internal deps)
  ↓
Task 2: models/classifier.py           (no internal deps)
         models/guidance.py             (imports models.classifier)
  ↓
Task 3: training/dataset.py            (no internal deps, I/O + data utils)
  ↓
Task 4: datasets/base.py               (no internal deps)
         datasets/arithmetic.py         (imports scoring.arithmetic)
         datasets/preference.py         (no internal deps)
         datasets/prepare.py            (no internal deps)
  ↓
Task 5: configs/                        (YAML files, no code deps)
  ↓
Task 6: generation/collector.py         (imports models.*, training.dataset, scoring.*)
         generation/strategies.py        (imports collector)
         generation/worker.py            (imports strategies, ray)
         generation/driver.py            (imports worker, datasets.*)
  ↓
Task 7: training/builder.py             (imports training.dataset)
         training/trainer.py             (imports builder, models.classifier, training.dataset)
  ↓
Task 8: eval/arithmetic.py              (imports models.*, scoring.*, training.dataset)
         eval/preference.py              (imports models.*, training.dataset)
  ↓
Task 9: cli/generate.py                 (imports generation.driver)
         cli/train.py                    (imports training.trainer)
         cli/eval.py                     (imports eval.*)
  ↓
Task 10: Validation
```
