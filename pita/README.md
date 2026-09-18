# PITA

Preference-guided inference-time alignment ([arXiv:2507.20067](https://arxiv.org/abs/2507.20067)).

The language model is **frozen**. All that trains is a small value classifier that tilts
the policy's logits during decoding, learned directly from preference feedback with no
reward model in the loop.

```
for round in 1..N:
    sample K responses per prompt      # from the frozen policy, guided by classifier_{r-1}
    rank them with PairRM              # preference feedback
    label: reward = win rate           # Bradley-Terry, in [0, 1]
    train classifier_r                 # token-level value regression
finally:
    AlpacaEval 2                       # held-out, length-controlled win rate
```

Round 1 samples unguided (`eta=0`); later rounds sample under the previous classifier, so
the training distribution tracks the policy that guidance actually produces.

## How guidance works

At each decode step the reference model proposes logits. We take its top-`k` candidates,
ask the classifier for the value `z` of each, and add an offset:

| `inference_mode` | offset |
|---|---|
| `expectation` (default) | `eta * z` |
| `bernoulli` | `log1p_exp(eta + z) - log1p_exp(z)` |
| `cd_baseline` | `eta * sigmoid(z)` (contrastive-decoding ablation) |
| `mle` loss head | `logsumexp(log_pmf + eta * atoms)`, re-centred |

`eta` is the single knob that trades reward against divergence from the policy.

This runs inside vLLM as a V1 logits processor (`pita/guidance.py`), so generation keeps
continuous batching and paged attention for the 8B policy while the ~1B classifier rides
along in lockstep with its own KV cache.

## Layout

```text
pita/
  pita/
    classifier.py   ValueClassifier: any AutoModel backbone + Q or V value head
    guidance.py     vLLM V1 logits processor + the classifier's KV bank
    masking.py      4D attention masks, plain torch
    data.py         ranked generations -> training tensors
    trainer.py      Accelerate training loop
    configs.py      dataclasses + YAML/CLI parser
    run_pita.py     training entry point
  scripts/          generate / combine / preload / rank / build_dataset (+ .sh drivers)
  recipes/          accelerate config, one YAML per model family
  evaluation/       AlpacaEval 2 generation
  tests/
```

## Model-agnostic by construction

The only hard requirement is that the policy and the classifier **share a tokenizer** —
guidance indexes the classifier's head with reference-model token ids.
`pita.classifier.validate_pair` checks the actual vocabularies at startup and sizes the
head from the reference model's `config.vocab_size`.

Beyond that there are no per-architecture code paths: the backbone is composed via
`AutoModel`, prompts go through `apply_chat_template`, and attention masks are built
locally rather than imported from a model file. **Adding a family is one YAML.**

## Running

```bash
# All three rounds plus AlpacaEval generation
bash run_pita_llama-3.sh
```

Or a single round:

```bash
# sample -> rank -> label  (8 GPUs, one process each)
bash scripts/generate.sh \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt UCLA-AGI/data-mistral-7b-instruct-sppo-iter2 \
    --out_path data-llama-3-8b-instruct-pita-iter2 \
    --eta 1.0 --classifier_path checkpoints/Llama-3-8B-Instruct-PITA-Iter1

# train the next classifier
bash scripts/pipeline.sh \
    --recipe recipes/pita/llama3.yaml \
    --train_file datasets/data-llama-3-8b-instruct-pita-iter2/train.parquet \
    --output_dir checkpoints/Llama-3-8B-Instruct-PITA-Iter2 \
    --classifier_path checkpoints/Llama-3-8B-Instruct-PITA-Iter1
```

## Evaluating

Generation and judging are separate, so the 805 responses can be produced on a GPU node
and scored anywhere:

```bash
python evaluation/alpaca_eval/generate.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --classifier_path checkpoints/Llama-3-8B-Instruct-PITA-Iter3 \
    --eta_sweep 0 0.5 1 2 4 \
    --output_dir results/pita-iter3

export OPENAI_API_KEY=...
alpaca_eval --model_outputs results/pita-iter3/eta_1.0/model_outputs.json \
            --annotators_config weighted_alpaca_eval_gpt4_turbo
```

`eta=0` in the sweep is the unguided reference policy — the baseline the guided numbers
have to beat.

## Memory

vLLM and the classifier share a GPU. The classifier's KV bank is
`max_num_seqs x max_model_len` (~8 GB at 64 x 2048 for a 1B model), so
`--gpu_memory_utilization` defaults to `0.80` rather than vLLM's `0.90`. Raise
`--max_num_seqs` for throughput and lower that fraction to match.

## Tests

```bash
pytest tests/
```

`tests/test_guidance_cache.py` is the important one: it asserts the incrementally-cached
classifier is numerically identical to a plain full forward, including across batch-slot
moves. A desynced cache does not crash — it quietly degrades guidance — so this is the
only thing standing between a subtle bug and a bad AlpacaEval number.
