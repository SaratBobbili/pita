#!/usr/bin/env bash
# Evaluate a DPO policy via math_reasoning/eval_ckpt.py with eta=0
# (unguided generation; classifier is loaded but disabled — same as ref_pass*).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MR_ROOT}"
export PYTHONPATH="${MR_ROOT}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-checkpoints/llama_3_8b_instruct_gsm8k/dpo_full/checkpoint-4205}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ID}/eval_gsm8k_eta0}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-checkpoints/llama_3_8b_instruct_gsm8k/training_costs/pita/ckpt_10000}"
ETA="${ETA:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"

python eval_ckpt.py \
  --ref_model_id "${MODEL_ID}" \
  --classifier_ckpt_path "${CLASSIFIER_CKPT}" \
  --eta "${ETA}" \
  --num_samples "${NUM_SAMPLES}" \
  --data_path dataset/gsm8k_test.jsonl \
  --train_eval_save_path dataset/gsm8k_test_eval.json \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
