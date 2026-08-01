#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MR_ROOT}"
export PYTHONPATH="${MR_ROOT}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-checkpoints/llama_3_8b_instruct_gsm8k/dpo_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ID}/eval_gsm8k}"

python "${SCRIPT_DIR}/eval.py" \
  --model_id "${MODEL_ID}" \
  --dataset_type gsm8k \
  --data_path dataset/gsm8k_test.jsonl \
  --train_eval_save_path dataset/gsm8k_test_eval.json \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
