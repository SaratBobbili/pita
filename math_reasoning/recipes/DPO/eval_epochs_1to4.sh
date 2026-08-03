#!/usr/bin/env bash
# Eval DPO epochs 1-3 (new run) + epoch 4 (old run). Epoch 5 already done.
# Run from math_reasoning/:  bash recipes/DPO/eval_epochs_1to4.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-checkpoints/llama_3_8b_instruct_gsm8k/training_costs/pita/ckpt_10000}"

declare -a CKPTS=(
  "checkpoints/llama_3_8b_instruct_gsm8k/dpo_full_all_epochs/checkpoint-841"   # epoch 1
  "checkpoints/llama_3_8b_instruct_gsm8k/dpo_full_all_epochs/checkpoint-1682"  # epoch 2
  "checkpoints/llama_3_8b_instruct_gsm8k/dpo_full_all_epochs/checkpoint-2523"  # epoch 3
  "checkpoints/llama_3_8b_instruct_gsm8k/dpo_full/checkpoint-3364"             # epoch 4
)

for MODEL_ID in "${CKPTS[@]}"; do
  OUTPUT_DIR="${MODEL_ID}/eval_gsm8k_eta0"
  echo "=== evaluating ${MODEL_ID} -> ${OUTPUT_DIR} ==="
  MODEL_ID="${MODEL_ID}" OUTPUT_DIR="${OUTPUT_DIR}" CLASSIFIER_CKPT="${CLASSIFIER_CKPT}" \
    bash recipes/DPO/launch_eval.sh
done

echo "=== done. epoch 5 already at checkpoints/.../dpo_full/checkpoint-4205/eval_gsm8k_eta0 ==="
