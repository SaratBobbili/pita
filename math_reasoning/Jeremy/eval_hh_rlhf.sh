#!/usr/bin/env bash
# Run from this directory (same cwd convention as train_hh_rlhf.sh).
#
# Training writes:
#   checkpoints/qwen_7b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/args.json
#   checkpoints/qwen_7b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/ckpt_<step>/
# eval_ckpt_hhrlhf.py loads args.json from the parent of --classifier_ckpt_path, so
# classifier_ckpt_path must be a ckpt_* subdirectory (not the run root).
#
# Optional: CKPT_PATH=... bash eval_hh_rlhf.sh
# Extra eval args:    CKPT_PATH=... bash eval_hh_rlhf.sh --quick_test

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_TYPE=hh_rlhf
INFERENCE_MODE=expectation
ETA=1.0
CKPT_PATH="Qwen/Qwen2.5-7B-Instruct"

export PITA_CLASSIFIER_DIR="${SCRIPT_DIR}"

mkdir -p "${CKPT_PATH}/${INFERENCE_MODE}_${ETA}"
python "${MR_ROOT}/eval_ckpt_hhrlhf.py" \
  --classifier_ckpt_path "${CKPT_PATH}" \
  --data_path "${SCRIPT_DIR}/anthropic_hh_test.json" \
  --eta "${ETA}" \
  --output_dir "${CKPT_PATH}/${INFERENCE_MODE}_${ETA}" \
  "$@" | tee "${CKPT_PATH}/${INFERENCE_MODE}_${ETA}/eval.log"
