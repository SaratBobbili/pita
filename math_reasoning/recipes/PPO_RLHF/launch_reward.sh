#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MR_ROOT}"
export PYTHONPATH="${MR_ROOT}:${PYTHONPATH:-}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
ACCEL_CONFIG="${SCRIPT_DIR}/configs/accelerate/deepspeed_zero3.yaml"
MAIN_PORT="${MAIN_PROCESS_PORT:-29501}"

if [[ ! -e "${SCRIPT_DIR}/data" ]]; then
  python "${SCRIPT_DIR}/prepare_data.py"
fi

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  "${SCRIPT_DIR}/train_reward.py" \
  "$@"
