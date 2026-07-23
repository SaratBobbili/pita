#!/usr/bin/env bash
# Entry stub for train recipes. Wire recipe scripts in later subtasks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEL_CONFIG="${ACCEL_CONFIG:-${ROOT}/configs/accelerate_configs/deepspeed_zero3.yaml}"

echo "pita_vllm/train launch stub"
echo "  ROOT=${ROOT}"
echo "  ACCEL_CONFIG=${ACCEL_CONFIG}"
echo "Pass a recipe command after configs exist, e.g.:"
echo "  accelerate launch --config_file \"\${ACCEL_CONFIG}\" python -m recipes.pita.<entrypoint> ..."
exit 1
