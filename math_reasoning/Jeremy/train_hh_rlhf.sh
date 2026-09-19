#!/usr/bin/env bash
DATASET_TYPE=hh_rlhf
INFERENCE_MODE=expectation
CKPT_DIR=/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/pita/hhrlhf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_PATH="${MR_ROOT}/hh_rlhf_train_pref_data.jsonl"
TRAIN_EVAL_PATH="${MR_ROOT}/anthropic_hh_train_eval.json"
TRAIN_CLASSIFIER="${MR_ROOT}/my_alpaca_eval_code/train_classifier.py"

export PYTHONPATH="${MR_ROOT}/my_alpaca_eval_code:${PYTHONPATH}"

mkdir -p ${CKPT_DIR}
cp "$0" ${CKPT_DIR}/
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
python -m accelerate.commands.launch --num_processes "${NUM_GPUS}" "${TRAIN_CLASSIFIER}" \
  --ref_model_id Qwen/Qwen2.5-7B-Instruct \
  --classifier_model_id Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_type ${DATASET_TYPE} \
  --data_path ${DATA_PATH} \
  --classifier_type Q \
  --train_eval_save_path $TRAIN_EVAL_PATH \
  --init_mode reuse \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --output_dir ${CKPT_DIR}/ \
  --track 1 \
  --wandb_entity jcarleton-texas-a-m-university \
  --wandb_project PITA \
  --wandb_run_name ${INFERENCE_MODE}_${DATASET_TYPE} \
  --eval_max_size 1000 \
  --num_epochs 100 | tee ${CKPT_DIR}/train.log

# Keep only the final checkpoint: pick the ckpt_* dir with the largest global_step
# (numeric sort on the suffix after the last underscore) and delete all others.
LATEST_CKPT=$(ls -d ${CKPT_DIR}/ckpt_*/ 2>/dev/null | awk -F'[_/]' '{print $(NF-1), $0}' | sort -n -k1,1 | tail -1 | awk '{print $2}')
find ${CKPT_DIR} -mindepth 1 -maxdepth 1 -type d -name 'ckpt_*' ! -path "${LATEST_CKPT%/}" -exec rm -rf {} +
