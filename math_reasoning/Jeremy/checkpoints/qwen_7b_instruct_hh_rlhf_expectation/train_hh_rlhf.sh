DATASET_TYPE=hh_rlhf
DATA_PATH=../hh_rlhf_train_pref_data.jsonl
TRAIN_EVAL_PATH=../anthropic_hh_train_eval.json
INFERENCE_MODE=expectation

# utils_hhrlhf.py lives in ../my_alpaca_eval_code; expose it on the import path
export PYTHONPATH=../my_alpaca_eval_code:${PYTHONPATH}

mkdir -p checkpoints/qwen_7b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}
cp "$0" checkpoints/qwen_7b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
python -m accelerate.commands.launch --num_processes "${NUM_GPUS}" train_classifier.py \
  --ref_model_id Qwen/Qwen2.5-7B-Instruct \
  --classifier_model_id Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_type ${DATASET_TYPE} \
  --data_path ${DATA_PATH} \
  --classifier_type Q \
  --train_eval_save_path $TRAIN_EVAL_PATH \
  --init_mode reuse \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --output_dir checkpoints/qwen_7b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/ \
  --track 1 \
  --wandb_entity jcarleton-texas-a-m-university \
  --wandb_project PITA \
  --wandb_run_name ${INFERENCE_MODE}_${DATASET_TYPE} \
  --max_eval_problems 1000 \
  --num_epochs 100 | tee checkpoints/qwen_7b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/train.log