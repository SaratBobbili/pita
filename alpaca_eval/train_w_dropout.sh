
# Parse and validate INFERENCE_MODE from command line
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <INFERENCE_MODE>"
  echo "INFERENCE_MODE must be 'expectation' or 'bernoulli'"
  exit 1
fi

INFERENCE_MODE="$1"
if [[ "$INFERENCE_MODE" != "expectation" && "$INFERENCE_MODE" != "bernoulli" ]]; then
  echo "Error: INFERENCE_MODE must be 'expectation' or 'bernoulli'"
  exit 1
fi

DATASET_TYPE=alpaca_eval
DATA_PATH=data/all_train_pref_data.jsonl
TRAIN_EVAL_PATH=data/alpaca_noisy_multi_preference_train_eval.json

# NOTE: If you wish to log to wandb, add the --wandb_entity, --wandb_project, and --wandb_run_name arguments 
# to the command below

mkdir -p checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}

# Uncomment the line below to copy this training script to the checkpoint directory for record-keeping
# cp "$0" checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/

python train_classifier.py \
  --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
  --dataset_type ${DATASET_TYPE} \
  --data_path ${DATA_PATH} \
  --classifier_type V \
  --classifier_dropout 0.2 \
  --attention_dropout 0.1 \
  --train_eval_save_path $TRAIN_EVAL_PATH \
  --init_mode reuse \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --ckpt_freq 1000 \
  --output_dir checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/ \
  --track 1 \
  --num_epochs 3 | tee checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/train.log