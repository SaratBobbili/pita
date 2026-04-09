
# Parse ETA and CKPT_PATH from command line
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <CKPT_PATH> <ETA>"
  echo "Example: $0 /path/to/checkpoint_dir 0.0"
  exit 1
fi

CKPT_PATH="$1"
ETA="$2"

DATASET_TYPE=alpaca_eval
TRAIN_EVAL_PATH=data/alpaca_noisy_multi_preference_train_eval.json

mkdir -p ${CKPT_PATH}/eval_eta=${ETA}
python eval_ckpt.py \
  --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
  --classifier_ckpt_path $CKPT_PATH \
  --classifier_type V \
  --eta ${ETA} \
  --loss_type bce \
  --output_dir ${CKPT_PATH}/eval_eta=${ETA} | tee ${CKPT_PATH}/eval_eta=${ETA}/train.log

  # Removed: --inference_mode ${INFERENCE_MODE} \