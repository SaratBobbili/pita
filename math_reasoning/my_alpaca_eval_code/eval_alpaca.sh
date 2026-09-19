DATASET_TYPE=alpaca_eval
TRAIN_EVAL_PATH=collected_data/alpaca/alpaca_noisy_multi_preference_train_eval.json
INFERENCE_MODE=expectation
ETA=6
CKPT_PATH=../checkpoints/alpaca/pair_loss/ckpt_5000

mkdir -p ${CKPT_PATH}/${INFERENCE_MODE}_${ETA}
python eval_ckpt.py \
  --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
  --classifier_ckpt_path $CKPT_PATH \
  --classifier_type V \
  --eta ${ETA} \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --output_dir ${CKPT_PATH}/${INFERENCE_MODE}_${ETA} | tee ${CKPT_PATH}/${INFERENCE_MODE}_${ETA}/train.log