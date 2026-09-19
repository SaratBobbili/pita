DATASET_TYPE=alpaca_eval
TRAIN_EVAL_PATH=data/alpaca_noisy_multi_preference_train_eval.json
INFERENCE_MODE=bernoulli
ETA=10.0
CKPT_PATH=checkpoints/llama_3_8b_instruct_${DATASET_TYPE}/ckpt_5000

python eval_ckpt.py \
  --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
  --classifier_ckpt_path $CKPT_PATH \
  --classifier_type V \
  --eta ${ETA} \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --output_dir $CKPT_PATH | tee checkpoints/llama_3_8b_instruct_${DATASET_TYPE}/train.log