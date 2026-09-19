TRAIN_EVAL_PATH=data/alpaca_noisy_multi_preference_train_eval.json
INFERENCE_MODE=expectation
ETA=1.0
CKPT_PATH=/data1/jcarleton/pita/alpaca_eval/checkpoints/llama_3_8b_instruct_hh_rlhf_expectation/ckpt_125000

mkdir -p ${CKPT_PATH}/${INFERENCE_MODE}_${ETA}
python eval_ckpt_hhrlhf.py \
  --data_path data/anthropic_hh_test_eval.json \
  --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
  --classifier_ckpt_path $CKPT_PATH \
  --classifier_type V \
  --eta ${ETA} \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --output_dir ${CKPT_PATH}/${INFERENCE_MODE}_${ETA} | tee ${CKPT_PATH}/${INFERENCE_MODE}_${ETA}/train.log