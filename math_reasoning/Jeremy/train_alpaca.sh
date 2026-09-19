DATASET_TYPE=alpaca_eval
DATA_PATH=data/alpaca_train_pref_data.jsonl
TRAIN_EVAL_PATH=data/alpaca_noisy_multi_preference_train_eval.json
INFERENCE_MODE=bernoulli

mkdir -p checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}
cp "$0" checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/
python train_classifier.py \
  --ref_model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --classifier_model_id meta-llama/Llama-3.2-1B-Instruct \
  --dataset_type ${DATASET_TYPE} \
  --data_path ${DATA_PATH} \
  --classifier_type V \
  --train_eval_save_path $TRAIN_EVAL_PATH \
  --init_mode reuse \
  --inference_mode ${INFERENCE_MODE} \
  --loss_type bce \
  --output_dir checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/ \
  --track 1 \
  --wandb_entity jcarleton-texas-a-m-university \
  --wandb_project PITA \
  --wandb_run_name ${INFERENCE_MODE}_${DATASET_TYPE} \
  --num_epochs 100 | tee checkpoints/llama_3_8b_instruct_${DATASET_TYPE}_${INFERENCE_MODE}/train.log