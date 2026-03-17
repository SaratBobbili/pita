#!/bin/bash
#SBATCH -J qwen
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --mem=200G
 
set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# (Optional) activate your env/modules here
# module load cuda/12.1
# source ~/.bashrc && conda activate verl-tool-env

python train_classifier_qwen.py \
  --ref_model_id Qwen/Qwen2.5-1.5B-Instruct \
  --classifier_model_id Qwen/Qwen2.5-1.5B-Instruct \
  --original_problems_path dataset/gsm8k_train.jsonl \
  --train_eval_save_path dataset/gsm8k_train_eval.json \
  --init_mode reuse \
  --inference_mode expectation \
  --loss_type bce \
  --dataset_type gsm8k \
  --data_paths collected_data/Qwen_7b_gsm8k/Qwen_7b_gsm8k_round1.jsonl \
  --drop_no_variation 1 \
  --eta 1 \
  --output_dir checkpoints/Qwen_1_5b_gsm8k/ \
  --num_epochs 5


