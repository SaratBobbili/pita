#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${MR_ROOT}"

QSHARP_DIR="${MR_ROOT}/qsharp_hf"
CKPT_DIR="${QSHARP_DIR}/round2_ckpt"
ZIP_PATH="/scratch/project/prj-02-llm-reasoning-shakkottai/pita/qsharp_hf_gsm8k_llama/round2_ckpt.zip"
ARGS_SRC="${MR_ROOT}/checkpoints/llama_3_8b_instruct_gsm8k/training_costs/qsharp_2/args.json"
SUMMARY="${QSHARP_DIR}/eta_summary.tsv"
TOP_K=20
TEMP=0.8
ETAS=(0.5 1.0 4.0 6.0 8.0 10.0)

mkdir -p "${QSHARP_DIR}"

if [[ ! -f "${CKPT_DIR}/model.safetensors" ]]; then
  echo "Unzipping ${ZIP_PATH} into ${QSHARP_DIR} ..."
  unzip -o "${ZIP_PATH}" -d "${QSHARP_DIR}"
fi

cp "${ARGS_SRC}" "${QSHARP_DIR}/args.json"

echo -e "eta\tpass@1\tmaj1@8\tKL" > "${SUMMARY}"

for ETA in "${ETAS[@]}"; do
  OUT_DIR="${QSHARP_DIR}/eta_${ETA}"
  mkdir -p "${OUT_DIR}"
  echo "===== eta=${ETA} ====="

  python eval_ckpt.py \
    --classifier_ckpt_path "${CKPT_DIR}" \
    --eta "${ETA}" \
    --data_path dataset/gsm8k_test.jsonl \
    --train_eval_save_path dataset/gsm8k_test_eval.json \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/eval.log"

  REWARD_STATS="${OUT_DIR}/reward_stats_eta_${ETA}_top_k_${TOP_K}_temp_${TEMP}.json"
  RESULTS_JSONL="${OUT_DIR}/inference_eval_results_eta_${ETA}_top_k_${TOP_K}_temp_${TEMP}.jsonl"

  KL="$(python my_alpaca_eval_code/compute_kl_avg.py "${RESULTS_JSONL}")"
  PASS1="$(python -c "import json; print(json.load(open('${REWARD_STATS}'))['single_sample_accuracy_mean'])")"
  MAJ8="$(python -c "import json; print(json.load(open('${REWARD_STATS}'))['majority_vote_accuracy_mean'])")"

  echo -e "${ETA}\t${PASS1}\t${MAJ8}\t${KL}" >> "${SUMMARY}"
  echo "eta=${ETA} pass@1=${PASS1} maj1@8=${MAJ8} KL=${KL}"
done

echo "Done. Summary: ${SUMMARY}"
cat "${SUMMARY}"
