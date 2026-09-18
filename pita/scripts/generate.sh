#!/bin/bash
# One PITA data-collection round: sample -> combine -> rank -> label.
#
# Derived from SPPO/scripts/generate.sh. Same 8-way data-parallel shape (one process per
# GPU, tensor_parallel_size=1), with the classifier checkpoint and guidance strength
# threaded through to the sampler, and compute_prob.py replaced by build_dataset.py.
#
# The policy is frozen, so --model is the same every round; only --classifier_path moves.
set -e
set -x

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
OUTDIR="data-llama-3-8b-instruct-pita-iter1"
PROMPTS="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"
TRAIN_FILE=""
PAIRS=5
MAXLEN=2048
MAX_MODEL_LEN=4096
CLASSIFIER=""
ETA=0.0
GUIDE_TOP_K=20
INFERENCE_MODE="expectation"
NUM_PROMPTS=20800

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --model) MODEL="$2"; shift ;;
    --out_path) OUTDIR="$2"; shift ;;
    --prompt) PROMPTS="$2"; shift ;;
    --train_file) TRAIN_FILE="$2"; shift ;;
    --pairs) PAIRS="$2"; shift ;;
    --maxlen) MAXLEN="$2"; shift ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift ;;
    --classifier_path) CLASSIFIER="$2"; shift ;;
    --eta) ETA="$2"; shift ;;
    --guide_top_k) GUIDE_TOP_K="$2"; shift ;;
    --inference_mode) INFERENCE_MODE="$2"; shift ;;
    --num_prompts) NUM_PROMPTS="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$TRAIN_FILE" ]; then
    TRAIN_FILE="datasets/${OUTDIR}/train.parquet"
fi

GUIDE_ARGS=()
if [ "$ETA" != "0.0" ] && [ "$ETA" != "0" ]; then
    if [ -z "$CLASSIFIER" ]; then
        echo "--eta $ETA requires --classifier_path"; exit 1
    fi
    GUIDE_ARGS=(--classifier_path "$CLASSIFIER" --eta "$ETA" \
                --guide_top_k "$GUIDE_TOP_K" --inference_mode "$INFERENCE_MODE")
fi

# Ceiling division so the last shard picks up the remainder.
FRAC_LEN=$(( (NUM_PROMPTS + ${#AVAILABLE_GPUS[@]} - 1) / ${#AVAILABLE_GPUS[@]} ))
echo "Using frac_len ${FRAC_LEN}"

#####################
# Generate
#####################
(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/generate.py \
            --model "$MODEL" --maxlen "$MAXLEN" --max_model_len "$MAX_MODEL_LEN" \
            --output_dir "generated/$OUTDIR" \
            --prompts "$PROMPTS" --pairs "$PAIRS" --world_size 1 \
            --frac_len "$FRAC_LEN" --data_frac $data_frac \
            "${GUIDE_ARGS[@]}" > "generate_log_${gpu_id}.txt" 2>&1 &
        ((data_frac+=1))
    done
    wait
)

python3 scripts/combine_generate.py --output_dir "generated/$OUTDIR" \
    --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --pairs "$PAIRS"

#####################
# Rank with PairRM
#####################
python3 scripts/preload.py

(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py \
            --model "$MODEL" --output_dir "$OUTDIR" --pairs "$PAIRS" \
            --numgpu ${#AVAILABLE_GPUS[@]} --frac_len "$FRAC_LEN" \
            --data_frac $data_frac --gpu $gpu_id --prompts "$PROMPTS" \
            > "rank_log_${gpu_id}.txt" 2>&1 &
        ((data_frac+=1))
    done
    wait
)

#####################
# Scores -> rewards
#####################
python3 scripts/build_dataset.py \
    --output_dir "generated/$OUTDIR" --ranking_dir "ranking/$OUTDIR" \
    --train_file "$TRAIN_FILE" --prompts "$PROMPTS" --pairs "$PAIRS" \
    --frac_len "$FRAC_LEN" --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" \
    --drop_no_variation
