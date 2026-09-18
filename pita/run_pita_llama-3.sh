#!/bin/bash
# Three PITA rounds on Llama-3-8B-Instruct, then AlpacaEval generation.
#
# Derived from SPPO/run_sppo_llama-3.sh. The structural difference: SPPO fed iteration i's
# output checkpoint in as iteration i+1's MODEL, because it trains the policy. PITA never
# touches the policy -- MODEL is constant and only CLASSIFIER advances.
#
# Round 1 samples unguided (eta=0, no classifier exists yet); later rounds sample under
# the previous round's classifier, which is what makes the training data on-policy.
set -e
set -x

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
RECIPE="recipes/pita/llama3.yaml"
ETA=1.0
PAIRS=5
ITERS=3
CLASSIFIER=""

for i in $(seq 1 $ITERS); do
    OUTDIR="data-llama-3-8b-instruct-pita-iter${i}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    TRAIN_FILE="datasets/${OUTDIR}/train.parquet"
    CKPT="checkpoints/Llama-3-8B-Instruct-PITA-Iter${i}"

    if [ "$i" -eq 1 ]; then
        ROUND_ETA=0.0
    else
        ROUND_ETA=$ETA
    fi

    bash scripts/generate.sh \
        --model "$MODEL" --prompt "$PROMPT" --out_path "$OUTDIR" \
        --train_file "$TRAIN_FILE" --pairs "$PAIRS" \
        --eta "$ROUND_ETA" --classifier_path "$CLASSIFIER"

    bash scripts/pipeline.sh \
        --recipe "$RECIPE" --train_file "$TRAIN_FILE" --output_dir "$CKPT" \
        --classifier_path "$CLASSIFIER"

    CLASSIFIER="$CKPT"
done

# Held-out benchmark. Judge separately with:
#   alpaca_eval --model_outputs <output_dir>/model_outputs.json
python3 evaluation/alpaca_eval/generate.py \
    --model "$MODEL" --classifier_path "$CLASSIFIER" --eta "$ETA" \
    --output_dir "${CLASSIFIER}/alpaca_eval"
