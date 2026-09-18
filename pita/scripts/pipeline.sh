#!/bin/bash
# Train the value classifier on one round's labelled data.
#
# Derived from SPPO/scripts/pipeline.sh, minus update_dataset.py: upstream rewrote its
# tracked recipe YAML in place before every run, leaving the repo dirty. The data path is
# a plain CLI override instead.
set -e
set -x

export OMP_NUM_THREADS=2

RECIPE="recipes/pita/llama3.yaml"
ACCEL_CONFIG="recipes/accelerate_configs/multi_gpu.yaml"
TRAIN_FILE=""
OUTPUT_DIR=""
CLASSIFIER=""
LEARNING_RATE="2.0e-5"
BATCH_SIZE=8
ACCUMULATE=1
EPOCHS=1
PORT=2930

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --recipe) RECIPE="$2"; shift ;;
    --accel_config) ACCEL_CONFIG="$2"; shift ;;
    --train_file) TRAIN_FILE="$2"; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift ;;
    --classifier_path) CLASSIFIER="$2"; shift ;;
    --learning_rate) LEARNING_RATE="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --accumulate) ACCUMULATE="$2"; shift ;;
    --num_epochs) EPOCHS="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$TRAIN_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "--train_file and --output_dir are required"; exit 1
fi

# Round 1 starts from a fresh head; later rounds continue the previous classifier.
RESUME_ARGS=()
if [ -n "$CLASSIFIER" ]; then
    RESUME_ARGS=(--classifier_path="$CLASSIFIER")
fi

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --main_process_port "$PORT" \
    -m pita.run_pita "$RECIPE" \
    --train_file="$TRAIN_FILE" \
    --output_dir="$OUTPUT_DIR" \
    --learning_rate="$LEARNING_RATE" \
    --batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$ACCUMULATE" \
    --num_epochs="$EPOCHS" \
    "${RESUME_ARGS[@]}"
