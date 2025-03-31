#!/bin/bash

# ATAC-seq model training script for DeepAllele
# This script trains various models on ATAC-seq data

# Common parameters - adjust as needed
OUT_ROOT="${OUT_ROOT:-./output/ATAC-seq}"
PROJECT_ID="${PROJECT_ID:-DeepAllele-ATAC}"

# Required: set your own Weights & Biases API key if using W&B
WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"

# ATAC-seq data location
DATA_DIR="${DATA_DIR:-./data}"
ATAC_DATA="${ATAC_DATA:-${DATA_DIR}/ATAC-seq-preprocessed.hdf5}"
ATAC_BATCHES=("sum" "sc")

# Function to run ATAC-seq model
run_atac_model() {
    local MODEL_TYPE=$1
    local BATCH_ID=$2
    local DEVICE=$3
    local OUT_DIR="${OUT_ROOT}/${MODEL_TYPE}/${BATCH_ID}"

    echo "Training ${MODEL_TYPE} model for ${BATCH_ID} on device ${DEVICE}"
    
    # Create output directory
    mkdir -p "$OUT_DIR"
    
    python unified_models.py \
        --mode train \
        --assay_type "ATAC" \
        --model_type "$MODEL_TYPE" \
        --in_folder "$ATAC_DATA" \
        --out_folder "$OUT_DIR" \
        --batch_id "$BATCH_ID" \
        --use_wandb \
        --wandb_api "$WANDB_API_KEY" \
        --project_id "$PROJECT_ID" \
        --random_seed_start 1 \
        --random_seed_end 5 \
        --device "$DEVICE" \
        --conv_layers 4 \
        --conv_repeat 1 \
        --kernel_number 512 \
        --kernel_length 15 \
        --filter_number 256 \
        --kernel_size 5 \
        --pooling_size 4 \
        --h_layers 2 \
        --hidden_size 512 \
        --learning_rate 1e-4
}

# Display configuration
echo "ATAC-seq Model Training"
echo "======================"
echo "Output directory: ${OUT_ROOT}"
echo "Data file: ${ATAC_DATA}"
echo "Batches to process: ${ATAC_BATCHES[*]}"
echo

# Run models for each batch
for batch in "${ATAC_BATCHES[@]}"; do
    echo "Processing batch: $batch"
    run_atac_model "single" "$batch" 0
    run_atac_model "multi" "$batch" 0
    run_atac_model "multi_ComputedRatio" "$batch" 0
    echo "Completed models for batch: $batch"
    echo
done

echo "All ATAC-seq models completed"
