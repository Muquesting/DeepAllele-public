#!/bin/bash

# ChIP-seq model training script for DeepAllele
# This script trains various models on ChIP-seq data

# Common parameters - adjust as needed
OUT_ROOT="${OUT_ROOT:-./output/ChIP-seq}"
PROJECT_ID="${PROJECT_ID:-DeepAllele-CHIP}"

# Required: set your own Weights & Biases API key if using W&B
WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"

# ChIP-seq data paths
DATA_DIR="${DATA_DIR:-./data}"
SPRET_DATA="${SPRET_DATA:-${DATA_DIR}/sequence_datasets_chip_SPRET_B6.hdf5}"
PWK_DATA="${PWK_DATA:-${DATA_DIR}/sequence_datasets_chip_PWK_B6.hdf5}"

# Function to run ChIP-seq model
run_chip_model() {
    local STRAIN=$1
    local MODEL_TYPE=$2
    local DATA_PATH=$3
    local DEVICE=$4
    local OUT_DIR="${OUT_ROOT}/${MODEL_TYPE}/${STRAIN}"

    echo "Training ${MODEL_TYPE} model for ${STRAIN} on device ${DEVICE}"
    
    # Create output directory
    mkdir -p "$OUT_DIR"
    
    python unified_models.py \
        --mode train \
        --assay_type "CHIP" \
        --model_type "$MODEL_TYPE" \
        --in_folder "$DATA_PATH" \
        --out_folder "$OUT_DIR" \
        --use_wandb \
        --wandb_api "$WANDB_API_KEY" \
        --project_id "$PROJECT_ID" \
        --random_seed_start 1 \
        --random_seed_end 6 \
        --device "$DEVICE" \
        --conv_layers 6 \
        --conv_repeat 1 \
        --kernel_number 512 \
        --kernel_length 15 \
        --filter_number 256 \
        --kernel_size 5 \
        --pooling_size 2 \
        --h_layers 2 \
        --hidden_size 512 \
        --learning_rate 1e-4
}

# Display configuration
echo "ChIP-seq Model Training"
echo "======================"
echo "Output directory: ${OUT_ROOT}"
echo "SPRET data file: ${SPRET_DATA}"
echo "PWK data file: ${PWK_DATA}"
echo

# Run SPRET models
echo "Processing SPRET models"
run_chip_model "SPRET" "single" "$SPRET_DATA" 0
run_chip_model "SPRET" "multi" "$SPRET_DATA" 0
echo "SPRET models completed"
echo

# Run PWK models
echo "Processing PWK models"
run_chip_model "PWK" "single" "$PWK_DATA" 0
run_chip_model "PWK" "multi" "$PWK_DATA" 0
echo "PWK models completed"
echo

echo "All ChIP-seq models completed"
