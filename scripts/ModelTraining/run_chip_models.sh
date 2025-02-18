#!/bin/bash

# Common parameters
OUT_ROOT="./output/ChIP-seq"
WANDB_API_KEY="62cb39b9e832e674d05386ec0a742e4855bfa1d0"
PROJECT_ID="DeepAllele-F1-CHIP"

# ChIP-seq data paths
SPRET_DATA="/data/tuxm/project/F1-ASCA/data/input/Chip-seq/processed_data/sequence_datasets_chip_SPRET_B6_20230126.hdf5"
PWK_DATA="/data/tuxm/project/F1-ASCA/data/input/Chip-seq/processed_data/sequence_datasets_chip_PWK_B6_20230126.hdf5"

# Function to run ChIP-seq model
run_chip_model() {
    local STRAIN=$1
    local MODEL_TYPE=$2
    local DATA_PATH=$3
    local DEVICE=$4
    local OUT_DIR="${OUT_ROOT}/${MODEL_TYPE}/${STRAIN}"

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

# Run SPRET models
run_chip_model "SPRET" "single" "$SPRET_DATA" 2
run_chip_model "SPRET" "multi" "$SPRET_DATA" 2

# Run PWK models
run_chip_model "PWK" "single" "$PWK_DATA" 0
run_chip_model "PWK" "multi" "$PWK_DATA" 0
