#!/bin/bash

# Common parameters
OUT_ROOT="./output/ATAC-seq"
WANDB_API_KEY="62cb39b9e832e674d05386ec0a742e4855bfa1d0"
PROJECT_ID="DeepAllele-F1-ATAC"

# ATAC-seq settings
ATAC_DATA="/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5"
ATAC_BATCHES=("sum" "sc")

# Function to run ATAC-seq model
run_atac_model() {
    local MODEL_TYPE=$1
    local BATCH_ID=$2
    local DEVICE=$3
    local OUT_DIR="${OUT_ROOT}/${MODEL_TYPE}/${BATCH_ID}"

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

# Run models for each batch
for batch in "${ATAC_BATCHES[@]}"; do
    run_atac_model "single" "$batch" 3
    run_atac_model "multi" "$batch" 3
    run_atac_model "multi_ComputedRatio" "$batch" 3
done
