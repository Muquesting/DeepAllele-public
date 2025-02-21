#!/bin/bash

# Common parameters
OUT_ROOT="./output/RNA-seq"
WANDB_API_KEY="62cb39b9e832e674d05386ec0a742e4855bfa1d0"
PROJECT_ID="DeepAllele-F1-RNA"

# RNA-seq settings
RNA_DATA="/homes/gws/tuxm/Project/F1-mouse-RNA/explore/unified-dataloader/RNA-seq-preprocessed.hdf5"
RNA_BATCHES=("MF_PC_IL4" "B_Fo_Sp_IL4")

# Function to run RNA-seq model
run_rna_model() {
    local MODEL_TYPE=$1
    local BATCH_ID=$2
    local DEVICE=$3
    local OUT_DIR="${OUT_ROOT}/${MODEL_TYPE}/${BATCH_ID}"

    python unified_models.py \
        --mode train \
        --assay_type "RNA" \
        --model_type "$MODEL_TYPE" \
        --in_folder "$RNA_DATA" \
        --out_folder "$OUT_DIR" \
        --batch_id "$BATCH_ID" \
        --use_wandb \
        --wandb_api "$WANDB_API_KEY" \
        --project_id "$PROJECT_ID" \
        --random_seed_start 1 \
        --random_seed_end 6 \
        --device "$DEVICE" \
        --conv_layers 4 \
        --conv_repeat 1 \
        --kernel_number 512 \
        --kernel_length 10 \
        --filter_number 256 \
        --kernel_size 3 \
        --pooling_size 5 \
        --h_layers 2 \
        --hidden_size 512 \
        --learning_rate 1e-4
}

# Run models for each batch
for batch in "${RNA_BATCHES[@]}"; do
    run_rna_model "single" "$batch" 1
    run_rna_model "multi" "$batch" 1
    run_rna_model "multi_ComputedRatio" "$batch" 1
done
