#!/bin/bash

# RNA-seq model training script for DeepAllele
# This script trains various models on RNA-seq data

# Common parameters - adjust as needed
OUT_ROOT="${OUT_ROOT:-./output/RNA-seq}"
PROJECT_ID="${PROJECT_ID:-DeepAllele-RNA}"

# Required: set your own Weights & Biases API key if using W&B
WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"

# RNA-seq data location
RNA_DATA="${RNA_DATA:-./data/RNA-seq-preprocessed.hdf5}"
RNA_BATCHES=("MF_PC_IL4" "B_Fo_Sp_IL4")

# Function to run RNA-seq model
run_rna_model() {
    local MODEL_TYPE=$1
    local BATCH_ID=$2
    local DEVICE=$3
    local OUT_DIR="${OUT_ROOT}/${MODEL_TYPE}/${BATCH_ID}"

    echo "Training ${MODEL_TYPE} model for ${BATCH_ID} on device ${DEVICE}"
    
    # Create output directory
    mkdir -p "$OUT_DIR"
    
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

# Display configuration
echo "RNA-seq Model Training"
echo "====================="
echo "Output directory: ${OUT_ROOT}"
echo "Data file: ${RNA_DATA}"
echo "Batches to process: ${RNA_BATCHES[*]}"
echo

# Run models for each batch
for batch in "${RNA_BATCHES[@]}"; do
    echo "Processing batch: $batch"
    run_rna_model "single" "$batch" 0
    run_rna_model "multi" "$batch" 0
    run_rna_model "multi_ComputedRatio" "$batch" 0
    echo "Completed models for batch: $batch"
    echo
done

echo "All RNA-seq models completed"
