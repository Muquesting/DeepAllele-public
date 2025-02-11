#!/bin/bash
# Script to run training with multiple seeds using reproduce_single_head_models.py

# Set the paths
IN_FOLDER="/homes/gws/tuxm/Project/F1-mouse-RNA/explore/unified-dataloader/RNA-seq-preprocessed.hdf5"
OUT_FOLDER="./output/"
PROJECT_ID="DeepAllele-F1-mouse-RNA-reproduce"  # Add project ID parameter

# (Optional) If using Weights & Biases, provide your API key
WANDB_API_KEY="62cb39b9e832e674d05386ec0a742e4855bfa1d0"

# Run training mode over seeds 0 to 4
python reproduce_single_head_models.py \
  --mode train \
  --in_folder "$IN_FOLDER" \
  --out_folder "$OUT_FOLDER" \
  --batch_id "MF_PC_IL4" \
  --use_wandb \
  --wandb_api "$WANDB_API_KEY" \
  --project_id "$PROJECT_ID" \
  --random_seed_start 1 \
  --random_seed_end 6 \
  --device 2 \
  --conv_layers 4 \
  --conv_repeat 1 \
  --kernel_number 512 \
  --kernel_length 10 \
  --filter_number 256 \
  --kernel_size 3 \
  --pooling_size 5 \
  --h_layers 1 \
  --hidden_size 512 \
  --learning_rate 1e-4

python reproduce_single_head_models.py \
  --mode train \
  --in_folder "$IN_FOLDER" \
  --out_folder "$OUT_FOLDER" \
  --batch_id "B_Fo_Sp_IL4" \
  --use_wandb \
  --wandb_api "$WANDB_API_KEY" \
  --project_id "$PROJECT_ID" \
  --random_seed_start 1 \
  --random_seed_end 6 \
  --device 2 \
  --conv_layers 4 \
  --conv_repeat 1 \
  --kernel_number 512 \
  --kernel_length 10 \
  --filter_number 256 \
  --kernel_size 3 \
  --pooling_size 5 \
  --h_layers 1 \
  --hidden_size 512 \
  --learning_rate 1e-4
# To run validation with a saved checkpoint, uncomment and modify the following:
# CHECKPOINT_PATH="/path/to/your_checkpoint.ckpt"
# python reproduce_single_head_models.py \
#   --mode validate \
#   --in_folder "$IN_FOLDER" \
#   --out_folder "$OUT_FOLDER" \
#   --checkpoint_path "$CHECKPOINT_PATH" \
#   --batch_id "MF_PC_IL4" \
#   --project_id "$PROJECT_ID" \
#   --device 0
