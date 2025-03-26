#!/bin/bash

# Updated paths for ChIP-seq processing
SEQUENCE_PATH="/homes/gws/tuxm/Project/F1-ASCA/data/input/Chip-seq/PU1_F1_ChIP/combined_peaks_gr_pu1_f1_"
C57_PWK_CHIP="/data/tuxm/project/F1-ASCA/data/input/Chip-seq/PU1_F1_ChIP/peaks_F1_BMDM_FPC_PWK_C57_PU1_notx.txt"
C57_SPRET_CHIP="/data/tuxm/project/F1-ASCA/data/input/Chip-seq/PU1_F1_ChIP/peaks_F1_BMDM_FSC_SPRET_C57_PU1_notx.txt"
OUTPUT_DIR="/homes/gws/tuxm/Project/DeepAllele-public/explore/preprocess-script/output"

# Default sequence length
SEQ_LENGTH=551

# Set sequence length from command line argument if provided
if [ -n "$1" ]; then
    SEQ_LENGTH=$1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running ChIP-seq preprocessing with sequence length: ${SEQ_LENGTH}"
echo "Output directory: ${OUTPUT_DIR}"

# Run the preprocessing script
python chip-seq_preprocess.py \
    --sequence-path "$SEQUENCE_PATH" \
    --c57-pwk-chip "$C57_PWK_CHIP" \
    --c57-spret-chip "$C57_SPRET_CHIP" \
    --output-dir "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

echo "ChIP-seq processing completed"