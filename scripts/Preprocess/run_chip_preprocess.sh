#!/bin/bash

# ChIP-seq preprocessing script for DeepAllele
# This script processes ChIP-seq data for model training

# Configuration - adjust these paths as needed
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Default paths - override with environment variables if needed
SEQUENCE_PATH="${SEQUENCE_PATH:-${DATA_DIR}/Chip-seq/PU1_F1_ChIP/combined_peaks_gr_pu1_f1_}"
C57_PWK_CHIP="${C57_PWK_CHIP:-${DATA_DIR}/Chip-seq/PU1_F1_ChIP/peaks_F1_BMDM_FPC_PWK_C57_PU1_notx.txt}"
C57_SPRET_CHIP="${C57_SPRET_CHIP:-${DATA_DIR}/Chip-seq/PU1_F1_ChIP/peaks_F1_BMDM_FSC_SPRET_C57_PU1_notx.txt}"

# Default sequence length
SEQ_LENGTH=551

# Set sequence length from command line argument if provided
if [ -n "$1" ]; then
    SEQ_LENGTH=$1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "ChIP-seq Preprocessing"
echo "===================="
echo "Sequence length: ${SEQ_LENGTH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Sequence path: ${SEQUENCE_PATH}"
echo

# Run the preprocessing script
python chip-seq_preprocess.py \
    --sequence-path "$SEQUENCE_PATH" \
    --c57-pwk-chip "$C57_PWK_CHIP" \
    --c57-spret-chip "$C57_SPRET_CHIP" \
    --output-dir "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

echo "ChIP-seq processing completed"