#!/bin/bash

# ATAC-seq preprocessing script for DeepAllele
# This script processes ATAC-seq data for model training

# Configuration - adjust these paths as needed
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Default paths - override with environment variables if needed
INPUT_CSV="${INPUT_CSV:-${DATA_DIR}/Rudensky_GSE154680_remapped_B6_Cast_WT_Treg_Tconv_ATAC_Shifted.csv}"
RAW_DATA_PATH="${RAW_DATA_PATH:-${DATA_DIR}/raw_data/}"
B6_SEQ_PATH="${B6_SEQ_PATH:-${RAW_DATA_PATH}/sequences/final_peaks_cast_unc_2021_11_08_B6SEQS.fa}"
CAST_SEQ_PATH="${CAST_SEQ_PATH:-${RAW_DATA_PATH}/sequences/final_peaks_cast_unc_2021_11_08_shifted_to_CASTEIJ.fa}"
SC_DATA_PATH="${SC_DATA_PATH:-${RAW_DATA_PATH}/counts/}"

# Default peak info path - can be configured
PEAKS_INFO_PATH="${PEAKS_INFO_PATH:-${RAW_DATA_PATH}/peaks_info_updated_2021_12_16.txt}"
# Alternative peak info path if needed
# PEAKS_INFO_PATH="${PEAKS_INFO_PATH:-${RAW_DATA_PATH}/counts/ocr_list.txt}"

# Output file name
OUTPUT_FILE="${OUTPUT_DIR}/ATAC-seq-preprocessed.hdf5"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "ATAC-seq Preprocessing"
echo "===================="
echo "Input CSV: ${INPUT_CSV}"
echo "B6 sequences: ${B6_SEQ_PATH}"
echo "CAST sequences: ${CAST_SEQ_PATH}"
echo "Peak info: ${PEAKS_INFO_PATH}"
echo "Output file: ${OUTPUT_FILE}"
echo

# Run the preprocessing script
python atac_preprocess.py \
    --input-csv "$INPUT_CSV" \
    --data-path "$RAW_DATA_PATH" \
    --b6-seq-path "$B6_SEQ_PATH" \
    --cast-seq-path "$CAST_SEQ_PATH" \
    --sc-data-path "$SC_DATA_PATH" \
    --peak-info "$PEAKS_INFO_PATH" \
    --output "$OUTPUT_FILE"

echo "ATAC-seq processing completed"