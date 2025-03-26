#!/bin/bash

# Updated paths for ATAC-seq processing based on notebook
INPUT_CSV="/homes/gws/tuxm/Project/F1-ASCA/data/input/Rudensky_GSE154680_remapped_B6_Cast_WT_Treg_Tconv_ATAC_Shifted.csv"
DATA_PATH="/data/tuxm/project/F1-ASCA/data/raw_data/"
B6_SEQ_PATH="/data/tuxm/project/F1-ASCA/data/raw_data/sequences/final_peaks_cast_unc_2021_11_08_B6SEQS.fa"
CAST_SEQ_PATH="/data/tuxm/project/F1-ASCA/data/raw_data/sequences/final_peaks_cast_unc_2021_11_08_shifted_to_CASTEIJ.fa"
SC_DATA_PATH="/data/tuxm/project/F1-ASCA/data/raw_data/counts/"

# Updated to use the more detailed peaks info file if available
PEAKS_INFO_PATH="/data/tuxm/project/F1-ASCA/data/raw_data/peaks_info_updated_2021_12_16.txt"
# If the above file doesn't exist, fall back to:
# PEAKS_INFO_PATH="/data/tuxm/project/F1-ASCA/data/raw_data/counts/ocr_list.txt"

OUTPUT_DIR="/homes/gws/tuxm/Project/DeepAllele-public/explore/preprocess-script/output"

# Output file name
OUTPUT_FILE="${OUTPUT_DIR}/ATAC-seq-preprocessed.hdf5"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running ATAC-seq preprocessing"
echo "Input CSV: ${INPUT_CSV}"
echo "B6 sequences: ${B6_SEQ_PATH}"
echo "CAST sequences: ${CAST_SEQ_PATH}"
echo "Peak info: ${PEAKS_INFO_PATH}"
echo "Output file: ${OUTPUT_FILE}"

# Run the preprocessing script
python atac_preprocess.py \
    --input-csv "$INPUT_CSV" \
    --data-path "$DATA_PATH" \
    --b6-seq-path "$B6_SEQ_PATH" \
    --cast-seq-path "$CAST_SEQ_PATH" \
    --sc-data-path "$SC_DATA_PATH" \
    --peak-info "$PEAKS_INFO_PATH" \
    --output "$OUTPUT_FILE"

echo "ATAC-seq processing completed"