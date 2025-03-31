#!/bin/bash

# General preprocessing script for DeepAllele
# This script serves as a generic preprocessor for RNA-seq data

# Configuration - adjust these paths as needed
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Default paths - override with environment variables if needed
VCF_FILE="${VCF_FILE:-${DATA_DIR}/CAST_EiJ.mgp.v5.indels.dbSNP142.normed.vcf.gz}"
GENE_META="${GENE_META:-${DATA_DIR}/gene_meta_new_TSS.csv}"
GENOME_FILE="${GENOME_FILE:-${DATA_DIR}/genome/mm10.fa}"
CAST_EXPR="${CAST_EXPR:-${DATA_DIR}/GeneExpression_Normalized_Lg2_Qntlnorm_CAST.tsv}"
B6_EXPR="${B6_EXPR:-${DATA_DIR}/GeneExpression_Normalized_Lg2_Qntlnorm_B6.tsv}"

# Default window size
WINDOW_SIZE=5000

# Set window size from command line argument if provided
if [ -n "$1" ]; then
    WINDOW_SIZE=$1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Output file name based on window size
OUTPUT_FILE="${OUTPUT_DIR}/RNA-seq-preprocessed-w${WINDOW_SIZE}.hdf5"

echo "RNA-seq Data Preprocessing"
echo "========================="
echo "Window size: ${WINDOW_SIZE}"
echo "Output file: ${OUTPUT_FILE}"
echo

# Run the preprocessing script
python rna_preprocess.py \
    --vcf-file "$VCF_FILE" \
    --gene-meta "$GENE_META" \
    --genome-file "$GENOME_FILE" \
    --cast-expr "$CAST_EXPR" \
    --b6-expr "$B6_EXPR" \
    --output "$OUTPUT_FILE" \
    --window-size "$WINDOW_SIZE"

echo "Processing completed for window size: ${WINDOW_SIZE}"
