#!/bin/bash

# Default paths for F1-mouse-RNA project
VCF_FILE="/homes/gws/tuxm/Project/F1-mouse-RNA/data/CAST_EiJ.mgp.v5.indels.dbSNP142.normed.vcf.gz"
GENE_META="/homes/gws/tuxm/Project/F1-mouse-RNA/data/gene_meta_new_TSS.csv"
GENOME_FILE="/homes/gws/tuxm/Project/Decipher-multi-modality/data/genome/mm10.fa"
CAST_EXPR="/homes/gws/tuxm/Project/F1-mouse-RNA/data/GeneExpression_Normalized_Lg2_Qntlnorm_CAST.tsv"
B6_EXPR="/homes/gws/tuxm/Project/F1-mouse-RNA/data/GeneExpression_Normalized_Lg2_Qntlnorm_B6.tsv"
OUTPUT_DIR="/homes/gws/tuxm/Project/DeepAllele-public/explore/preprocess-script/output"

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

echo "Running preprocessing with window size: ${WINDOW_SIZE}"
echo "Output file: ${OUTPUT_FILE}"

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
