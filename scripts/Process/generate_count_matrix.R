
args <- commandArgs(trailingOnly = TRUE)
# Run as generate_count_matrix.R [bed_file_path] [fragments_file_path] [out_file]

bedfile <- as.character(args[1])
fragments_path <- as.character(args[2])
outfile <- as.character(args[3])

library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(dplyr)
library(SummarizedExperiment)
library(biovizBase)
library(Matrix)
library(EnsDb.Mmusculus.v79)
library(BSgenome.Mmusculus.UCSC.mm10)

set.seed(44)

# Read in peaks
cust_peaks <- read.table(bedfile,sep='\t')
cust_peaks_ranges <- getPeaks(bedfile, sort_peaks = T)
chrom_to_keep1 <- rep('chr',21)
chrom_to_keep2 <- c(seq(1:19),'X','Y')
chrom_to_keep <- paste(chrom_to_keep1,chrom_to_keep2,sep = '')
cust_peaks_ranges <- cust_peaks_ranges[cust_peaks_ranges@seqnames%in%chrom_to_keep,]

fragments <- CreateFragmentObject(
  path = fragments_path,
  validate.fragments = FALSE
)

ct_mtx <- FeatureMatrix(
  fragments,
  cust_peaks_ranges,
  cells = NULL,
  process_n = 10000,
  sep = c("-", "-"),
  verbose = TRUE
)
save(ct_mtx, file=outfile)
