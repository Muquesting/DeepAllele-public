
args <- commandArgs(trailingOnly = TRUE)
# Run as extendBed.R [bedfile] [extension_size] [outfile]
####### annotate ocrs associated with different genes ######
library(EnsDb.Mmusculus.v79)
library(BSgenome.Mmusculus.UCSC.mm10)
library(GenomicRanges)
library(dplyr)
library(tidyr)

bedfile <- as.character(args[1])
ext_size <- as.character(args[2])
outfile <- as.character(args[3])

extend_us_ds_bed <- function(gene.ranges,dist_us=20000,dist_ds=20000){
  gene_vec_coords <- gene.ranges %>% as.data.frame()
  gene_vec_adjust_tss <- data.frame('chr'=gene_vec_coords$seqnames,
                                    'start'=gene_vec_coords$start-dist_us,'end'=gene_vec_coords$end+dist_ds)

  gene_vec_gr <- makeGRangesFromDataFrame(gene_vec_adjust_tss)

  return(gene_vec_gr)
}


######## read in bed file ######
bed_table <- read.table(bedfile,sep='\t',header=F)
bed_table <- bed_table[,c(1,2,3)]
colnames(bed_table) <- c('chr','start','end')
bed_table <- makeGRangesFromDataFrame(bed_table)

####### extend #######
library(stringr)

# extend equal size on either side of peak in bed file
bed_x <- extend_us_ds_bed(bed_table,ext_size,ext_size)
bed_x <- as.data.frame(bed_x)
bed_x <- bed_x[,c(1,2,3),drop=F]
write.table(bed_x, file=outfile,sep='\t',quote=F,col.names = F,row.names = F)

# version of bed file without "chr" prefix on chromosomes
bed_x_nochr <- bed_x
bed_x_nochr[,1] <- sapply(bed_x_nochr[,1],function(x){str_replace_all(x,'chr','')})
write.table(bed_x_nochr, file=gsub('.bed','_nochr.bed',outfile),sep='\t',quote=F,col.names = F,row.names = F)