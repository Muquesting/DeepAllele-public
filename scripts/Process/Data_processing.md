# Process data

This file provides the individual steps to process the data for DeepAllele's allele-specific model training and analysis. 

## Overview

The following are reference papers for the overall mapping strategy:

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4062837/

http://csbio.unc.edu/CCstatus/Media/mod_acmbcb2013.pdf

http://csbio.unc.edu/CCstatus/Media/suspenders_acmbcb2013.pdf

with a github overview here:

https://github.com/holtjma/suspenders/wiki/Lapels-and-Suspenders-Pipeline

In a nutshell, the mapping strategy is to:
1) create two reference "pseudogenomes": the standard B6 mm10 sequence and a synthetic Cast genome sequence created using a Cast vcf variant file from the mouse genome project 
2) map all reads to both genomes 
3) convert Cast mapping to corresponding B6 coordinates 
4) determine allele of origin for each read (3 outcomes: B6-specific, Cast specific, or equally likely) 
5) create final counts (all in B6 mm10 coordinates): the B6 counts consist of all B6-specific reads + random 1⁄2 of the nonspecific reads and the Cast counts consist of all Cast-specific reads + other 1⁄2 of the nonspecific reads. Everything is converted to B6 coordinates now so don’t need to switch back and forth between different references and can use existing annotations.


## Allele specific mapping

Here we describe how we generate allele specific bam files from raw data. We describe standard bulk data mapping first, with specific modifications for single-cell data below.

### Download genomes and vcf files

Download your strain-specific genome, vcf, and mod files for mapping.

Genomes and VCF files are available at the following from the Mouse Genome Project:
```
https://ftp.ebi.ac.uk/pub/databases/mousegenomes/REL-1505-SNPs_Indels/strain_specific_vcfs/
https://www.mousegenomes.org/snps-indels/
```

In our case, MOD files, strain-specific genome, and corresponding vcf files were downloaded from the UNC collaborative cross project: The previous version of this site (http://csbio.unc.edu/CCstatus/index.py?run=Pseudo) is not available but an archive can be accessed here: https://web.archive.org/web/20221225061827/http://csbio.unc.edu/CCstatus/index.py?run=Pseudo

For ease of reproducibility, we provide the specific versions used in the paper in the variants/ directory.

MOD files can also be generated with your custom vcf and genome build using pylapels, suspenders per above.

### Map all reads to both genomes

We use bowtie2 to map reads to both genomes

First we merge technical replicates and lanes:

```
cat $FASTQ_DIR/${sample_ID}_*_R1.fastq.gz > $FASTQ_DIR/${sample_ID}.R1.fastq.gz
cat $FASTQ_DIR/${sample_ID}_*_R2.fastq.gz > $FASTQ_DIR/${sample_ID}.R2.fastq.gz

FASTQS="$FASTQ_DIR/${sample_ID}.R1.fastq.gz $FASTQ_DIR/${sample_ID}.R2.fastq.gz"

```
and perform and standard adapter and quality adapter_trimming with trimgalore

```
trim_galore --cores 4 -o $OUTPUT_DIR/$sample_ID --fastqc --paired $FASTQS
```

We use bowtie2 for read alignment, aligning separately to each genome

```
bowtie2 --very-sensitive-local --no-mixed --no-discordant --phred33 -I 10 -X 1000 --threads 8 -x $GENOME_INDEX_B6 -1 $OUTPUT_DIR/$sample_ID/${sample_ID}.R1_val_1.fq.gz -2 $OUTPUT_DIR/$sample_ID/${sample_ID}.R2_val_2.fq.gz -S $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.sam

bowtie2 --very-sensitive-local --no-mixed --no-discordant --phred33 -I 10 -X 1000 --threads 8 -x $GENOME_INDEX_CAST -1 $OUTPUT_DIR/$sample_ID/${sample_ID}.R1_val_1.fq.gz -2 $OUTPUT_DIR/$sample_ID/${sample_ID}.R2_val_2.fq.gz -S $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.sam

```

We filter alignments to keep mapped reads and remove reads mapping to blacklist regions:

```
# convert to bam file
samtools view -b $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.sam > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.bam
# 
samtools view -bh -f 3 -F 12 -q 2 $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.bam > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.mapped.bam
# sort by coordinates
samtools view -h $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.mapped.bam | sed '/chrM/d;/random/d;/chrUn/d' | samtools sort -@ 8 -O bam -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.bam
# Remove duplicates
java -Xms32000m -jar $SCRIPT_DIR/picard-2.8.0.jar MarkDuplicates I=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.bam O=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.nodup.bam M=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.dup.metrics.txt REMOVE_DUPLICATES=true ASSUME_SORTED=true

# Remove reads in blacklisted regions
bedtools intersect -abam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.nodup.bam -b $BLACKLIST -wa -v > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam

# Index bam file
cp $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.nodup.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam
samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam
rm -f $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.sam

samtools view -b $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.sam > $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.bam
samtools view -bh -f 3 -F 12 -q 2 $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.bam > $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.mapped.bam
samtools view -h $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.mapped.bam | sed '/chrM/d;/random/d;/chrUn/d' | samtools sort -@ 8 -O bam -o $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.bam
java -Xms32000m -jar $SCRIPT_DIR/picard-2.8.0.jar MarkDuplicates I=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.bam O=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.nodup.bam M=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.dup.metrics.txt REMOVE_DUPLICATES=true ASSUME_SORTED=true

bedtools intersect -abam $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.nodup.bam -b $BLACKLIST -wa -v > $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam

cp $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.nodup.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam
samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam
rm -f $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.sam

```

### Convert Cast mapping to corresponding B6 coordinates using lapels, suspenders

It is recommended to use easy-install (http://packages.python.org/distribute/easy_install.html) for the installation.
```
easy_install lapels
# or
pip install lapels

easy_install suspenders
```

```
#
pylapels -n -f -p 8 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.lapels.medpar.bam $MOD_B6 $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam
#
pylapels -n -f -p 8 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.lapels.medpar.bam $MOD_CAST $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam
# 
pysuspenders $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.lapels.medpar.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.lapels.medpar.bam

```

### Determine allele of origin for each read

Three possible outcomes: B6-specific, Cast specific, or equally likely. 
Split reads into each of these groups.
Also create files that have strain specific + random 1/2 of non-specific

```  
samtools view -b -d po:1 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.bam
samtools view -b -d po:2 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.bam
samtools view -b -d po:3 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam

samtools view -H $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.bam > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam
cp $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sam

samtools view $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.bam | awk '{if(NR%4==1 || NR%4==2){print >> \"$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam\"} else {print >> \"$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sam\"}}
samtools view -h -b -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.bam
samtools view -h -b -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.bam

samtools merge -f -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.po3.merged.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sorted.bam
samtools merge -f -o $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.po3.merged.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sorted.bam

#index merged bams
samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.po3.merged.bam
samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.po3.merged.bam

# B6 only counts
mv $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.only.bam; samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.only.bam
# CAST only counts
mv $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.only.bam; samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.only.bam


```


## Peak calling
You can call peaks on your data using your favorite peak caller.

In our case, for Tregs single cell and bulk ATAC data, we use a previously defined set of Treg-specific open chromatin peaks (251bp fixed width peaks), which called peaks on Treg scATAC data, using cluster-specific peaks (called using MACS2).
For those overlapping known Immgen ATAC peaks, the Immgen version was kept. For new peaks not in the Immgen reference, the new peaks were added.
Available here:

  https://doi.org/10.1073/pnas.2411301121
  
  https://www.ncbi.xyz/geo/query/acc.cgi?acc=GSE216910
  
  https://www.ncbi.xyz/geo/query/acc.cgi?acc=GSM5712663

For the F1 ChIP data, we directly used data available from the source publication where they had already called peaks. They had also already quantified the counts.

  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109965
  https://doi.org/10.1016/j.cell.2018.04.018


### Create final counts for bulk ATAC using featureCounts

(all in B6mm10 coordinates now): the B6 counts consist of all B6-specific reads + random ½ of the non specific reads and the Cast counts consist of all Cast-specific reads + other ½ of the non specific reads. Everything is converted to B6 coordinates now so there is no need to switch back and forth between different references and one can use existing annotations. We use featureCounts to get the counts.

```
CASTBAM=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.po3.merged.bam
B6BAM=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.po3.merged.bam

featureCounts -p -B -F SAF -T 8 -a saf/treg_pks.saf -o B6_counts.bed $B6BAM
featureCounts -p -B -F SAF -T 8 -a saf/treg_pks.saf -o CAST_counts.bed $CASTBAM

```

## Extended peaks from existing bed files

To extend peaks to different sizes for modeling, we use the following R code:

```
####### annotate ocrs associated with different genes ######
library(EnsDb.Mmusculus.v79)
library(BSgenome.Mmusculus.UCSC.mm10)
library(GenomicRanges)
library(dplyr)
library(tidyr)

extend_us_ds_bed <- function(gene.ranges,dist_us=20000,dist_ds=20000){
  gene_vec_coords <- gene.ranges %>% as.data.frame()
  gene_vec_adjust_tss <- data.frame('chr'=gene_vec_coords$seqnames,
                                    'start'=gene_vec_coords$start-dist_us,'end'=gene_vec_coords$end+dist_ds)

  gene_vec_gr <- makeGRangesFromDataFrame(gene_vec_adjust_tss)

  return(gene_vec_gr)
}




get_TSS_posns <- function(ref_flat,dist_us=150,dist_ds=150){
  minus_strand_tx <- ref_flat %>% dplyr::filter(strand=='-') %>% data.frame()
  pos_strand_tx <- ref_flat %>% dplyr::filter(strand=='+') %>% data.frame()

  minus_tss <- minus_strand_tx[,c('chrom','txEnd','txEnd')] %>% data.frame()
  pos_tss <- pos_strand_tx[,c('chrom','txStart','txStart')] %>% data.frame()

  colnames(minus_tss) <- c('chr','start','end')
  colnames(pos_tss) <- c('chr','start','end')

  all_tss <- rbind(pos_tss, minus_tss) %>% data.frame()

  all_tss <- makeGRangesFromDataFrame(all_tss)
  all_tss <- extend_us_ds(all_tss,dist_us,dist_ds)

  all_tss <- as.data.frame(all_tss)
  all_tss <- all_tss[,c(1,2,3)]
  colnames(all_tss) <- c('chr','start','end')

  all_tss <- makeGRangesFromDataFrame(all_tss)

  return(all_tss)
}

get_gene_tss <- function(genes,ref_flat,dist_us=200,dist_ds=50){
  ref_flat_f <- ref_flat %>% dplyr::filter(genename %in% genes) %>% data.frame()
  ref_flat_f_tss <-get_TSS_posns(ref_flat_f, dist_us,dist_ds)
  return(ref_flat_f_tss)
}


getDataFile <- function(filename){
  obj_n <- load(filename)
  print(obj_n)
  res <- get(obj_n)
  rm(obj_n)
  return(res)
}


######## read in bed file ######
cast_bed <- read.table('peaks_info_updated_2021_12_16.txt',sep='\t',header=F)

cast_bedf <- cast_bed[,c(1,2,3)]
colnames(cast_bedf) <- c('chr','start','end')
cast_bedf <- makeGRangesFromDataFrame(cast_bedf)

####### extend #######
library(stringr)

# extend 125 on either side #  500 2x
cast_2x <- extend_us_ds_bed(cast_bedf,125,125)
cast_2x <- as.data.frame(cast_2x)
cast_2x <- cast_2x[,c(1,2,3),drop=F]
cast_2x$peakid <- cast_bed[,4]
write.table(cast_2x, file='peaks_2x_500.bed',sep='\t',quote=F,col.names = F,row.names = F)

cast_2x_nochr <- cast_2x
cast_2x_nochr[,1] <- sapply(cast_2x_nochr[,1],function(x){str_replace_all(x,'chr','')})
write.table(cast_2x_nochr, file='peaks_2x_500_nochr.bed',sep='\t',quote=F,col.names = F,row.names = F)

# extend 375 on either side # 1000 4x
cast_4x <- extend_us_ds_bed(cast_bedf,375,375)
cast_4x <- as.data.frame(cast_4x)
cast_4x <- cast_4x[,c(1,2,3),drop=F]
cast_4x$peakid <- cast_bed[,4]
write.table(cast_4x, file='peaks_4x_1000.bed',sep='\t',quote=F,col.names = F,row.names = F)

cast_4x_nochr <- cast_4x
cast_4x_nochr[,1] <- sapply(cast_4x_nochr[,1],function(x){str_replace_all(x,'chr','')})
write.table(cast_4x_nochr, file='peaks_4x_1000_nochr.bed',sep='\t',quote=F,col.names = F,row.names = F)

# extend 875 on either side # 2000 8x
cast_8x <- extend_us_ds_bed(cast_bedf,875,875)
cast_8x <- as.data.frame(cast_8x)
cast_8x <- cast_8x[,c(1,2,3),drop=F]
cast_8x$peakid <- cast_bed[,4]
write.table(cast_8x, file='peaks_8x_2000.bed',sep='\t',quote=F,col.names = F,row.names = F)


cast_8x_nochr <- cast_8x
cast_8x_nochr[,1] <- sapply(cast_8x_nochr[,1],function(x){str_replace_all(x,'chr','')})
write.table(cast_8x_nochr, file='peaks_8x_2000_nochr.bed',sep='\t',quote=F,col.names = F,row.names = F)


# extend 1875 on either side # 4000 16x
cast_16x <- extend_us_ds_bed(cast_bedf,1875,1875)
cast_16x <- as.data.frame(cast_16x)
cast_16x <- cast_16x[,c(1,2,3),drop=F]
cast_16x$peakid <- cast_bed[,4]
write.table(cast_16x, file='peaks_16x_4000.bed',sep='\t',quote=F,col.names = F,row.names = F)


cast_16x_nochr <- cast_16x
cast_16x_nochr[,1] <- sapply(cast_16x_nochr[,1],function(x){str_replace_all(x,'chr','')})
write.table(cast_16x_nochr, file='peaks_16x_4000_nochr.bed',sep='\t',quote=F,col.names = F,row.names = F)

```

## Data processing for modeling: Sequence inputs and count matrix generation

### Create sequence fastas for both genomes

We use 2 different strategies to create sequence fastas for peak regions. One is to use the MMARGE package and the second is to use a custom python script.

#### MMARGE

Use bed file and two genomes to generate fasta files for regions in bed file

Used the MMARGE package (https://github.com/vlink/marge) for this, to generate a version of the genome sequence with the variants added.
Use package to shift bed files of each peak from B6 to CAST coordinates (MMARGE.pl shift_to_strain) and then use bedtools per below to extract the fasta using both the new genome .fa and the shifted peaks coords.

We use the following functions from MMARGE (As detailed in the MMARGE documentation:https://github.com/vlink/marge/blob/master/MMARGE_documentation.pdf)

```
MARGE.pl prepare_files
MARGE.pl shift_to_strain
MARGE.pl create_genomes
```

Once these genomes have been generated and beds shifted to appropriate coordinates, can extract fastas with bedtools:

```
bedtools getfasta -name -fi mmarge_cast_genome_combined.fa -bed ${bed} -fo ${bed}.fa
```

#### Manual variant insertion

Alternatively, can insert variants individually from a vcf file as follows in insert_variants.py

```
python insert_variants.py
```


## Processing specific to single-cell data:

The following are some specific details of how to process data for single-cell ATAC data.

### Map all reads to both genomes

We use bowtie2 to map reads to both genomes (showing for scATAC data here).

Note, this code was used in: 

0_f1_mapping/f1_pipeline_scatac/4a_conda_bowtie_index_build.sh
0_f1_mapping/f1_pipeline_scatac/5a_conda_bowtie_map.sh
0_f1_mapping/f1_pipeline_scatac/6a_sam_merge_pre_pylapel.sh

```
    bowtie2-build --threads 8 ${ref} ${refout}
    bowtie2 --threads 16 -x ${btref} -1 ${fastq1} -2 ${fastq2} -S ${samout}

    samtools view -b ${samf1} > ${samf1}.bam
    samtools sort -@ 8 ${samf1}.bam -o ${samf1}.sort.bam
    samtools index ${samf1}.sort.bam

    echo "samtools 2"
    samtools view -b ${samf2} > ${samf2}.bam
    samtools sort -@ 8 ${samf2}.bam -o ${samf2}.sort.bam
    samtools index ${samf2}.sort.bam

    echo "samtools merge"
    samtools merge -@ 8 ${samf1}.${samf2}.merge.bam ${samf1}.sort.bam ${samf2}.sort.bam

    rm ${samf1}.bam
    rm ${samf2}.bam

    samtools index ${samf1}.${samf2}.merge.bam

```


### Convert Cast mapping to corresponding B6 coordinates usign lapel, suspenders

0_f1_mapping/f1_pipeline_scatac/7a_run_pylapel_medpar.sh
0_f1_mapping/f1_pipeline_scatac/8a_run_suspenders_sort.sh

```
pylapels -n -p 12 -o $outdir/${samf1}.${samf2}.merge.lapels.medpar.bam ${mod} ${samf1}.${samf2}.merge.bam
samtools sort -n -o ${indir}/${b6bam}.sort.bam ${indir}/${b6bam}.bam
samtools sort -n -o ${indir}/${castbam}.sort.bam ${indir}/${castbam}.bam

samtools index ${indir}/${b6bam}.sort.bam
samtools index ${indir}/${castbam}.sort.bam

pysuspenders ${outdir}/${outbam}.bam ${indir}/${b6bam}.sort.bam ${indir}/${castbam}.sort.bam

```

### Determine allele of origin for each read
0_f1_mapping/f1_pipeline_scatac/9a_split_bam_poi.sh

Three possible outcomes: B6-specific, Cast specific,or equally likely)

```  
samtools view -b -d po:1 -o ${bamdir}/${bam}_po1.bam ${bamdir}/${bam}
samtools view -b -d po:2 -o ${bamdir}/${bam}_po2.bam ${bamdir}/${bam}
samtools view -b -d po:3 -o ${bamdir}/${bam}_po3.bam ${bamdir}/${bam}

```

### Use sinto to convert aligned bam into fragments file for downstream processing

This step is for single cell data only. Sinto can be found here (https://github.com/timoast/sinto?tab=readme-ov-file)

0_f1_mapping/f1_pipeline_scatac/10a_sinto_create_fragments_short.sh

```
samtools view -H ${bamdir}/${bam} | sed -e 's/SN:\([0-9XY]\)/SN:chr\1/' -e 's/SN:MT/SN:chrM/' | samtools reheader - ${bamdir}/${bam} > ${bamdir}/${bam}_chr.bam

samtools sort -o ${bamdir}/${bam}_chr.sorted.bam ${bamdir}/${bam}_chr.bam
samtools index ${bamdir}/${bam}_chr.sorted.bam

cd ${outdir}

sinto fragments -b ${bamdir}/${bam}_chr.sorted.bam -p 8 -f ${bam}.fragments.bed --barcode_regex "[^:]*"

# sort, compress, and index
sort -k1,1 -k2,2n ${bam}.fragments.bed > ${bam}.fragments.sort.bed
bgzip -@ 8 ${bam}.fragments.sort.bed
tabix -p bed ${bam}.fragments.sort.bed.gz

rm ${bam}.fragments.bed
```

### Randomly split reads that are equally likely from each genome across two alleles to create combined fragments files

This step is optional, depending on whether one wants to consider reads that cannot be mapped to either one of the genomes. 

0_f1_mapping/f1_pipeline_scatac/11a_split_fragments.py
using 0_f1_mapping/f1_pipeline_scatac/11b_split_fragment_shell.sh

```
python split_fragments.py ${bam}.fragments.sort.bed.gz

# sort, compress, and index
sort -k1,1 -k2,2n ${fragfile_pre}.split1.bed > ${fragfile_pre}.split1.sort.bed
bgzip -@ 4 ${fragfile_pre}.split1.sort.bed
tabix -p bed ${fragfile_pre}.split1.sort.bed.gz


sort -k1,1 -k2,2n ${fragfile_pre}.split2.bed > ${fragfile_pre}.split2.sort.bed
bgzip -@ 4 ${fragfile_pre}.split2.sort.bed
tabix -p bed ${fragfile_pre}.split2.sort.bed.gz

```

Combine Fragments across files
0_f1_mapping/f1_pipeline_scatac/12a_combine_fragments.py
0_f1_mapping/f1_pipeline_scatac/12b_combine_fragments_shell.sh

```
python combine_fragments.py -A ${fragfile1} -B ${fragfile2} -O ${outpath_pre}.bed

# sort, compress, and index
sort -k1,1 -k2,2n ${outpath_pre}.bed > ${outpath_pre}.sort.bed
bgzip -@ 2 ${outpath_pre}.sort.bed
tabix -p bed ${outpath_pre}.sort.bed.gz
```

### For scATAC, count matrices can be generated using the Signac (https://stuartlab.org/signac/) package:

````
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(dplyr)
library(SummarizedExperiment)
library(biovizBase)
library(Matrix)
library(EnsDb.Mmusculus.v79)
library(BSgenome.Mmusculus.UCSC.mm10)
library(motifmatchr)

set.seed(44)


cust_peaks <- read.table(cust_peaks_path,sep='\t')
# Read in mod Immgen Peaks
cust_peaks_ranges <- getPeaks(cust_peaks_path, sort_peaks = T)
chrom_to_keep1 <- rep('chr',21)
chrom_to_keep2 <- c(seq(1:19),'X','Y')
chrom_to_keep <- paste(chrom_to_keep1,chrom_to_keep2,sep = '')
cust_peaks_ranges <- cust_peaks_ranges[cust_peaks_ranges@seqnames%in%chrom_to_keep,]

fragments_path <- fragpath

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

save(ct_mtx, file=paste0(save_path_pre,'ct_mtx_signac.rds'))

````
