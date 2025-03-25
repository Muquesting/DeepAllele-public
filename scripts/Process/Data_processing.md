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

For ease of reproducibility, we provide the specific versions used in the paper in the variants/ directory: https://www.dropbox.com/scl/fo/nlvkd7sz1iad8pzfqfew7/AJv5zQbWOTfEPxSQqM9zViI?rlkey=zn8vz9kxeyk8pl5zv2tfx1qmj&e=1&dl=0

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

We filter alignments to keep mapped reads and remove reads mapping to blacklist regions. For removing duplicates, we use Picard. This can be downloaded here: https://broadinstitute.github.io/picard/ 
The blacklist used in this step can be downloaded from the ENCODE project: https://mitra.stanford.edu/kundaje/akundaje/release/blacklists/mm10-mouse/ 

```
wget https://mitra.stanford.edu/kundaje/akundaje/release/blacklists/mm10-mouse/mm10.blacklist.bed.gz
BLACKLIST=mm10.blacklist.bed.gz

# convert to bam file
samtools view -b $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.sam > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.bam

# Filter reads from the BAM file that have both paired end reads
# and reads are properly aligned to its mate (-f 3).
# Exclude reads where the read is unmapped or where the mate in umapped (-F 12)
# Keep only reads with mapping quality > 2 (-q2)
samtools view -bh -f 3 -F 12 -q 2 $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.bam > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.mapped.bam

# sort by coordinates
samtools view -h $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.mapped.bam | sed '/chrM/d;/random/d;/chrUn/d' | samtools sort -@ 8 -O bam -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.bam

# Remove duplicates with picard
java -Xms32000m -jar $SCRIPT_DIR/picard-2.8.0.jar MarkDuplicates I=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.bam O=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.nodup.bam M=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.dup.metrics.txt REMOVE_DUPLICATES=true ASSUME_SORTED=true

# Remove reads in blacklisted regions
bedtools intersect -abam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.filtered.sortedbyCoord.nodup.bam -b $BLACKLIST -wa -v > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam

# Index bam file
samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam
rm -f $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.bowtie2.sam

```

Perform the same steps for the CAST genome.

```

samtools view -b $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.sam > $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.bam
samtools view -bh -f 3 -F 12 -q 2 $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.bam > $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.mapped.bam
samtools view -h $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.mapped.bam | sed '/chrM/d;/random/d;/chrUn/d' | samtools sort -@ 8 -O bam -o $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.bam
java -Xms32000m -jar $SCRIPT_DIR/picard-2.8.0.jar MarkDuplicates I=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.bam O=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.nodup.bam M=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.dup.metrics.txt REMOVE_DUPLICATES=true ASSUME_SORTED=true

bedtools intersect -abam $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.filtered.sortedbyCoord.nodup.bam -b $BLACKLIST -wa -v > $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam

samtools index $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam
rm -f $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.bowtie2.sam

```

### Convert Cast mapping to corresponding B6 coordinates using lapels, suspenders

It is recommended to use easy-install or pip (http://packages.python.org/distribute/easy_install.html) for the installation. 
See https://pypi.org/project/lapels/, https://pypi.org/project/suspenders/

```
easy_install lapels
# or
pip install lapels

easy_install suspenders
#or 
pip install suspenders
```

```
# Run pylapels on B6 alignments across 8 threads (without new headers and allowing forced overwrites), which uses the provided MOD file to convert to a common set of B6 coordinates. 
pylapels -n -f -p 8 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.lapels.medpar.bam $MOD_B6 $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.final.bam

# Run pylapels on CAST alignments across 8 threads (without new headers and allowing forced overwrites), which uses the provided MOD file to convert to a common set of B6 coordinates.
pylapels -n -f -p 8 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.lapels.medpar.bam $MOD_CAST $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.final.bam

# Uses pysuspenders to assign each read to an allele of origin
pysuspenders $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.lapels.medpar.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.lapels.medpar.bam

```

### Determine allele of origin for each read

Three possible outcomes: B6-specific, Cast specific, or equally likely. 
Split reads into each of these groups.
Also create files that have strain specific + random 1/2 of non-specific

```  
# Get B6 specific reads from suspenders-processed alignments
samtools view -b -d po:1 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.bam

# Get CAST specific reads from suspenders-processed alignments
samtools view -b -d po:2 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po2.bam

# Get non-specific reads from suspenders-processed alignments
samtools view -b -d po:3 -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.bam

# Randomly split non-specific reads into 2 files
samtools view -H $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.bam > $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam
cp $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sam

samtools view $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.bam | awk '{if(NR%4==1 || NR%4==2){print >> \"$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam\"} else {print >> \"$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sam\"}}
samtools view -h -b -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.bam
samtools view -h -b -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sam; samtools sort -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.2.bam

# Merge B6 specific reads with 1/2 of non-specific reads
samtools merge -f -o $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.po3.merged.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po1.sorted.bam $OUTPUT_DIR/$sample_ID/${sample_ID}.B6.CAST.suspenders.po3.1.sorted.bam

# Merge CAST specific reads with 1/2 of non-specific reads
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
The saf was provided in processing_files or can be downloaded https://www.dropbox.com/scl/fo/uvysvn3jsastgo5s1xjn2/AFoLX69qDNrxZagIuvxuCdE?rlkey=sj6tjyoc4n18vwyx0xtyuueej&dl=0
The SAF file is a specific format for featureCounts. For more details see here https://subread.sourceforge.net/featureCounts.html

```
CASTBAM=$OUTPUT_DIR/$sample_ID/${sample_ID}.CAST.po3.merged.bam
B6BAM=$OUTPUT_DIR/$sample_ID/${sample_ID}.B6.po3.merged.bam

featureCounts -p -B -F SAF -T 8 -a saf/treg_pks.saf -o B6_counts.bed $B6BAM
featureCounts -p -B -F SAF -T 8 -a saf/treg_pks.saf -o CAST_counts.bed $CASTBAM

```

## Extended peaks from existing bed files

To extend peaks to different sizes for modeling, we use the following R code (extendBed.R)
```
#Run as 
Rscript extendBed.R [bedfile] [extension_size] [outfile]

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
# prepares files with vcf's
MMARGE.pl prepare_files -ind cast -files vcf_snps.vcf,vcf_indels.vcf -core 12 -genome genome_name

# create genome fasta
MMARGE.pl create_genomes -genome genome_name -ind cast

# shifts B6 files to CAST coordinates (to enable extraction of region FASTAs directly from CAST genome fasta)
MMARGE.pl shift_to_strain -dir directory -files peaks.bed -ind cast −bed
```

Once these genomes have been generated and beds shifted to appropriate coordinates, can extract fastas with bedtools:
here, bed is the version shifted to cast coordinates above and genome is the cast genome fasta generated above

```
bedtools getfasta -name -fi mmarge_cast_genome_combined.fa -bed ${bed} -fo ${bed}.fa
```

#### Manual variant insertion

Alternatively, can insert variants individually from a vcf file as follows in insert_variants.py
This script inserts the variants into the FASTAs from the BED file. The reason this is necessary is because the variants in the vcf and the peaks in the bed are in the same coordinates so this uses those add the corresponding variants into just the regions of interest. It does not create the full genome.

On the other hand, MMARGE per above does create a full genome and then we can use MMARGE to get shifted bed coordinates that can be used to extract the strain-specific region FASTAs. 

```
python insert_variants.py -f [fasta_file] -s [snp_file] -d [indel_file] -o [output_fasta_path] [optional --filter_pass]

```


## Processing specific to single-cell data:

The following are some specific details of how to process data for single-cell ATAC data.

### Map all reads to both genomes

We use bowtie2 to map reads to both genomes (showing for scATAC data here).

```
ref = reference .fa (genome fasta) file
refout = name for bowtie index built from genome reference
fastq1 = R1 and fastq2 = R2 (R3 in 10x scATAC) to align
samout = name of samfile output

bowtie2-build --threads 8 ${ref} ${refout}
bowtie2 --threads 16 -x ${refout} -1 ${fastq1} -2 ${fastq2} -S ${samout}

# here showing processing for sam files aligned across 2 different lanes (samf1 = L1, samf2 = L2
# both corresponding to samout from previous)
samtools view -b ${samf1} > ${samf1}.bam
samtools sort -@ 8 ${samf1}.bam -o ${samf1}.sort.bam
samtools index ${samf1}.sort.bam

echo "samtools 2"
samtools view -b ${samf2} > ${samf2}.bam
samtools sort -@ 8 ${samf2}.bam -o ${samf2}.sort.bam
samtools index ${samf2}.sort.bam

# merging across lanes
echo "samtools merge"
samtools merge -@ 8 ${samf1}.${samf2}.merge.bam ${samf1}.sort.bam ${samf2}.sort.bam

rm ${samf1}.bam
rm ${samf2}.bam

# index files
samtools index ${samf1}.${samf2}.merge.bam
samtools index ${samf1}.${samf2}.merge.bam

```


### Convert Cast mapping to corresponding B6 coordinates usign lapel, suspenders

```
pylapels -n -p 12 -o $outdir/${samf1}.${samf2}.merge.lapels.medpar.bam ${mod} ${samf1}.${samf2}.merge.bam
samtools sort -n -o ${indir}/${b6bam}.sort.bam ${indir}/${b6bam}.bam
samtools sort -n -o ${indir}/${castbam}.sort.bam ${indir}/${castbam}.bam

samtools index ${indir}/${b6bam}.sort.bam
samtools index ${indir}/${castbam}.sort.bam

pysuspenders ${outdir}/${outbam}.bam ${indir}/${b6bam}.sort.bam ${indir}/${castbam}.sort.bam

```

### Determine allele of origin for each read

Three possible outcomes: B6-specific, Cast specific,or equally likely)

```  
# B6 specific
samtools view -b -d po:1 -o ${bamdir}/${bam}_po1.bam ${bamdir}/${bam}
# Cast specific
samtools view -b -d po:2 -o ${bamdir}/${bam}_po2.bam ${bamdir}/${bam}
# non-specific
samtools view -b -d po:3 -o ${bamdir}/${bam}_po3.bam ${bamdir}/${bam}

```

### Use sinto to convert aligned bam into fragments file for downstream processing
This step is for single cell data only. Sinto can be found here (https://github.com/timoast/sinto?tab=readme-ov-file)
This creates a special format called a fragments file used in scATAC data analysis. This provides the regex of where sinto should find the barcode in the fasta. https://timoast.github.io/sinto/basic_usage.html#create-scatac-seq-fragments-file 

```
samtools view -H ${bamdir}/${bam} | sed -e 's/SN:\([0-9XY]\)/SN:chr\1/' -e 's/SN:MT/SN:chrM/' | samtools reheader - ${bamdir}/${bam} > ${bamdir}/${bam}_chr.bam

samtools sort -o ${bamdir}/${bam}_chr.sorted.bam ${bamdir}/${bam}_chr.bam
samtools index ${bamdir}/${bam}_chr.sorted.bam

cd ${outdir}

# create scATAC fragments file, providing pattern of barcode to sinto
sinto fragments -b ${bamdir}/${bam}_chr.sorted.bam -p 8 -f ${bam}.fragments.bed --barcode_regex "[^:]*"

# sort, compress, and index
sort -k1,1 -k2,2n ${bam}.fragments.bed > ${bam}.fragments.sort.bed
bgzip -@ 8 ${bam}.fragments.sort.bed
tabix -p bed ${bam}.fragments.sort.bed.gz

rm ${bam}.fragments.bed
```

### Randomly split reads that are equally likely from each genome across two alleles to create combined fragments files

This step is optional, depending on whether one wants to consider reads that cannot be mapped to either one of the genomes. 

```
python split_fragments.py -F ${bam}.fragments.sort.bed.gz

python split_fragments.py -F ${fragfile_pre}.bed.gz

# sort, compress, and index
sort -k1,1 -k2,2n ${fragfile_pre}.split1.bed > ${fragfile_pre}.split1.sort.bed
bgzip -@ 4 ${fragfile_pre}.split1.sort.bed
tabix -p bed ${fragfile_pre}.split1.sort.bed.gz


sort -k1,1 -k2,2n ${fragfile_pre}.split2.bed > ${fragfile_pre}.split2.sort.bed
bgzip -@ 4 ${fragfile_pre}.split2.sort.bed
tabix -p bed ${fragfile_pre}.split2.sort.bed.gz

```

Combine Fragments across files

```
python combine_fragments.py -A ${fragfile1} -B ${fragfile2} -O ${outpath_pre}.bed

# sort, compress, and index
sort -k1,1 -k2,2n ${outpath_pre}.bed > ${outpath_pre}.sort.bed
bgzip -@ 2 ${outpath_pre}.sort.bed
tabix -p bed ${outpath_pre}.sort.bed.gz
```

### For scATAC, count matrices can be generated using the Signac (https://stuartlab.org/signac/) package:

```
#Run as Rscript 
generate_count_matrix.R [bed_file_path] [fragments_file_path] [out_file]

```
