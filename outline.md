# A contrastive sequence-to-function modeling approach for investigating allele-specific epigenomic regulation

## TODO: Github repo

- [ ] Link to GEO accession numbers
- [ ] Link to processed data on zenodo
- [ ] Link to trained models on zenodo

- [ ] Tutorials/docs for general use or repo
	-  [ ] Kernel analysis

- [ ] Scripts for paper (bash, py. ipynb)

	- [ ] Data processing (Kait, Xinming)
		- [ ] Generate input sequences for both genomes in given length
		- [ ] Generate .bam files for both genomes
		- [ ] Transform .bams into count matrices
		- [ ] Normalize counts and compute fold changes for input to model

	- [ ] Model training
		- [ ] Define model specifics
		- [ ] Train model - select training parameters and loss
		- [ ] Load model
		- [ ] Model training curves
		- [ ] Model performance plots
	
	- [ ] Model analysis
		- [ ] Generate attributions
		- [ ] Extract and cluster motifs from attributions
		- [ ] Determine OCRs with model knowledge
		- [ ] Determine main variants from ISM
		- [ ] Plot attributions for selected sequences
		- [ ] Assign TFs to extracted motifs
		- [ ] Plot PWM tree for extracted motifs
		- [ ] Determine the fraction of motifs that are disrupted

## Figure 1
Model + Performance
- Performance for 2ATAC + 2CHIP + 2RNA-seq
	- [ ] Add Kumba and Liang and get their data processing pipeline for RNA-seq? 
	- [ ] Ask for accession number

## Figure 2
Motif and variant analysis for CHIP, ATAC, + RNA

## Figure S1
Additional Performance plots
- [ ] Spearman Counts + Ratio
- [ ] Comparison to replicates when available
- [ ] Performance for varying input sequence length
- [ ] Performance for varying model depth

## Figure S2
Correlation between ISM from different models (CHIP, ATAC, RNA-seq?)
Correlation between DeepSHAP from different models

## Figure S3
PWMs in attributions in CHIP and ATAC + RNA-seq

## Figure S4
Predictable OCRs and variant effect predictions CHIP

## Figure S5
Predictable OCRs and variant effect predictions ATAC

## Figure S6
Predictable OCRs and variant effect predictions RNA-seq

## Figure S7
Motifs in RNA-seq

## Figure S8
Motif analysis with DeepSHAP or ISM (the complementary of Figure 2)

