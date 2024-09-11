#!/bin/bash

set -e
set -u
set -x

# Analyze learned regulatory grammar with sequence attributions



## 1. Make predictions with the model and determine trusted set of sequences

### Get models and processed data from the paper

### Set relevant PATHs for subsequent analysis
# Set shortcuts to scripts, models, data, and location for output

scriptdir=/PATH/to/DeepAllele-public/scripts/ #path to scripts
intdir=${scriptdir}/Analyze_models/ $ path to analysis scripts

modeldir=/PATH/to/${dataset}/models/ # path to model parameters
datadir=/PATH/to/${dataset}/data/ # path to data 
outdir=/PATH/to/${dataset}/results/ # path to results
# For example 
dataset=atac0/ # for some ATAC-seq data set

# For example
modeltype=mh # for multi-head, 
initialization=init0 # seed for model initialization
resultdir=${outdir}/${modeltype}/${initialization}/

ourmodelsdir=${modeldir}/${modeltype}/${initialization}/

### Get one-hot encodings of test set sequences, and measured values

hdf5_path=${ourmodelsdir}$<Name of hdf5 file>
batch_id=XXX

python ${intdir}get_predictions.py --which_fn save_seqs_obs_labels --save_dir ${datadir} --hdf5_path ${hdf5_path} --batch_id ${batch_id}
# Returns
seqs=${datadir}${batch_id}_${train_or_val}_seqs.npy
obs=${datadir}${batch_id}_${train_or_val}_obs.npy
seq_labels=${datadir}${batch_id}_${train_or_val}_seq_labels.npy

### Get model predictions for one-hot encoded test sequences

ckpt_path=${ourmodelsdir}<model file.ckpt>
device=gpu
python ${intdir}get_predictions.py --which_fn get_predictions --save_dir ${outdir} --ckpt_path ${ckpt_path} --seqs_path ${seqs} --mh_or_sh ${modeltype} --device ${device}
# Returns
preds=${outdir}${modeltype}_predictions.txt

### Determine sequences for which model makes reasonable ratio predictions

python ${intdir}zscore_datacolumn.py $preds --column -2

# Use cut-offs of abs(ratio) > 1 and abs(z-score(pred.ratio))) > 1.65 to classify the data points
ratio_cut=1.0
predz_cut=1.65
python ${intdir}classify_prediction.py $vals ${preds%.txt}_zscore.txt $ratio_cut $predz_cut --column1 -1 --column2 -1
# Returns
validlist=${preds%.txt}_zscore_-1_eval_on_${vals%.txt}_-1_cut${ratio_cut}_and_${predz_cut}.txt

# Plot measured allelic ratios versus predicted and color by class
python ${intdir}scatter_comparison_plot.py $vals $preds 'Measured allelic log2-ratio' 'Predicted allelic log2-ratio' --column -1 -2 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_Measuredvspredicted_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist ${validlist} --contour log --cmap grey,red --legend

## 2. Compute ISM values for all variants between two alleles and identify main variant and its impact

### Load model parameters and make predictions for inserting each variant present into each genome. Save (variant prediction - reference prediction) for each variant, each genome. 


python ${intdir}get_variant_ism.py --save_dir ${outdir} --seqs_path ${seqs} --ckpt_path ${ckpt_path} --device ${device} 
# Returns
all_variant_ism=${outdir}variant_ism_res.csv


variant_info=${outdir}variant_info.csv
aligned_sequences=${outdir}aligned_seqs.npy


python ${intdir}averageISM.py $all_variant_ism
# Returns
var=${outdir}/${modeltype}/${initialization}/ISM_avg_variant_effect.txt

python ${intdir}compute_variant_prediction.py $var
# Returns
varpred=${var%.txt}_seqpred.txt

### Check how well the sum if individual variant effects represents predictions

python ${intidir}scatter_comparison_plot.py $varpred $pred 'Sum ISM effects on allelic log2-ratio' 'Predicted allelic log2-ratio' --column -1 -2 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_SumVarDAvspredicted_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist $validlist --contour log --cmap grey,red

### Determine main variant
python ${intdir}determine_main_variant.py $var
# Returns
mainvar=${var%.txt}_mainvar.txt

### Plot main variant effect versus predicted effect with coloring of predictable cases

python ${intdir}scatter_comparison_plot.py $mainvar $pred 'Main variant ISM effect on allelic log2-ratio' 'Predicted allelic log2-ratio' --column -1 -2 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_mainVarDAvspredicted_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist $validlist --contour log --cmap grey,red

### Plot main variant effect versus measured effect with coloring of predictable cases
python ${intdir}scatter_comparison_plot.py $vals $mainvar 'Measured allelic log2-ratio' 'Main variant ISM effect on allelic log2-ratio' --column -1 -1 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_mainVarDAvsmeasured_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist $validlist --contour log --cmap grey,red -xlim -2.5,3.5 -ylim -2.5,3.

## 3. Compute sequence attributions

### Perform ISM on test set sequences

python ${intdir}get_attributions.py --which_fn get_ism_res --save_dir ${outdir} --ckpt_path ${ckpt_path} --seqs_path ${seqs} 

# Returns
ism=${outdir}ism_res.npy


python ${intdir}correct_ism.py $ism
# Returns
ismatt=${outdir}/${modeltype}/${initialization}/ism_res.imp.npy

### Perform DeepLift on sequences

python ${intdir}get_attributions.py --which_fn get_deeplift_res --save_dir ${outdir} --ckpt_path ${ckpt_path} --seqs_path ${seqs_path} 

# Returns
deeplift=${outdir}deeplift_attribs.npy



### If available, check correlation between ISM from two models with different random initialization
# Computes correlations between attributions maps and plots histgram

ismatt0=${outdir}/${modeltype}/${initialization0}/ism_res.imp.npy
ismatt1=${outdir}/${modeltype}/${initialization1}/ism_res.imp.npy

python ${intdir}compute_attribution_correlation.py $ismatt0 $ismatt1 $labels $labels 'ISM_0 vs ISM_1' ${outdir}/${modeltype}/Correlation_hist_ism0_vs_ism1.jpg --seqs $seqs $seqlabels input

### Visualize individual attribution maps

# Separate attribution maps for input to visualization
plotdir=${outdir}/attributions_plots
mkdir $plotdir
mkdir ${datadir}/sequences
python ${intdir}separate_attributionmaps.py ${labels} ${ismatt} --outdir $plotdir
python ${intdir}separate_attributionmaps.py ${labels} ${seqs} --outdir ${datadir}/sequences

# Plot individual attribution maps of selected regions
ism_example=${outdir}seq_idx_99_*_ism_res.imp.npy
seq_example=${datadir}seq_idx_99_*.npy
python ${intdir}plot_attributions.py $ism_example ${seq_example} --ratioattributions


## 4. Extract seqlets from attributions and analyze learned mechanisms systematically

### Identify motifs in attributions and extract their attributions scores for all four bases


python ${intdir}extract_motifs.py $labels $ismatt $seqs 1.96 1 4 --normed --ratioattributions
# Returns
seqlets=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_seqlets.meme # Z-scored and sign adjusted seqlets from attribution map
seqstats=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_seqmotifstats.txt # Motif statistics for each sequence, how many common, and how many unique
seqleteffects=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_seqleteffects.txt # Mean effect, Delta mean, Max effect, delta max effect
seqletloc=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_otherloc.txt # Locations of motif in other allele

### Cluster extracted motifs
python ${intdir}cluster_seqlets.py $seqlets complete --distance_threshold 0.05 --distance_metric correlation_pvalue --clusternames --reverse_complement
# Returns combined motifs for all clusters
clustermotifs=${seqlets%.meme}ms4_cldcomplete0.05corpvapfms.meme
seqletclusters=${seqlets%.meme}ms4_cldcomplete0.05corpva.txt

python ${intdir}cluster_seqlets.py $seqlets complete --distance_threshold 0.05 --distance_metric correlation_pvalue --clusternames --reverse_complement --approximate_cluster_on 15000

python ${intdir}cluster_seqlets.py $seqlets $seqletclusters --clusternames --reverse_complement

python ${intdir}count_seq_for_clusters.py $seqletclusters
# Returns
clusterperc=${seqletclusters%.txt}_motifpercinseq.txt

python ${intdir}select_largest_clusters.py $seqletclusters 20
# Returns list of clusters that are have more than 20 members
clusterlist20=${seqletclusters%.txt}_Ngte20list.txt

### Plot PWMs of clusters 
python ${intdir}plot_pwm_logos.py $clustermotifs --basepwms $seqlets --clusterfile $seqletclusters --select 19,21

### Plot the extracted motifs in tree with percentage of sequence that contain motif
python ${intdir}plot_pwm_tree.py $clustermotifs --set $clusterlist20 --savefig ${clusterlist20%list.txt} --reverse_complement --joinpwms 0.2 --savejoined --pwmfeatures $clusterperc barplot=True+ylabel='in % of sequences'
joinedmotifs=${clusterlist20%list.txt}joined0.2pfms.meme
joinedmotifperc=${clusterlist20%list.txt}joined0.2pfmfeats.npz

### Normalize clustered (hypothetical) Contribution Weight Matrices and find matching TF motifs with Tomtom

python ${intdir}parse_motifs_tomeme.py.py $clustermotifs --standardize --exppwms --norm --strip 0.1 --round 3 --set $clusterlist20
# Returns normalized Position probability matrices
clustermeme=${seqlets%.meme}_Ngte20list.meme


motifdatabase=PATH/to/motifdatabase/
tfdatabase=${motifdatabase}JASPAR2020_CORE_vertebrates_non-redundant_pfms.TFnames.meme
tomtom -thresh 0.5 -dist pearson -text $clustermeme $tfdatabase > ${clustermeme%.meme}.tomtom.tsv
# return tomtom tsv file
clustertomtom={clustermeme%.meme}.tomtom.tsv

python ${intdir}replace_motifname_with_tomtom_match.py ${clustertomtom} q 0.05 $clustermeme --only_best 4 --reduce_clustername '_' --reduce_nameset ';' --generate_namefile
# Returns
clustertfname={clustertomtom%.tomtom.tsv}q0.05best4_altnames.txt
python ${intdir}plot_pwm_tree.py $clustermotifs --set $clusterlist20 --savefig ${clusterlist20%list.txt} --reverse_complement --pwmfeatures $clusterperc barplot=True+ylabel='in % of sequences' --pwmnames $clustertfname

### Alternatively, use the joined motifs from the first plot to visualize the detected motifs in compact format

python ${intdir}parse_motifs_tomeme.py.py $joinedmotifs --standardize --exppwms --norm --strip 0.1 --round 3 

tomtom -thresh 0.2 -dist pearson -text ${joinedmotifs%.meme}.meme $tfdatabase > ${joinedmotifs%.meme}.tomtom.tsv

#### Make a name file for the combined clusters from tomtom tsv
python ${intdir}replace_motifname_with_tomtom_match.py ${joinedmotifs%.meme}.tomtom.txt q 0.05 ${joinedmotifs%.meme}.meme --only_best 4 --reduce_nameset ';' --generate_namefile

#### Plot the compact tree with the associated TFs and short names for the clusters
python ${intdir}plot_pwm_tree.py $joinedmotifs --savefig ${joinedmotifs%.meme} --reverse_complement --pwmnames ${joinedmotifs%.meme}q0.05best4_altnames.txt --pwmfeatures ${joinedmotifperc}.npz barplot=True+ylabel='in % of sequences'

## 5. Analyze mechanisms that are affected by variants. 

### Determine if main variants are inside a motif

python ${intdir}motif_variant_location.py $mainvar $seqletclusters $seqletloc $validlist --savemotiflist --TFenrichment $clustertomtom
# Returns pie chart and list of clusters affected by a variant






