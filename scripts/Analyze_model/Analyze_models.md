# Analyze learned regulatory grammar with sequence attributions

Here, we demonstrate how to use sequence attributions from trained DeepAllele models to make predictions for variant effect sizes, and to determine the regulatory mechanisms that lead to allelic imbalances in the data set. DeepAllele shows improved performance for predictions of allelic imbalances over classical models that were only trained on counts. The ratio head uses sequences from both alleles as input to predict allelic ratios. This additional objective improves predictions for allelic ratios. Attributions from the "ratio"-head return importance scores for both alleles. These sequence attributions contain motifs that are important for predicting both allelic counts, and the ratio. The differences between the size of motifs in the two allelic sequences is directely related to the ratio predictions.


## 1. Make predictions with the model and determine trusted set of sequences
No model is perfect! To determine which model predictions we can trust, we use the model's predictions for unseen sequences in the test set to determine for which it might have learned the correct mechanisms. 

### Get models and processed data from the paper
Use `download_models.sh` to download the processed data and trained models from Zenodo. This bash script also initializes PATH structure for `atac/`, `chip/`, and `rna/` with `models/`, `data/` and `results/` in each of them.

### Set relevant PATHs for subsequent analysis
```
# Set shortcuts to scripts, models, data, and location for output

scriptdir=/PATH/to/DeepAllele-public/scripts/ #path to scripts
intdir=${scriptdir}/Analyze_models/ $ path to analysis scripts

modeldir=/PATH/to/${dataset}/models/ # path to model parameters
datadir=/PATH/to/${dataset}/data/ # path to data 
outdir=/PATH/to/${dataset}/results/ # path to results
# For example 
dataset=atac0/ # for some ATAC-seq data set
```

We assume that the results of this analysis will be saved in the following structure:
```
# For example
modeltype=mh # for multi-head, 
initialization=init0 # seed for model initialization
resultdir=${outdir}/${modeltype}/${initialization}/
```
`initialization` can be different random starting points from which the model learns, different model architectures, or anything that defines the models specific training run. It describes a variant of the model.

We assume that your models are located in the following structure:
```
ourmodelsdir=${modeldir}/${modeltype}/${initialization}/
```

### Get one-hot encodings of test set sequences, and measured values

Extract data from hdf5 path 
```
hdf5_path=${ourmodelsdir}$<Name of hdf5 file>
batch_id=XXX

python ${intdir}get_predictions.py --which_fn save_seqs_obs_labels --save_dir ${datadir} --save_label ${save_label} --hdf5_path ${hdf5_path} --batch_id ${batch_id}
# Returns
seqs=${datadir}${save_label}_${batch_id}_${train_or_val}_seqs.npy
obs=${datadir}${save_label}_${batch_id}_${train_or_val}_obs.npy
seq_labels=${datadir}${save_label}_${batch_id}_${train_or_val}_seq_labels.npy
```

### Get model predictions for one-hot encoded test sequences

Load a model from checkpoint and make predictions for sequences
```
ckpt_path=${ourmodelsdir}<model file.ckpt>
save_label=YYY
device=gpu
python ${intdir}get_predictions.py --which_fn get_predictions --save_dir ${outdir} --ckpt_path ${ckpt_path} --seqs_path ${seqs} --save_label ${save_label} --mh_or_sh ${modeltype} --device ${device}
# Returns
preds=${outdir}${save_label}_${modeltype}_predictions.txt
```

### Determine sequences for which model makes reasonable ratio predictions

Ratio predictions are not perfectly scaled compared to the real data distribution because the majority of sites experience no allelic imbalance. We use Z-score normalization to determine which ratio predictions are significant.
```
python ${intdir}zscore_datacolumn.py $preds --column -2

# Use cut-offs of abs(ratio) > 1 and abs(z-score(pred.ratio))) > 1.65 to classify the data points
ratio_cut=1.0
predz_cut=1.65
python ${intdir}classify_prediction.py $vals ${preds%.txt}_zscore.txt $ratio_cut $predz_cut --column1 -1 --column2 -1
# Returns
validlist=${preds%.txt}_zscore_-1_eval_on_${vals%.txt}_-1_cut${ratio_cut}_and_${predz_cut}.txt

# Plot measured allelic ratios versus predicted and color by class
python ${intdir}scatter_comparison_plot.py $vals $preds 'Measured allelic log2-ratio' 'Predicted allelic log2-ratio' --column -1 -2 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_Measuredvspredicted_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist ${validlist} --contour log --cmap grey,red --legend
```

## 2. Compute ISM values for all variants between two alleles and identify main variant and its impact

### Load model parameters and make predictions for inserting each variant present into each genome. Save (variant prediction - reference prediction) for each variant, each genome. 

```

python ${intdir}get_variant_ism.py --save_dir ${outdir} --seqs_path ${seqs} --ckpt_path ${ckpt_path} --device ${device} --save_label ${save_label}

# Returns
all_variant_ism=${outdir}{save_label}_variant_ism_res.csv
```

get_variant_ism.py will save two additional intermediate files used for computing variant ism: 

```
variant_info=${outdir}variant_info.csv
aligned_sequences=${outdir}aligned_seqs.npy
```
`variant_info` describes all variants present in the genomes by index and sequence. `aligned_sequences` contains aligned sequences used to insert variants at the correct index in each genome. 

Average ISMs of variants from two alleles

```
python ${intdir}averageISM.py $all_variant_ism
# Returns
var=${outdir}/${modeltype}/${initialization}/ISM_avg_variant_effect.txt
```

Compute the (prediction from) the sum of all ISMs
```
python ${intdir}compute_variant_prediction.py $var
# Returns
varpred=${var%.txt}_seqpred.txt
```

### Check how well the sum if individual variant effects represents predictions

```
python ${intidir}scatter_comparison_plot.py $varpred $pred 'Sum ISM effects on allelic log2-ratio' 'Predicted allelic log2-ratio' --column -1 -2 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_SumVarDAvspredicted_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist $validlist --contour log --cmap grey,red
```

### Determine main variant
```
python ${intdir}determine_main_variant.py $var
# Returns
mainvar=${var%.txt}_mainvar.txt
```

### Plot main variant effect versus predicted effect with coloring of predictable cases

```
python ${intdir}scatter_comparison_plot.py $mainvar $pred 'Main variant ISM effect on allelic log2-ratio' 'Predicted allelic log2-ratio' --column -1 -2 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_mainVarDAvspredicted_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist $validlist --contour log --cmap grey,red
```

### Plot main variant effect versus measured effect with coloring of predictable cases
```
python ${intdir}scatter_comparison_plot.py $vals $mainvar 'Measured allelic log2-ratio' 'Main variant ISM effect on allelic log2-ratio' --column -1 -1 --savefig ${outdir}/${modeltype}/${initialization}/LogAllelicRatio_mainVarDAvsmeasured_predictabilitycolor --alpha 0.8 --plotdiagonal --zeroxaxis --zeroyaxis --vlim 0,1 --size 5 --lw 0 --colorlist $validlist --contour log --cmap grey,red -xlim -2.5,3.5 -ylim -2.5,3.
```

## 3. Compute sequence attributions

### Perform ISM on test set sequences

```
python ${intdir}get_attributions.py --which_fn get_ism_res --save_dir ${outdir} --ckpt_path ${ckpt_path} --seqs_path ${seqs} --save_label ${save_label}

# Returns
ism=${outdir}${save_label}_ism_res.npy

```

Subtract mean from ISM to get attribution maps.
Attributions from ISM are generated from the zero-sum gauge of the linear approximation. 
```
python ${intdir}correct_ism.py $ism
# Returns
ismatt=${outdir}/${modeltype}/${initialization}/ism_res.imp.npy
```

### Perform DeepLift on sequences

```
python ${intdir}get_attributions.py --which_fn get_deeplift_res --save_dir ${outdir} --ckpt_path ${ckpt_path} --seqs_path ${seqs_path} --save_label ${save_label}

# Returns
deeplift=${outdir}${save_label}__deeplift_attribs.npy

```
By default, this saves hypothetical attributions from a uniform baseline. 
Multipliers represent local attributions, hypothetical attributions are generated from the zero-sum gauge of the linear approximation. Hypothetical attributions are more suited to detect motifs. Briefly, they determine the preference of the model for all four bases at each position, as if these bases were inserted into the sequence. This is similar to the concept of PWMs.


### If available, check correlation between ISM from two models with different random initialization
```
# Computes correlations between attributions maps and plots histgram

ismatt0=${outdir}/${modeltype}/${initialization0}/ism_res.imp.npy
ismatt1=${outdir}/${modeltype}/${initialization1}/ism_res.imp.npy

python ${intdir}compute_attribution_correlation.py $ismatt0 $ismatt1 $labels $labels 'ISM_0 vs ISM_1' ${outdir}/${modeltype}/Correlation_hist_ism0_vs_ism1.jpg --seqs $seqs $seqlabels input
```

### Visualize individual attribution maps

Plot attribution maps for both sequences and the variant effects for each variant between the two alleles.
```
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
```


## 4. Extract seqlets from attributions and analyze learned mechanisms systematically

### Identify motifs in attributions and extract their attributions scores for all four bases

Attributions are normalized to Z-scores and significant positions are determined with a threshold of 1.96. Seqlets are extracted if 4 or more subsequent positions with at max one gap are siginficant. Seqlets are saved in .meme file with the direction of the attributions being adjusted to the sign of their mean. Moreover, statistics are kept about the location of the seqlet in both sequences (after alignment), and the mean difference of these motifs for downstream analysis. 

```
python ${intdir}extract_motifs.py $labels $ismatt $seqs 1.96 1 4 --normed --ratioattributions
# Returns
seqlets=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_seqlets.meme # Z-scored and sign adjusted seqlets from attribution map
seqstats=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_seqmotifstats.txt # Motif statistics for each sequence, how many common, and how many unique
seqleteffects=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_seqleteffects.txt # Mean effect, Delta mean, Max effect, delta max effect
seqletloc=${ismatt%.npy}_seqlets.cut1.96maxg1minsig4_otherloc.txt # Locations of motif in other allele
```

### Cluster extracted motifs
Seqlets are aligned using Pearson correlation coefficient. The p-values for the correlation are computed and used for agglomerative clustering with complete linkage, i.e all seqlets in one cluster have at least a correlation equivalent to 0.05 to all other sequences in the cluster. Alternatively, other linkages or clustering methods can be used.
```
python ${intdir}cluster_seqlets.py $seqlets complete --distance_threshold 0.05 --distance_metric correlation_pvalue --clusternames --reverse_complement
# Returns combined motifs for all clusters
clustermotifs=${seqlets%.meme}ms4_cldcomplete0.05corpvapfms.meme
seqletclusters=${seqlets%.meme}ms4_cldcomplete0.05corpva.txt
```

If memory is an issue, the clusters can be approximated from a random subset, while the left-out PWMs will be aligned to these clusters and assigned to them if they fulfill requirements of the selected linkage. 
```
python ${intdir}cluster_seqlets.py $seqlets complete --distance_threshold 0.05 --distance_metric correlation_pvalue --clusternames --reverse_complement --approximate_cluster_on 15000
```

Can be rerun with cluster assignments to just get the combined PWMs with.
```
python ${intdir}cluster_seqlets.py $seqlets $seqletclusters --clusternames --reverse_complement
```

Determine the percentage of sequences with a motif that this cluster appears in
```
python ${intdir}count_seq_for_clusters.py $seqletclusters
# Returns
clusterperc=${seqletclusters%.txt}_motifpercinseq.txt
```

If the number of clusters is too large, filter noisy motifs by setting a reasonable cutoff for the minimum number of members in these sequences. E.g. only consider clusters that have at least 20 members in either one Allele of 10k sequences.
```
python ${intdir}select_largest_clusters.py $seqletclusters 20
# Returns list of clusters that are have more than 20 members
clusterlist20=${seqletclusters%.txt}_Ngte20list.txt
```

### Plot PWMs of clusters 
If wanted, also plot original pwms aligned to it with `--basepwms` and `--clusterfile`
```
python ${intdir}plot_pwm_logos.py $clustermotifs --basepwms $seqlets --clusterfile $seqletclusters --select 19,21
```

### Plot the extracted motifs in tree with percentage of sequence that contain motif
Use hirarchical clustering to determine the sequence relationship of the extracted clusters and plot it in tree structure. Visualization can combine clusters if they are too similar visually. 
```
python ${intdir}plot_pwm_tree.py $clustermotifs --set $clusterlist20 --savefig ${clusterlist20%list.txt} --reverse_complement --joinpwms 0.2 --savejoined --pwmfeatures $clusterperc barplot=True+ylabel='in % of sequences'
```
`--joinpwms` Joins all motifs that are correlated more than 0.8, or closer than 0.2 in correlation distance. `--savejoined` saves these combined PWMs and their features if given.
```
joinedmotifs=${clusterlist20%list.txt}joined0.2pfms.meme
joinedmotifperc=${clusterlist20%list.txt}joined0.2pfmfeats.npz
```

### Normalize clustered (hypothetical) Contribution Weight Matrices and find matching TF motifs with Tomtom

Filter and normalize CWMs to classical pwms for usage with tomtom
```
python ${intdir}parse_motifs_tomeme.py.py $clustermotifs --standardize --exppwms --norm --strip 0.1 --round 3 --set $clusterlist20
# Returns normalized Position probability matrices
clustermeme=${seqlets%.meme}_Ngte20list.meme
```

Use tomtom to get matching TFs. Be careful that your motif database is "cleaned" and uses TF names and not Motif-IDs as Jaspar f.e.. Tomtom uses Pearson correlation to compare motifs from database to motifs of clusters. We select a q-value (default) cutoff of 0.5 (default) to keep motifs with low p-value but failing q-values, for later summary analysis

```
motifdatabase=PATH/to/motifdatabase/
tfdatabase=${motifdatabase}JASPAR2020_CORE_vertebrates_non-redundant_pfms.TFnames.meme
tomtom -thresh 0.5 -dist pearson -text $clustermeme $tfdatabase > ${clustermeme%.meme}.tomtom.tsv
# return tomtom tsv file
clustertomtom={clustermeme%.meme}.tomtom.tsv
```

Make a name file for the combined clusters from tomtom tsv
```
python ${intdir}replace_motifname_with_tomtom_match.py ${clustertomtom} q 0.05 $clustermeme --only_best 4 --reduce_clustername '_' --reduce_nameset ';' --generate_namefile
# Returns
clustertfname={clustertomtom%.tomtom.tsv}q0.05best4_altnames.txt
```
Plot the clustered sequlets with assigned names
```
python ${intdir}plot_pwm_tree.py $clustermotifs --set $clusterlist20 --savefig ${clusterlist20%list.txt} --reverse_complement --pwmfeatures $clusterperc barplot=True+ylabel='in % of sequences' --pwmnames $clustertfname
```

### Alternatively, use the joined motifs from the first plot to visualize the detected motifs in compact format

Normalize the joined and combined PWM from the previous plot
```
python ${intdir}parse_motifs_tomeme.py.py $joinedmotifs --standardize --exppwms --norm --strip 0.1 --round 3 
```

Run tomtom
```
tomtom -thresh 0.2 -dist pearson -text ${joinedmotifs%.meme}.meme $tfdatabase > ${joinedmotifs%.meme}.tomtom.tsv
```

#### Make a name file for the combined clusters from tomtom tsv
```
python ${intdir}replace_motifname_with_tomtom_match.py ${joinedmotifs%.meme}.tomtom.txt q 0.05 ${joinedmotifs%.meme}.meme --only_best 4 --reduce_nameset ';' --generate_namefile
```

#### Plot the compact tree with the associated TFs and short names for the clusters
```
python ${intdir}plot_pwm_tree.py $joinedmotifs --savefig ${joinedmotifs%.meme} --reverse_complement --pwmnames ${joinedmotifs%.meme}q0.05best4_altnames.txt --pwmfeatures ${joinedmotifperc}.npz barplot=True+ylabel='in % of sequences'
```

## 5. Analyze mechanisms that are affected by variants. 

### Determine if main variants are inside a motif

Compare locations of motifs and main variants and determine if the main variant is obviously disturbing TF binding
```
python ${intdir}motif_variant_location.py $mainvar $seqletclusters $seqletloc $validlist --savemotiflist --TFenrichment $clustertomtom
# Returns pie chart and list of clusters affected by a variant
```
`--TFenrichment` uses all TF names in the tomtom file associated with a cluster (p<0.05) that is affected by a variant, then sums over the -log10 p-values multiplied with the number of affected motifs for that cluster, and reassigns the TF names of the clusters based on the TF with the highest probability to affect these motifs. We call these motifs TF-like motifs to summarize which TF is likely affecting the motif based on the amount of affected motif and not only on the often noisy motif matches.






