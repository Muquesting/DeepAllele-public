# Analyze_model

- [x] Add additional distance measures to cluster_seqlets
- [x] Allow torch_align_motifs to only generate correlation matrix
	- [ ] Allow sparse correlation matrix
- [ ] use argparse throughout
- [ ] allow extract_motifs to work with single sequence only
- [ ] Check if giving motif locations to plot_attributions still works
- [ ] Provide plot_attributions to use predicted ISM from other file
- [ ] Put the trained model parameters on zenodo to download into models/ with wget
	- [ ] Have a download .sh
	- [ ] Have the same for processed data
- [ ] i.e. 2 ATAC, 2 Chip, 2 RNA + 2 inits
- [ ] Document pre-processing
- [ ] Should we use tangermeme? 	
	- [ ] We use dinuc shuffle and deepliftshap
- [ ] Include readfromckpt for load surrogate model
	- [ ] and potentially at other places
- [ ] Put sequences and attributions in npz together
	- [ ] Change input to other scripts
- [ ] Include scripts to extract variants and make variant calls.

# Compare_model

- [ ] Add folder for Compare_models
- [ ] Define sets of well predicted ratios
- [ ] Plot number of sets in venn diagram
- [ ] Determine number of main variants from MH model in motifs from SH model
	- [ ] Compare to number in MH
- [ ] Plot main var effects against each other
- [ ] Plot main var effects against motif diff
- [ ] Plot motif diff or means against each other

