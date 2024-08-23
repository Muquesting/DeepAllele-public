import numpy as np
import sys, os
from functools import reduce
import matplotlib.pyplot as plt

from DeepAllele.io import readtomtom
from DeepAllele.plotlib import piechart
import argparse

'''
Uses location of identifies main variants and compares it to location of motifs
Summarizes how often motif clusters are affected by the main variant
If given tomtom names, will compute enrichment of TF names in the motifs that
are affected and return percentages for TF-like clusters instead.

'''

def assign_motifcluster_to_location(motifclusters, motifalign):
    '''
    Extract motif location and assign each location the defined cluser id
    '''
    motifloc = []
    motifclust = []
    for m, mn in enumerate(motifclusters[:,0]):
        loc = np.arange(int(motifalign[m,1].split(',')[0]), int(motifalign[m,1].split(',')[-1])+1, dtype = int)
        motifloc.append(loc)
        motifclust.append([motifclusters[m,1] for l in range(len(loc))])
    motifloc = np.array(motifloc)
    motifclust = np.array(motifclust)
    return motifloc, motifclust

def variant_locations(mainvar):
    '''
    returns int with variant locations for each sequence
    '''
    varloc = []
    for var in mainvar[:,1]:
        if ':' in var:
            vl = var.split(':')
            vl = np.arange(int(vl[0]), int(vl[1])+1)
        else:
            vl = np.arange(int(var), int(var)+1)
        varloc.append(vl)
    varloc = np.array(varloc)
    return varloc


def assign_cluster_to_variant(mainvar, varloc, motifloc, motifclust):
    varclust = []
    for v in range(len(mainvar)):
        seq = mainvar[v,0]
        potmot = np.isin(motifseq, seq)
        moloc = motifloc[potmot]
        if len(moloc) > 0:
            inter = np.isin(np.concatenate(moloc), varloc[v])
            hitclust, hn = np.unique(np.concatenate(motifclust[potmot])[inter], return_counts = True)
            if len(hitclust) == 1:
                varclust.append(hitclust[0])
            elif len(hitclust) > 1:
                varclust.append(hitclust[np.argmax(hn)])
        else:
            varclust.append('NAN')
    return np.array(varclust)

def assign_tfnames_to_clusters_frequencies(tfnames, pvalues, tfcluster, clusters, nseqincluster, add = '-like'):
    # Determine set of assigned tfs for any clusters
    tfs = np.unique(tfnames)
    sigtfs = np.zeros(len(tfs)) # saves the total amount of significance
    for t, tf in enumerate(tfs):
        tfc = tfcluster[tfnames == tf]
        tfp = pvalues[tfnames == tf]
        for c, tc in enumerate(tfc):
            # significance of tf is percentage of that cluster times the 
            # p-value of the assignment
            mask = clusters==tc
            if np.sum(mask) > 0:
                sigtfs[t] += nseqincluster[clusters==tc] * tfp[c]
            
    # sort tfs by these significance
    sorttfs = np.argsort(-sigtfs)
    # Assign new names to clusters
    ncl = np.copy(clusters)
    ncl = ncl.astype('<U100')
    ncl[:] = '-'
    for s in sorttfs:
        tfc = tfcluster[tfnames == tfs[s]]
        mask = np.isin(clusters, tfc)
        mask = (ncl == '-') * mask
        # Assign all cluster without a name the name of the most
        # significant TF
        ncl[mask] = tfs[s]+add
    # Assign clusters without a TF name, the 'old' cluster id
    if '-' in ncl:
        ncl[ncl == '-'] = clusters[ncl=='-']
    # Join clusters if they were assigned the same TF
    clusters = np.unique(ncl)
    ncln = np.zeros(len(clusters))
    for c, cc in enumerate(clusters):
        ncln[c] = np.sum(nseqincluster[ncl == cc])
    nseqincluster = ncln
    sort = np.argsort(-nseqincluster)
    clusters, nseqincluster = clusters[sort], nseqincluster[sort]
    return clusters, nseqincluster

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='motif_variant_location',
                    description='Checks if variants are located in seqlets and which clusters are affected')
    parser.add_argument('mainvarfile', type=str, 
                        help = 'File containing the main variant for each sequence, variant locations are based of alignment from both alleles, or when indels are inserted in allele B')
    parser.add_argument('clusterfile', type=str, 
                        help='File with cluster assignment of seqlets')
    parser.add_argument('motiflocfile', type=str, 
                        help = 'File with motif locations when alleles are aligned and indels are inserted')
    parser.add_argument('validseqfile', type=str,
                        help = 'File that contains list of sequences with valid model predictions')
    parser.add_argument('--TFenrichment', type=str, default = None,
                        help = 'tomtom file with assigned TFs from database')
    parser.add_argument('--outname', type=str, default = None,
                        help = 'Define outname')
    parser.add_argument('--savemotiflist', action='store_true')

    args = parser.parse_args()
    
    mainvar = np.genfromtxt(args.mainvarfile, dtype = str) # file with main variant locations
    motifclusters = np.genfromtxt(args.clusterfile, dtype = str) # clusters of motifs
    motifalign = np.genfromtxt(args.motiflocfile, dtype = str) # location of motif in alignments
    seqnames = np.genfromtxt(args.validseqfile, dtype = str) # list of predictable sequences
    
    outname = args.outname
    if args.outname is None:
        outname = os.path.splitext(args.mainvarfile)[0]

    if not np.array_equal(motifclusters[:,0], motifalign[:,0]):
        print('Warning: motifclusters and motiflocations dont contain same motifs')
        sys.exit()
    # Generate sequence ids for motifs in motifclusters and motifalign
    motifseq = np.array(['seq_idx_'+m.split('_')[2] for m in motifclusters[:,0]])

    # Reduce mainvar, motifclusters, and motifalign to sequences in seqnames
    motmask = np.argsort(motifseq)[np.isin(np.sort(motifseq), seqnames)]
    motifalign, motifclusters, motifseq = motifalign[motmask], motifclusters[motmask], motifseq[motmask]
    # Sort main also alphabetically 
    varmask = np.argsort(mainvar[:,0])[np.isin(np.sort(mainvar[:,0]), seqnames)]
    mainvar = mainvar[varmask]

    # Extract motif location and assign each location the defined cluser id
    motifloc, motifclust = assign_motifcluster_to_location(motifclusters, motifalign)

    # Make variant locations comparable to motiflocs
    varloc = variant_locations(mainvar)

    # Determine cluster id of motif that every variant falls into
    varclust = assign_cluster_to_variant(mainvar, varloc, motifloc, motifclust)
    
    N=len(varclust)
    print(f'{N} variants investigated')
    percoutside = round(np.sum(varclust == 'NAN')*100/len(varclust),2)
    print(f'{percoutside} % outside any motif')
    meanNinside = round(np.sum(varclust != 'NAN')/len(np.unique(varclust)),1)
    print(f'Mean number of sequences in a motif cluster {meanNinside}')
    
    

    # Return number of sequenes that are affecte by a cluster
    cl, cln = np.unique(varclust[varclust != 'NAN'], return_counts = True)
    sort = np.argsort(-cln)
    cl, cln = cl[sort], (cln[sort]/N)*100

    # only put cluster names into the nameset if they have more than one sequence that they affect
    if args.savemotiflist:
        np.savetxt(os.path.splitext(args.mainvarfile)[0]+'_clusterset.txt', np.array([cl, cln]).T, fmt = '%s')
        print(f'Saved affected clusters in {os.path.splitext(args.mainvarfile)[0]}_clusterset.txt')

    if args.TFenrichment is not None:
        outname += '_TFset'
        tfcl, tfnames, pvalues, qvalues = readtomtom(args.TFenrichment)
        pvalues = -np.log10(pvalues)
        tfmask = pvalues > -np.log10(0.05)
        tfcl, tfnames, pvalues = tfcl[tfmask], tfnames[tfmask], pvalues[tfmask]
        # Assign TF names to clusters based on the most prominent TF in all of them
        cl, cln = assign_tfnames_to_clusters_frequencies(tfnames, pvalues, tfcl, cl, cln, add = '-like')
        


    # Reduce all cluster with less than 2 entries to Others
    min_viz_perc = 2
    mask = cln > min_viz_perc
    cl = np.append(cl[mask], [f'Other<{min_viz_perc}%'])
    cln = np.append(cln[mask], np.sum(cln[~mask]))
    cl = np.append(cl, ['Outside']) # append outside for -1, or for variants outside a motif
    cln = np.append(cln, np.sum(varclust == 'NAN')*100/N)

    fig = piechart(cln, cl, cmap = 'BuPu', cmap_range=[0.6,0.5], explode_size = 0.1, explode_indices = -1, labels_on_side = True, explode_color = 'firebrick')
    
    fig.savefig(outname+'_clusterpie.jpg', dpi = 300, bbox_inches = 'tight')
    print(outname+'_clusterpie.jpg')
    plt.show()



