# cluster_pwms.py
import numpy as np
import sys, os
from scipy.stats import pearsonr 
from sklearn.cluster import AgglomerativeClustering

from DeepAllele.io import readin_motif_files, write_meme_file
from DeepAllele.motif_analysis import torch_compute_similarity_motifs, combine_pwms, assign_leftout_to_cluster
import argparse

def complement_clustering(clusters, pwmnames, pwm_set, logs, pwmnames_left, pwm_left, randmix):
    '''
    Wrapper to save space in main. 
    1) Computes distancce matrix between triplets from assigned clusters and
    left out data points.
    2) Assigns left out data points based on distance cut off and linkage 
    function to the already existing clusters
    3) If data points could not be assigned, performs another clustering of
    those independently
    4) Combines cluster sets and returns names, clusters and pwms back to
    original order
    
    '''
    triplets, tripclusters = _determine_triplets(clusters, logs) # three members of the cluster to which left out data points will be measured
    corr_left, logs_left, ofs_left, revcomp_matrix_left = torch_compute_similarity_motifs(pwm_set[triplets], pwm_left, fill_logp_self = 1000, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = True)
    
    if args.clusteronlogp:
        checkmat = 10**-logs_left.reshape(3,-1,logs_left.shape[-1])
    else:
        checkmat = corr_left.reshape(3,-1,corr_left.shape[-1])
    
    clusters_left = assign_leftout_to_cluster(tripclusters, checkmat, args.linkage, args.distance_threshold)
    
    print(len(clusters_left)-len(np.where(clusters_left == -1)[0]), 'added to assigned clusters')
    
    if len(np.where(clusters_left == -1)[0]) > 1 and len(np.where(clusters_left == -1)[0]) <= args.approximate_cluster_on:
        print(f'Reclustering of {len(np.where(clusters_left == -1)[0])} clusters')
        corr_left, logs_left, ofs_left, revcomp_matrix_left = torch_compute_similarity_motifs(pwm_left[clusters_left == -1], pwm_left[clusters_left == -1], fill_logp_self = 1000, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = True)

        if args.clusteronlogp:
            clustering = AgglomerativeClustering(n_clusters = None,metric = 'precomputed', linkage = args.linkage, distance_threshold = args.distance_threshold).fit(10**-logs_left)
        else:
            clustering = AgglomerativeClustering(n_clusters = None, metric = 'precomputed', linkage = args.linkage, distance_threshold = args.distance_threshold).fit(corr_left)
        clusters_left[clusters_left == -1] = np.amax(clusters) + clustering.labels_

    resort = np.argsort(randmix)
    clusters = np.append(clusters, clusters_left)[resort]
    pwmnames = np.append(pwmnames, pwmnames_left)[resort]
    pwm_set = np.array(list(pwm_set)+list(pwm_left), dtype = object)[resort]

    return clusters, pwmnames, pwm_set

def _determine_triplets(clusters, similarity):
    '''
    Determine three data points in a cluster that are the furthest apart from
    each other. Use these three data points to determine if another data point
    should be assigned to that cluster.
    Unfortunately, the three loops take a while to run.
    '''
    uclusters = np.unique(clusters)
    bests = []
    #print(similarity)
    for u, uc in enumerate(uclusters):
        mask = np.where(clusters == uc)[0]
        csim = similarity[mask][:,mask]
        bestdist = 3*csim[0,0]
        best = [0,0,0]
        for i in range(len(mask)):
            for j in range(i, len(mask)):
                for h in range(j, len(mask)):
                    dist = csim[i,j] + csim[i,h] + csim[j,h]
                    if dist < bestdist:
                        best = i,j,h
                        bestdist = np.copy(dist)
        #print(best, bestdist)
        bests.append(best)
        
    return np.concatenate(bests), np.repeat(uclusters,3)

def combine_pwms_separately(pwm_set, clusters):
    '''
    Compute offsets and distances for each cluster separately to avoid memory
    issues
    '''
    clusterpwms = [] 
    for c, uc in enumerate(np.unique(clusters)):
        if uc >=0:
            mask = clusters == uc
            corr_left, logs_left, ofs_left, revcomp_matrix_left = torch_compute_similarity_motifs(pwm_set[mask], pwm_set[mask], fill_logp_self = 1000, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = True)
            clusterpwms.append(combine_pwms(np.array(pwm_set, dtype = object)[mask], clusters[mask], logs_left, ofs_left, revcomp_matrix_left)[0])
    return clusterpwms








if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='cluster_seqlets',
                    description='Aligns and computes Pearson correlation distance between all seqlets, then uses agglomerative clustering to determine groups, and combines seqlets into common motif')
    parser.add_argument('pwmfile', type=str, 
                        help='This can be a meme, a txt, or npz file with pwms and pwmnames, OR the .npz file with the stats from previous clustering')
    parser.add_argument('linkage', type=str, 
                        help='Linkage type for agglomerative clustering, or file with clusters from previous computation to generate combined PWMs')
    parser.add_argument('--distance_threshold', type=float, 
                        help='Clustering distance cut-off to form groups', default = None)
    parser.add_argument('--n_clusters', type=float, 
                        help='If given, clustering motifs into N clusters instead of using distance threshold', default = None)
    parser.add_argument('--outname', type=str, default = None)
    parser.add_argument('--infocont', action='store_true')
    parser.add_argument('--clusteronlogp', action='store_true', 
                        help = 'Uses the p-values for clustering instead of the correlation.')
    parser.add_argument('--clusternames', action='store_true', 
                        help = 'If True combines original names with ; to long name for meme file. By default returns identifier as name')
    parser.add_argument('--save_stats', action='store_true')
    parser.add_argument('--reverse_complement', action='store_true', 
                        help = 'Determines if reverse complement will be compared as well')
    parser.add_argument('--approximate_cluster_on', default = None, type = int, 
                        help='Define number of random motifs on which clustering will be performed, while rest will be assiged to best matching centroid of these clusters. Should be used if memory is too small for large distance matrix')
    parser.add_argument('--min_overlap', type = int, default = 4)
    
    args = parser.parse_args()
    
    if args.outname is None:
        outname = os.path.splitext(args.pwmfile)[0]+'ms'+str(args.min_overlap)
        if args.infocont:
            outname+='ic'
    else:
        outname = args.outname
    
    infmt= os.path.splitext(args.pwmfile)[1]
    
    # Determine if file contains only pwms, or also statistics that were saved 
    # in previous run
    isstatsfile = False
    if infmt == '.npz':
        pf = np.load(args.pwmfile, allow_pickle = True)
        pffiles = pf.files()
        if ('correlation' in pfiles) and ('offsets' in pf):
            isstatsfile = True
    ofs = None
    if isstatsfile:
        pwmnames =pf['pwmnames']
        correlation = pf['correlation']
        logs = pf['logpvalues']
        ofs = pf['offsets']
        pwm_set = pf['pwms']
        revcomp_matrix = pf['revcomp_matrix']
        if args.outname is None:
            outname = outname.split('_stats')[0]
       
    else:
        pwm_set, pwmnames, nts = readin_motif_files(args.pwmfile)
        
        if args.approximate_cluster_on is not None: # Approximation of clusters
            # cluster only subset and assign left over seqlets to assigned 
            # clusters based on the similarity to the three most distant 
            # points in the cluster. Reduced memory by only computing 
            # similarity to three points per cluster.
            np.random.seed(0)
            if args.outname is None:
                outname += 'aprx'+str(args.approximate_cluster_on)
            rand_mix = np.random.permutation(len(pwm_set))
            rand_set,rand_left = rand_mix[:args.approximate_cluster_on], rand_mix[args.approximate_cluster_on:]
            
            pwm_left = pwm_set[rand_left]
            pwmnames_left = pwmnames[rand_left]
            pwm_set = pwm_set[rand_set]
            pwmnames = pwmnames[rand_set]
        
        
        # Align and compute correlatoin between seqlets using torch conv1d.
        correlation, logs, ofs, revcomp_matrix= torch_compute_similarity_motifs(pwm_set, pwm_set, fill_logp_self = 1000, min_sim = args.min_overlap, infocont = args.infocont, reverse_complement = args.reverse_complement, exact = True)
        
        # Save computed statistics for later
        if args.save_stats and args.approximate_cluster_on is None:
            np.savez_compressed(outname+'_stats.npz', pwmnames = pwmnames, correlation = correlation, logpvalues = logs, offsets = ofs, pwms = pwm_set, revcomp_matrix = revcomp_matrix)
    
    # Linkage defines the linkage function for clustering, or can be a file
    # with cluster assigments. In this case, distance_treshold is ignored and
    # only combine_pwms is executed. 
    if os.path.isfile(args.linkage):
        if args.outname is None:
            outname = os.path.splitext(args.linkage)[0]
        clusters = np.genfromtxt(args.linkage, dtype = str)
        if not np.array_equal(pwmnames, clusters[:,0]):
            sort = []
            for p, pwn in enumerate(pwmnames):
                sort.append(list(clusters[:,0]).index(pwn))
            clusters = clusters[sort]
        clusters = clusters[:,1].astype(int)
        if ofs is None:
            clusterpwms = combine_pwms_separately(pwm_set, clusters)
        else:
            clusterpwms = combine_pwms(pwm_set, clusters, logs, ofs, revcomp_matrix)
    else:
        if args.outname is None:
            outname += '_cld'+args.linkage
            if args.n_clusters:
                outname += 'N'+str(args.n_clusters)
            else:
                outname += str(args.distance_threshold)
            
        if args.clusteronlogp:
            outname += 'pv'
            #logs = 10**-logs
            clustering = AgglomerativeClustering(n_clusters = args.n_clusters, metric = 'precomputed', linkage = args.linkage, distance_threshold = args.distance_threshold).fit(10**-logs)
        else:
            clustering = AgglomerativeClustering(n_clusters = args.n_clusters, metric = 'precomputed', linkage = args.linkage, distance_threshold = args.distance_threshold).fit(correlation)
        
        clusters = clustering.labels_
        
        if args.approximate_cluster_on is not None:
            
            clusters, pwmnames, pwm_set = complement_clustering(clusters, pwmnames, pwm_set, logs, pwmnames_left, pwm_left, rand_mix)
            
            clusterpwms = combine_pwms_separately(pwm_set, clusters)
        
        else:
            clusterpwms = combine_pwms(pwm_set, clusters, logs, ofs, revcomp_matrix)
        
        np.savetxt(outname + '.txt', np.array([pwmnames,clusters]).T, fmt = '%s')
        print(outname + '.txt')
        
    print(len(pwm_set), 'form', len(np.unique(clusters)), 'clusters')
        
    if args.clusternames:
        clusternames = [str(i) for i in np.unique(clusters) if i >= 0]
    else:
        clusternames = [';'.join(np.array(pwmnames)[clusters == i]) for i in np.unique(clusters)]
     
    write_meme_file(clusterpwms, clusternames, 'ACGT', outname +'pfms.meme', )
    
    uclust, nclust = np.unique(clusters, return_counts = True)
    uclust, nclust = uclust[uclust > -1], nclust[uclust > -1]
    nhist, yhist = np.unique(nclust, return_counts = True)
    for n, nh in enumerate(nhist):
        print(nh, yhist[n])


    
    
    
    
    
    
    
    
    
