import numpy as np
import sys, os

'''
Returns file with percentage of appearances in all sequences
'''

if __name__ == '__main__':
    
    clustfile=np.genfromtxt(sys.argv[1], dtype = str)
    seqs = np.array([n.split('_')[2] for n in clustfile[:,0]])
    useqs = np.unique(seqs)
    clusternames = clustfile[:,1].astype(int)
    ucluster = np.unique(clusternames)

    nseq = np.zeros(len(ucluster), dtype = float)
    for c, uc in enumerate(ucluster):
        nseq[c] = len(np.unique(seqs[clusternames == uc]))

    # appearance in percent of sequences with a motif
    nseq = np.around(nseq/len(useqs) *100,2)
    
    if '--addCluster' in sys.argv:
        ucluster = np.array(['Cluster_'+str(u) for u in ucluster])
    
    np.savetxt(os.path.splitext(sys.argv[1])[0]+'_motifpercinseq.txt', np.array([ucluster.astype(str), nseq]).T, fmt = '%s')
    
    


