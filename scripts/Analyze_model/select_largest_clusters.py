import numpy as np
import sys, os

'''
Writes list with cluster ids of clusters that are larger than ncut
'''

if __name__ == '__main__':
        
    clusters = np.genfromtxt(sys.argv[1], dtype = str)
    ncut = int(sys.argv[2])

    uclust, nclust = np.unique(clusters[:,1], return_counts = True)

    unclust, nnclust = np.unique(nclust, return_counts = True)
    sort = np.argsort(unclust)
    unclust, nnclust = unclust[sort], nnclust[sort]
    if '--verbose' in sys.argv:
        for u, unc in enumerate(unclust):
            print(unc, nnclust[u])

    uclust =uclust[nclust >= ncut]

    print(f'Clusters larger than {ncut}: {len(uclust)}')

    if '--addCluster' in sys.argv:
       uclust=['Cluster_'+str(uc) for uc in uclust]

    np.savetxt(os.path.splitext(sys.argv[1])[0]+'_Ngte'+str(ncut)+'list.txt', uclust, fmt = '%s')


