import numpy as np
import sys, os

clusters = np.genfromtxt(sys.argv[1], dtype = str)
ncut = int(sys.argv[2])

uclust, nclust = np.unique(clusters[:,1], return_counts = True)

unclust, nnclust = np.unique(nclust, return_counts = True)
sort = np.argsort(unclust)
unclust, nnclust = unclust[sort], nnclust[sort]
for u, unc in enumerate(unclust):
    print(unc, nnclust[u])

uclust =uclust[nclust >= ncut]

print(len(uclust))

uclust=['Cluster_'+str(uc) for uc in uclust]

np.savetxt(os.path.splitext(sys.argv[1])[0]+'_Ngte'+str(ncut)+'list.txt', uclust, fmt = '%s')


