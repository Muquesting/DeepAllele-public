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


    if '--addCluster' in sys.argv:
        ucluster = np.array(['Cluster_'+str(u) for u in ucluster])


    if '--split_effect_direction' in sys.argv:
        dirfile = sys.argv[sys.argv.index('--split_effect_direction')+1]
        direction = np.genfromtxt(dirfile, dtype = str)
        if np.array_equal(clustfile[:,0], direction[:,0]):
            direction = np.sign(direction[:,1].astype(float))
            nseq = np.zeros((len(ucluster),2), dtype = float)
            for c, uc in enumerate(ucluster):
                for d, ed in enumerate([-1,1]):
                    nseq[c,d] = len(np.unique(seqs[(clusternames == uc)&(direction == ed)]))
            # appearance in percent of sequences with a motif
            nseq = np.around(nseq/len(useqs) *100,2)
            np.savetxt(os.path.splitext(sys.argv[1])[0]+'_motifpercposinseq.txt', np.array([ucluster.astype(str), nseq[:,1]]).T, fmt = '%s')
            np.savetxt(os.path.splitext(sys.argv[1])[0]+'_motifpercneginseq.txt', np.array([ucluster.astype(str), nseq[:,0]]).T, fmt = '%s')
        else:
            print('Effect direction file and cluster file do not contain the same sequences, or not in the same order')
            sys.exit()

    else:
        nseq = np.zeros(len(ucluster), dtype = float)
        for c, uc in enumerate(ucluster):
            nseq[c] = len(np.unique(seqs[clusternames == uc]))

        # appearance in percent of sequences with a motif
        nseq = np.around(nseq/len(useqs) *100,2)

        print(f'Created {os.path.splitext(sys.argv[1])[0]}_motifpercinseq.txt')
        np.savetxt(os.path.splitext(sys.argv[1])[0]+'_motifpercinseq.txt', np.array([ucluster.astype(str), nseq]).T, fmt = '%s')
    
    


