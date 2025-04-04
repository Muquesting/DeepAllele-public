import numpy as np
import sys, os
from functools import reduce
import matplotlib.pyplot as plt

from DeepAllele.io import readtomtom
from DeepAllele.plotlib import piechart
import argparse

if __name__ == '__main__':

    affcluster = np.genfromtxt(sys.argv[1], dtype = str)[:, 2:]
    N_vars = len(affcluster)
    affcluster = affcluster[affcluster[:, 0] != 'NAN']

    motifeffects = np.genfromtxt(sys.argv[2], dtype = str)

    uclust = np.unique(affcluster[:,1])
    perc = np.zeros((len(uclust), 2))
    for u, uc in enumerate(uclust):
        mask = affcluster[:, 1] == uc
        maskseq = affcluster[mask, 0]
        # print(maskseq)
        maskeffect = motifeffects[np.isin(motifeffects[:, 0], maskseq), 1].astype(float)
        maskeffect = np.sign(maskeffect)
        # print(uc, maskeffect)
        perc[u,0] = np.sum(maskeffect>0)
        perc[u,1] = np.sum(maskeffect<0)

    perc = np.round(perc/N_vars*100,2)

    np.savetxt(os.path.splitext(sys.argv[1])[0]+'_clustposperc.txt', np.array([uclust, perc[:,0]]).T.astype(str), fmt = '%s')
    np.savetxt(os.path.splitext(sys.argv[1])[0]+'_clustnegperc.txt', np.array([uclust, perc[:,1]]).T.astype(str), fmt = '%s')




