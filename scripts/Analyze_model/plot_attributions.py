import numpy as np
import sys, os
import matplotlib.pyplot as plt
import logomaker as lm
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import pearsonr

from DeepAllele.plotlib import plot_attribution


def check_attributions(att):
    pearson=pearsonr(att[...,0].flatten(), att[...,1].flatten())[0]
    if pearson < 0:
        print(Warning('It seems like your attributions in allele A and B are anticorrelated, which indicates that you should use --ratioattributions'))
    


if __name__ == '__main__':
    att = np.load(sys.argv[1])
    seq = np.load(sys.argv[2])
    outname = os.path.splitext(sys.argv[1])[0]
    mlocs = None

    # If you have a file that defines motif locations, this will put frames around the bases
    if '--motif_location' in sys.argv:
        mlocfile = np.genfromtxt(sys.argv[sys.argv.index('--motif_location')+1], dtype = str)[:,0]
        mlocname = sys.argv[sys.argv.index('--motif_location')+2]
        mlocs = np.array([m.rsplit('_',3) for m in mlocfile]) # keeps sequence name, locations separated by '-', allele, and last number which indicates if motif was found in both sequences or only one
        keep = [i for i, name in enumerate(mlocs[:,0]) if name in mlocname]
        mlocs = mlocs[keep].astype(object)
        for m, ml in enumerate(mlocs):
            mlocs[m][1] = np.array(ml[1].split('-'), dtype = int)
        mlocs[:,[2,3]] = mlocs[:,[2,3]].astype(int)

    # Modify the attribution values if not yet processed
    if '--centerattributions' in sys.argv:
        att -= (np.sum(att, axis = -2)/4)[...,None,:]
    elif '--decenterattributions' in sys.argv:
        att -= seq * att
    elif '--meaneffectattributions' in sys.argv:
        att -= (np.sum((seq == 0)*att, axis = -2)/3)[...,None,:]

    seq_based = True
    if '--showall_attributions' in sys.argv:
        seq_based = False
        outname += '.all'

    add_perbase = False
    if '--add_perbase' in sys.argv:
        add_perbase = True

    # Attributions from the model ratio are mirror images of one another. 
    # For analysis, we need to switcht the sign
    if '--ratioattributions' in sys.argv:
        att[..., 1] = -att[...,1]
    check_attributions(att)

    if len(np.shape(att)) == 4:
        att = att.squeeze(0)
    
    fmt = 'jpg'
    if '--format' in sys.argv:
        fmt = sys.argv[sys.argv.index('--format')+1]
    
    dpi = 200
    if '--dpi' in sys.argv:
        dpi = int(sys.argv[sys.argv.index('--dpi')+1])
    
    fig = plot_attribution(seq, att, motifs = mlocs, seq_based = seq_based, add_perbase = add_perbase)
    fig.savefig(outname+'.'+fmt, dpi = dpi, bbox_inches = 'tight')
    print(outname+'.'+fmt)


