import numpy as np
import sys, os
from scipy.stats import pearsonr
import glob
import matplotlib.pyplot as plt

def histogram(x, nbins = None, xlabel = None, logy = False):
    fig = plt.figure(figsize = (3.5,3.5))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if nbins is None:
        nbins = np.linspace(-0.5, int(np.amax(x))+0.5, int(np.amax(x))+2)

    ax.hist(x, bins = nbins, color = 'Indigo', alpha = 0.6, histtype = 'bar', lw = 1 )
    ax.hist(x, bins = nbins, color = 'Indigo', alpha = 1., histtype = 'step' )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale('log')
    return fig


if __name__ == '__main__':
    
    filea = np.load(sys.argv[1]) # sequence attributions .npy 
    fileb = np.load(sys.argv[2]) # second sequence attributions .npy
    
    namesa = np.load(sys.argv[3]) # identifiers for first sequences
    namesb = np.load(sys.argv[4]) # identifiers for second sequences
    
    # sort so that sequences match
    asort = np.argsort(namesa)[np.isin(np.sort(namesa), namesb)]
    filea, namesa = filea[asort], namesa[asort]
    bsort = np.argsort(namesb)[np.isin(np.sort(namesb), namesa)]
    fileb, names = fileb[bsort], namesb[bsort]

    # xlabel for histogram
    xlabel = sys.argv[5]
    outname = sys.argv[6] # File name to save correlation and histogram in

    # sequences can be provided to make sure that attributions in padded areas
    # or gaps are removed before computing correlations, i.e. 'seqtype = input'
    # or to only cosider attributions at the 'reference' base.
    seqtype = None
    if '--seqs' in sys.argv:
        seqs = sys.argv[sys.argv.index('--seqs')+1] # sequence .npy
        seqlabels = sys.argv[sys.argv.index('--seqs')+2] # sequence idenifyer .npy
        seqtype = sys.argv[sys.argv.index('--seqs')+3] # input, or reference
        if seqtype not in ['reference', 'input']:
            raise ValueError(f'{seqtype} not a valid seqtype')
        seqs = np.load(seqs)
        seqlabels = np.load(seqlabels)
        sort = np.argsort(seqlabels)[np.isin(np.sort(seqlabels), names)]
        seqs, seqlabels = seqs[sort], seqlabels[sort]
        if not np.array_equal(seqlabels, names):
            print('sequences dont match data')
            sys.exit()
        outname = os.path.splitext(outname)[0]+'.'+seqtype+os.path.splitext(outname)[1]
    
    corr = []
    for n, name in enumerate(names):
        ata = filea[n]
        atb = fileb[n]
        if seqtype is not None:
            a, b = [], []
            x, y = [[],[]], [[],[]]
            for j in range(2):
                # only consider values at reference base positions in input
                mask = np.where(seqs[n,...,j] == 1)
                # only consider values at positions with a base in input
                if seqtype == 'input':
                    mask = np.unique(mask[0])
                a.append(ata[...,j][mask].flatten())
                b.append(atb[...,j][mask].flatten())
            ata = np.concatenate(a)
            atb = np.concatenate(b)
        
        corr.append(pearsonr(ata.flatten(), atb.flatten())[0])

    print('Median correlation', np.median(corr))
    np.savetxt(outname, np.array([names, corr]).T, fmt = '%s')
    fig = histogram(corr, nbins= np.linspace(-1,1,41), xlabel = xlabel)
    fig.savefig(os.path.splitext(outname)[0]+'hist.jpg', dpi = 250, bbox_inches = 'tight')
    print('Histogram saved as', os.path.splitext(outname)[0]+'hist.jpg')
    plt.show()



