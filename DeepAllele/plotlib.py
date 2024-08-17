import numpy as np
import sys, os
import matplotlib.pyplot as plt
import logomaker as lm
import pandas as pd
from Bio.Align import PairwiseAligner

def align_seq(s1, s2):
    '''
    Aligns two sequences and returns indexes to match bases. To assign values
    of the original_array to positions in the alignment use: 
    aligned_seq1[l1] = original_seq1[i1]
    Returns
    -------
    i1 : 
        indexes of bases in sequences1 one in original sequence without gaps
    l1 : 
        indexes of bases in sequence1 in alignment (gaps will be left out)
    i2 : 
        indexes of bases in sequence2 one in original sequence without gaps
    l2 : 
        indexes of bases in sequence2 in alignment (gaps will be left out)
    '''
    aligner = PairwiseAligner()
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    aligner.match_score = 1
    aligner.mismatch_score = 0
    alignments = aligner.align(s1,s2)
    best = alignments[0]
    i1, i2 = [],[]
    l1, l2 = [],[]
    j = 0
    k = 0
    for i in range(len(best[0])):
        if best[0][i] != '-':
            i1.append(j)
            l1.append(i)
            j += 1
        if best[1][i] != '-':
            i2.append(k)
            l2.append(i)
            k += 1
    return np.array(i1),np.array(i2),np.array(l1),np.array(l2)

def seq_from_onehot(x, nts = 'ACGT'):
    '''
    Transforms one-hot encoding into sequence. 
    Expects 'ACGT' order
    '''
    order = np.where(x == 1)[1]
    seq = ''.join(np.array(list(nts))[order])
    return seq

def align_onehot(seq):
    s1 = seq_from_onehot(seq[...,0])
    s2 = seq_from_onehot(seq[...,1])
    i1, i2, l1, l2 = align_seq(s1,s2)
    return i1, i2, l1, l2

def add_frames(att, locations, colors, ax):
    '''
    Adds frames around locations in colors
    
    Parameters
    ---------
    locations :
        list of tuples
    '''
    att = np.array(att)
    for l, loc in enumerate(locations):
        mina, maxa = np.amin(np.sum(np.ma.masked_greater(att[loc[0]:loc[1]+1],0),axis = 1)), np.amax(np.sum(np.ma.masked_less(att[loc[0]:loc[1]+1],0),axis = 1))
        x = [loc[0]-0.5, loc[1]+0.5]
        ax.plot(x, [mina, mina], c = cmap[colors[l]])
        ax.plot(x, [maxa, maxa], c = cmap[colors[l]])
        ax.plot([x[0], x[0]] , [mina, maxa], c = cmap[colors[l]])
        ax.plot([x[1], x[1]] , [mina, maxa], c = cmap[colors[l]])


def _plot_attribution(att, ax, nts = list('ACGT'), labelbottom=False, bottom_axis = False, ylabel = None, ylim = None, xticks = None, xticklabels = None):
    '''
    Plot single atribubtion from np.array
    '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(bottom_axis)
    ax.tick_params(bottom = labelbottom, labelbottom = labelbottom)
    att = pd.DataFrame(att, columns = nts)
    lm.Logo(att, ax = ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

def plot_attribution(seq, att, add_perbase = False, motifs = None, seq_based = True, nts = list('ACGT'), plot_effect_difference = True, plot_difference = True):
    '''
    Plots attributions of two sequences and illustrates the variant effect sizes
    
    Parameters
    ----------
    seq : 
        one-hot encoded sequence of shape = (length, channels=4, n_seqs=2)
    att : 
        sequence attributions encoded sequence of shape = (length, channels=4, n_seqs=2)
    add_perbase :
        if True add heatmap with per base attributions for each sequence and difference
    motifs :
        list of lists (sequence_name, tuple of locations, allele, and color)
    seq_based:
        if True, only the attributions at a present base are shown. Otherwise, 
        also attributions from bases that are not present in the sequence. This
        can be helpful to spot motifs that would be preferred by the model.
    
    TODO: 
        Automatically set xlim
        Use matplotlib axgrid to make space for heatmaps smaller
    '''
    
    
    if add_perbase:
        pbatt = att
        pbvlim = np.amax(np.absolute(pbatt))
    
    if seq_based:
        att = seq * att

    # Align one-hot encodings to visualize on top of each other
    si1, si2, ti1, ti2 = align_onehot(seq)
    maxlen = np.amax(np.concatenate([ti1,ti2]))+1
    
    attA = np.zeros((maxlen,4))
    attB = np.zeros((maxlen,4))
    fracAB = np.zeros((maxlen,4))
    seqAB = np.zeros((maxlen,4))
    
    attA[ti1] = att[si1,:,0]
    attB[ti2] = att[si2,:,1]
    fracAB = attA - attB
    
    seqAB[ti1] = seq[si1,:,0]
    seqAB[ti2] -= seq[si2,:,1]

    if add_perbase:
        pbattA = np.zeros((maxlen,4))
        pbattB = np.zeros((maxlen,4))
        pbattA[ti1] = pbatt[si1,:,0]
        pbattB[ti2] = pbatt[si2,:,1]
        

    mina,minb = np.array(np.sum(np.ma.masked_greater(attA,0), axis = -1)), np.array(np.sum(np.ma.masked_greater(attB,0), axis = -1))
    
    maxa,maxb = np.array(np.sum(np.ma.masked_less(attA,0), axis = -1)), np.array(np.sum(np.ma.masked_less(attB,0), axis = -1))
    
    attlim = [min(np.amin(np.append(mina,minb)),0),np.amax(np.append(maxa,maxb))]

    fig = plt.figure(figsize = (0.04*maxlen, 3+3*int(add_perbase)))
    
    ax0 =  fig.add_subplot(4+3*int(add_perbase), 1, 1)
    _plot_attribution(attA, ax0, ylabel = 'B6', ylim = attlim)
    

    if motifs is not None:
        mask = motifs[:,-2] == 0 # Only consider motifs in sequence 0 here
        colors = motifs[mask,-1] # get the colors of these
        locations = [ti1[l] for l in motifs[mask,1]] # get the locs of these
        add_frames(attA, locations, colors, ax0)
    
    if add_perbase: # add another subplot with a heatmap for per base effecst
        ax0b =  fig.add_subplot(4+3*int(add_perbase), 1, 2)
        ax0b.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False)
        ax0b.imshow(pbattA.T, vmin = -pbvlim, vmax = pbvlim, cmap = 'coolwarm_r')

    ax1 =fig.add_subplot(4+3*int(add_perbase), 1, 2+1*int(add_perbase))
    _plot_attribution(attB, ax1, ylabel = 'CAST', ylim = attlim)


    if motifs is not None:
        mask = motifs[:,-2] == 1
        colors = motifs[mask,-1]
        locations = [ti2[l] for l in motifs[mask,1]]
        add_frames(attB, locations, colors, ax1)

    if add_perbase:
        ax1b =  fig.add_subplot(4+3*int(add_perbase), 1, 4)
        ax1b.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False)
        ax1b.imshow(pbattB.T, vmin = -pbvlim, vmax = pbvlim, cmap = 'coolwarm_r')

    if plot_difference:
        # Plot the difference between the two attributions
        axf =  fig.add_subplot(4+3*int(add_perbase),1,3+2*int(add_perbase))
        
        if plot_effect_difference:
            sumeffect = np.sum(fracAB*np.absolute(seqAB), axis = 1)
            fraclim = [min(np.amin(sumeffect),attlim[0]), max(np.amax(sumeffect),attlim[1])]
            _plot_attribution(fracAB*np.absolute(seqAB),axf, labelbottom = True, bottom_axis = True, ylabel = 'Effect\nsize', ylim = fraclim, xticks = [0,125,250], xticklabels=['-125', '0', '125'])
        else:
            sumeffect = np.sum(fracAB, axis = 1)
            fraclim = [min(np.amin(sumeffect),attlim[0]), max(np.amax(sumeffect),attlim[1])]
            _plot_attribution(fracAB,axf, labelbottom = True, bottom_axis = True, ylabel = 'Effect\nsize', ylim = fraclim, xticks = [0,125,250], xticklabels=['-125', '0', '125'])

        if add_perbase:
            axfb =  fig.add_subplot(4+3*int(add_perbase), 1, 6)
            axfb.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False)
            axfb.imshow(pbattA.T-pbattB.T, vmin = -pbvlim, vmax = pbvlim, cmap = 'coolwarm_r')
    
    return fig
