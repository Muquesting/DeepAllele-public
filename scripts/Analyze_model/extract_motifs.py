
import numpy as np
import sys, os
from DeepAllele.motif_analysis import find_motifs, align_onehot
from DeepAllele.io import write_meme_file
import argparse
from scipy.stats import pearsonr

def check_attributions(att):
    pearson=pearsonr(att[...,0].flatten(), att[...,1].flatten())[0]
    if pearson < 0:
        print(Warning(f'It seems like your attributions in allele A and B are anticorrelated, which indicates that you should use --ratioattributions {pearson}'))
    
def check_decimal(r):
    pot = 10
    i = -1
    while True:
        if r> pot**i:
            return i +1
        i -= 1




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='extract_motifs',
                    description='extracts motifs from attributions based on their height and subsequent size')
    parser.add_argument('seqlabels', type=str)
    parser.add_argument('atts', type=str)
    parser.add_argument('seqs', type=str)
    parser.add_argument('--cut', type=float, default = 1.96, help='Z-score cut off for calling significance of attributions')
    parser.add_argument('--maxgap', type=int, default = 1, help = 'Maximum number of gap bases')
    parser.add_argument('--minsig', type=int, default = 4, help='Minimum number of subsequent significant bases to call something a motif.')
    
    parser.add_argument('--atreference', action='store_false', help='If True, uses attribution at reference to extract motifs, otherwise, uses the max attribution at a given position')
    parser.add_argument('--normed', action='store_false', help='If True attributions will be normalized before ')
    parser.add_argument('--ratioattributions', action='store_false', help='If True attributions are coming from the ratio head and need to be adjusted for sign')
    parser.add_argument('--outname', type=str, default = None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    
    outname = args.outname
    if outname is None:
        outname = os.path.splitext(args.atts)[0]+'_seqlets.cut'+str(args.cut)+'maxg'+str(args.maxgap)+'minsig'+str(args.minsig)

    stats = np.load(args.seqlabels)
    atts = np.load(args.atts)
    seqs = np.load(args.seqs)

    anames = [] # names of motifs
    amotifs = [] # motifs taken from z-scored ism
    ameans = [] # mean value of motifs
    amax = []
    astats = [] # statistics each input, how many motifs in each variant.
    otherloc = [] # location in the other sequence
    ameandiff = [] # difference of attributions between two sequences 
    amaxdiff = []
    
    z = 1
    if args.normed:
        z=np.sqrt(np.mean(atts**2)) # compute standard deviation to mean = 0
    ro = abs(check_decimal(z) -3) # estimate rounding precision for mean, max stats


    names = []
    for s, stat in enumerate(stats):
        name = 'seq_idx_'+str(s)+'_'+stat
        ism = atts[s]
        names.append(name)
        if args.ratioattributions:
            ism[...,-1] = -ism[...,-1]
        if args.verbose:
            print(name)
            check_attributions(ism)
        elif s%1000 == 0:
            print(f'{s}/{len(stats)}')
        seqonehot = seqs[s]
        lseqs = [len(np.where(seqonehot[:,:,j] == 1)[0]) for j in range(2)]
        
        seqo = align_onehot(seqonehot) # align sequences to match motif locations for 'common'
        align = seqo[-2:] # location of bases in aligned sequences
        seqot = seqo[:2] # translation of location of bases in first and second sequence. 
        # only consider positions that were covered by both sequences in the alignment
        seqo = []
        seqo.append(seqot[0][np.isin(align[0], align[1])])
        seqo.append(seqot[1][np.isin(align[1], align[0])])
        
        zsm = ism/z # z-score norm ism with std
        seqmotifs = []
        
        for j in range(2): # iterate over both seqences
            if args.atreference:
                refatts = np.sum(zsm[:lseqs[j],:,j] * seqonehot[:lseqs[j], :,j],axis =1)
            else:
                refatts = np.argmax(np.absolute(zsm[:lseqs[j],:,j]), axis = 1)
                refatts = zsm[np.arange(lseqs[j]),refatts,j]
            motifs = find_motifs(refatts, args.cut, args.maxgap, args.minsig) # find the motifs in the ISM
            seqmotifs.append(motifs)
            if args.verbose:
                print(len(motifs), 'motifs found for allele', j)
        
        # determine if the motifs are present in both sequences from the location of these bases in the other sequence
        common = [np.zeros(len(seqmotifs[j]), dtype = int) for j in range(2)]
        theotherloc = [[] for j in range(2)]
        for j in range(2):
            # translate index from seq1 to indeces/location in seq2
            if len(seqmotifs[j]) > 0 and len(seqmotifs[j-1]) > 0: # only do this if motifs in both sequences present
                matchloc = seqo[j][np.isin(seqo[j-1],np.concatenate(seqmotifs[j-1]))]
                for m, mot in enumerate(seqmotifs[j]):
                    common[j][m] = int(np.isin(matchloc, mot).any())
            if len(seqmotifs[j]) > 0:
                for m, mot in enumerate(seqmotifs[j]):
                    theotherloc[j].append(np.array(align[j])[mot])        
        
        # compute how many consistent between both 
        # compute how many lost in a, and b --> per-sequence file
        astats.append([np.sum(common[0]==1), np.sum(common[1]==1), np.sum(common[0] == 0), np.sum(common[1] == 0)])
        
        # compute mean activity for each motif --> Mean file
        for j in range(2):
            for i, smo in enumerate(seqmotifs[j]):
                mot = zsm[seqmotifs[j][i][0]:seqmotifs[j][i][-1]+1,:, j] # saved pwm is taken from zscored ism to align them
                motf = ism[seqmotifs[j][i][0]:seqmotifs[j][i][-1]+1,:, j] * seqonehot[seqmotifs[j][i][0]:seqmotifs[j][i][-1]+1,:, j]
                motmean = np.mean(motf)*4 # mean is computed from original isms
                motmax = motf[np.argmax(np.abs(motf))//motf.shape[1], np.argmax(np.abs(motf))%motf.shape[1]]
                # compute mean of the same bases in the other sequence to get delta mean
                altlocmot = seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]
                # some bases might be not present because of insertions
                if len(altlocmot) > 0:
                    altmot = ism[np.amin(seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]):np.amax(seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]) +1,:, j-1] * seqonehot[np.amin(seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]):np.amax(seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]) +1,:, j-1]
                    altmotmean=np.mean(altmot)*4
                    altmotmax=altmot[np.argmax(np.abs(altmot))//altmot.shape[1],np.argmax(np.abs(altmot))%altmot.shape[1]]
                else:
                    altmotmean = 0
                motmeandiff = altmotmean - motmean
                motmaxdiff = altmotmax-motmax
                # use sign(mean) * z-score attribution --> pwm file npz
                mot = mot * np.sign(motmean) # adjust signs to make comparable and only align motifs with positive correlation
                # also save file that contains information about where to find this motif in the other sequence
                otherloc.append(','.join(np.array(theotherloc[j][i]).astype(str)))
                amotifs.append(np.around(mot,2).T)
                # name by seq, start-end, and j, common or not --> pwm file
                anames.append(name+'_'+str(seqmotifs[j][i][0])+'-'+str(seqmotifs[j][i][-1])+'_'+str(j)+'_'+str(common[j][i]))
                #print(anames[-1])
                ameans.append(motmean)
                ameandiff.append(motmeandiff)
                amax.append(motmax)
                amaxdiff.append(motmaxdiff)

    np.savetxt(outname+'_otherloc.txt' ,np.array([anames, otherloc]).T, fmt = '%s')
    np.savetxt(outname+'_seqleteffects.txt' ,np.array([anames, np.around(ameans,ro), np.around(ameandiff,ro), np.around(amax,ro), np.around(amaxdiff,ro)]).T, fmt = '%s', header = 'seqlet_idx mean_effect delta_mean_effect max_effect delta_max_effect')
    np.savetxt(outname+'_seqmotifstats.txt' ,np.append(np.array(names).reshape(-1,1), np.array(astats).astype(int), axis = 1), fmt = '%s', header = 'seq_idx common_in_A common_in_B unique_in_A unique_in_B')
    write_meme_file(amotifs, anames, 'ACGT', outname+'_seqlets.meme')






