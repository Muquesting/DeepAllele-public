import numpy as np
import sys, os
from align_seq import align_onehot

def write_meme_file(pwm, pwmname, alphabet, output_file_path):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    n_filters = len(pwm)
    print(n_filters)
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= "+alphabet+" \n")
    meme_file.write("strands: + -\n")

    print("Saved PWM File as : {}".format(output_file_path))

    for i in range(0, n_filters):
        meme_file.write("\n")
        meme_file.write("MOTIF %s \n" % pwmname[i])
        meme_file.write("letter-probability matrix: alength= "+str(len(alphabet))+" w= %d \n"% np.count_nonzero(np.sum(pwm[i], axis=0)))

        for j in range(0, np.shape(pwm[i])[-1]):
            for a in range(len(alphabet)):
                if a < len(alphabet)-1:
                    meme_file.write(str(pwm[i][ a, j])+ "\t")
                else:
                    meme_file.write(str(pwm[i][ a, j])+ "\n")

    meme_file.close()



def find_motifs(a, s, cut, mg, msig, avg = True):
    
    # Determine significant bases either based on all hypothetical attributions or only on the ones at the reference
    if avg:
        aloc = np.sum(np.absolute(a*s), axis = -1) >= cut
    else:
        aloc = np.sum(np.absolute(a) >= cut, axis = -1) > 0 # find locations where significant, > cut
    
    # Find motifs only based on what is present in the sequence, not what model would prefer
    a = -np.sum(a*s, axis = -1) # compute effect at reference
    
    sign = np.sign(a) # get sign of effects

    motiflocs = []

    gap = mg +1 # gapsize count
    msi = 1 # sign of motif
    potloc = [] # potential location of motif
    i = 0 
    while i < len(a):
        #print(potloc, gap)
        if aloc[i]: # if location significant
            if len(potloc) == 0: # if potloc just started
                msi = np.copy(sign[i]) # determine which sign the entire motif should have
            if sign[i] == msi: # check if base has same sign as rest of motif
                potloc.append(i)
                gap = 0
            elif msi *np.mean(a[max(0,i-mg): min(len(a),i+mg+1)]) < cut: # if sign does not match the sign of rest, then treat as gap
                gap += 1
                if gap > mg: # check that gap is still smaller than maximum gap size
                    if len(potloc) >= msig: 
                        motiflocs.append(potloc)
                        #print(a[potloc], a[potloc[0]:potloc[-1]])
                    if len(potloc) > 0:
                        i -= gap
                    gap = mg + 1
                    potloc = []
        elif msi *np.mean(a[max(0,i-mg): min(len(a),i+mg+1)]) < cut:
            gap +=1
            if gap > mg:
                if len(potloc) >= msig:
                    motiflocs.append(potloc)
                    #print(a[potloc], a[potloc[0]:potloc[-1]])
                if len(potloc) > 0:
                    i -= gap
                gap = mg + 1
                potloc = []
        i += 1
    if len(potloc) >= msig:
        motiflocs.append(potloc)
    return motiflocs

if __name__ == '__main__':
    statfile = sys.argv[1] # seq_labels
    atts = sys.argv[2] # deeplift attributions 
    seqs = sys.argv[3] # one-hot encoded sequences
    cut = float(sys.argv[4]) # significance cut off
    maxgap = int(sys.argv[5]) # max gap size
    minsig = int(sys.argv[6]) # minimum number of significant bases
    avg = True # whether to determine significance on reference or on all base effects


    outname = os.path.splitext(statfile)[0]+os.path.splitext(atts)[0]+'cut'+str(cut)+'maxg'+str(maxgap)+'minsiq'+str(minsig)

    stats = np.load(statfile)
    atts = np.load(atts)
    seqs = np.load(seqs)

    anames = [] # names of motifs
    amotifs = [] # motifs taken from z-scored ism
    ameans = [] # mean value of motifs
    astats = [] # statistics each input, how many motifs in each variant.
    otherloc = [] # location in the other sequence
    ameandiff = [] # difference of attributions between two sequences 

    z = 1
    if '--normed' in sys.argv:
        z=np.sqrt(np.mean(atts**2))

    for s, stat in enumerate(stats):
        name = 'seq_idx_'+str(s)+'_'+stat
        ism = atts[s]
        if '--ratioattributions' in sys.argv:
            ism[...,-1] = -ism[...,-1]
        
        seqonehot = seqs[s]
        lseqs = [len(np.where(seqonehot[:,:,j] == 1)[0]) for j in range(2)]
        seqo = align_onehot(seqonehot[None]) # align sequences to match motif locations for 'common'
        align = seqo[-1] # location of bases in aligned sequences
        seqo = seqo[:2] # translation of location of bases in first and second sequence. 
        print(name)
        zsm = ism/z # z-score norm ism with std
        seqmotifs = []
        for j in range(2): # iterate over both seqences
            motifs = find_motifs(zsm[:lseqs[j],:,j], seqonehot[:lseqs[j], :,j], cut, maxgap, minsig, avg = avg) # find the motifs in the ISM
            seqmotifs.append(motifs)
            print(len(motifs))
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
                motmean = np.mean(-np.sum(ism[seqmotifs[j][i][0]:seqmotifs[j][i][-1]+1,:, j], axis = -1)/3) # mean is computed from original isms
                # compute mean of the same bases in the other sequence to get delta mean
                altlocmot = seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]
                # some bases might be not present because of insertions
                if len(altlocmot) > 0:
                    altmotmean = np.mean(-np.sum(ism[np.amin(seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]):np.amax(seqo[j-1][np.isin(seqo[j], seqmotifs[j][i])]) +1,:, j-1], axis = -1)/3)
                else:
                    altmotmean = 0
                motmeandiff = altmotmean - motmean
                # use sign(mean) * z-score attribution --> pwm file npz
                mot = mot * np.sign(motmean) # adjust signs to make comparable and only align motifs with positive correlation
                # also save file that contains information about where to find this motif in the other sequence
                otherloc.append(','.join(np.array(theotherloc[j][i]).astype(str)))
                amotifs.append(mot.T)
                # name by seq, start-end, and j, common or not --> pwm file
                anames.append(name+'_'+str(seqmotifs[j][i][0])+'-'+str(seqmotifs[j][i][-1])+'_'+str(j)+'_'+str(common[j][i]))
                #print(anames[-1])
                ameans.append(motmean)
                ameandiff.append(motmeandiff)

    np.savetxt(outname+'_otherloc.txt' ,np.array([anames, otherloc]).T, fmt = '%s')
    np.savetxt(outname+'_meanatt.txt' ,np.array([anames, ameans, ameandiff]).T, fmt = '%s')
    np.savetxt(outname+'_seqmotstats.txt' ,np.append(stats[:,[0]], np.array(astats).astype(int), axis = 1), fmt = '%s')
    write_meme_file(amotifs, anames, 'ACGT', outname+'_attmotifs.meme')






