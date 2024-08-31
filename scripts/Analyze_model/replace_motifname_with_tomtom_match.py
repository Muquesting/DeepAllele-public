import numpy as np
import sys, os

from DeepAllele.io import readtomtom, isint
import argparse 

'''
Either replaces motif name with TF name from tomtom in meme file
or generates a separte file to translate motif names to TF names

'''

if __name__ == '__main__':
    
    
    
    tomtom = sys.argv[1] # output tsv from tomtom
    tnames, target, pvals, qvals = readtomtom(tomtom)
    
    vals = sys.argv[2] 
    # detemine if pvals or qvals should be used to determine association
    if vals == 'q':
        stat = qvals
    elif vals == 'p':
        stat = pvals
    
    cut = float(sys.argv[3]) # determine cutoff for q of p-value
    
    pwms = open(sys.argv[4], 'r').readlines() # pwm file that contains the same names as 
    outname = os.path.splitext(sys.argv[4])[0]
    if '--outname' in sys.argv:
        outname = sys.argv[sys.argv.index('--outname')+1]
    outname += vals + str(cut)
    
    # mask all values with too high statistics
    mask = stat <= cut
    tnames, target, stat = tnames[mask], target[mask], stat[mask]
    
    # the word characterizing the row with the pwm's name
    nameline = 'MOTIF'
    if '--nameline' in sys.argv:
        nameline = sys.argv[sys.argv.index('--nameline')+1]
    
        
    # change tnames if they were cut by tomtom
    tnames = tnames.astype('U1000')
    pwmnames = []
    for l, line in enumerate(pwms):
        linesplit = line.split()
        if len(linesplit) > 0:
            if linesplit[0].upper() == nameline.upper(): # check fi nameline is first string in line
                pwm_name = line[len(linesplit[0])+1:].strip() # name of the cluster
                pwmnames.append(pwm_name)
    for t, tname in enumerate(tnames):
        if tname not in pwmnames:
            #print(tname)
            for pwm_name in pwmnames:
                if tname in pwm_name:
                    tnames[t] = pwm_name
                    #print(tnames[t])
    
    # split the taret names if they contain for example version names species names
    # target names is a copy of target otherwise, target can be used for filtering later
    targetnames = np.copy(target)
    if '--split_tomtomnames' in sys.argv:
        split = sys.argv[sys.argv.index('--split_tomtomnames')+1] # character used for split
        s = int(sys.argv[sys.argv.index('--split_tomtomnames')+2]) # index of entry used after split
        targetnames = np.array([tn.split(split)[s] for tn in target])
    
    # filter target names for a specific key word, f.e. MOUSE, or a specific version
    filt = None
    if '--filter_tomtomnames' in sys.argv:
        filt = sys.argv[sys.argv.index('--filter_tomtomnames')+1]
        outname += filt
    
    # if True only most significant naem will be assigned, otherwise all that pass
    only_best = False
    if '--only_best' in sys.argv:
        if isint(sys.argv[sys.argv.index('--only_best')+1]):
            only_best = int(sys.argv[sys.argv.index('--only_best')+1])
        else:
            only_best = 1
        outname += 'best'+str(only_best)
    
    # if cluster name should be reduced from Clust_X to X for example
    rsplit = None
    if '--reduce_clustername' in sys.argv:
        rsplit = sys.argv[sys.argv.index('--reduce_clustername')+1]
    
    # Splits cluster name at rnsplit and joins the first 4 and adds '...' if more than 4 are present
    rnsplit = None
    if '--reduce_nameset' in sys.argv:
        rnsplit = sys.argv[sys.argv.index('--reduce_nameset')+1]
        
    usepwmid = False
    if '--usepwmid' in sys.argv:
        usepwmid = True
            
        
    
    # If selected create a file with assigned gene names instead of pwm file with differed names
    gennamefile = False
    if '--generate_namefile' in sys.argv:
        gennamefile = True
        print(outname+'_altnames.txt')
        modpwms = open(outname+'_altnames.txt', 'w')
    else:
        print(outname+os.path.splitext(sys.argv[4])[1])
        modpwms = open(outname+os.path.splitext(sys.argv[4])[1], 'w')
    
    icount = -1
    for l, line in enumerate(pwms):
        linesplit = line.split()
        if len(linesplit) > 0:
            if linesplit[0].upper() == nameline.upper(): # check fi nameline is first string in line
                icount += 1
                pre = line[:len(linesplit[0])+1] # use identical nameline and delimiter as in original file
                if usepwmid:
                    orig_name = str(icount)
                else:
                    orig_name = line[len(linesplit[0])+1:].strip() # name of the cluster
                pot_name = line[len(linesplit[0])+1:].strip() # name of the cluster
                mask = tnames == pot_name # find all possible names for this cluster
                if rsplit is not None: # check if pot_name should potentially be trimmed
                    if rnsplit is not None:
                        pot_name = pot_name.split(rnsplit)
                        for p, ptn in enumerate(pot_name):
                            pot_name[p] = ptn.split(rsplit)[-1]
                        if len(pot_name) > 4:
                            pot_name = pot_name[:5]
                            pot_name[4] = '...'
                        pot_name = rnsplit.join(np.array(pot_name))
                    else:
                        if rsplit == 'empty':
                            pot_name = ''
                        elif rsplit == 'Number':
                            pot_name = str(icount)
                        else:
                            pot_name = pot_name.split(rsplit)[-1]
                # check for any matches
                if np.sum(mask) > 0:
                    ptarget, ptarname = target[mask], targetnames[mask]
                    # see if we have to filter these names for a string
                    if filt is not None:
                        keep = [filt in pt for pt in  ptarget]
                        if np.sum(keep) > 0:
                            ptarname, ptarget = ptarname[keep], ptarget[keep]
                        else:
                            ptarname = np.array([ptarname[0]+'*']) # * means that we couldn't find any with the filter but used the best name that did not have this filter
                    if only_best and len(ptarname) > 0: # only keep the most promising result in the name
                        ptarname = ptarname[:only_best]
                    pot_name = pot_name + '('+','.join(ptarname)+')'
                if gennamefile: # write a file that maps the original name to the new name
                    nline = str(orig_name)+'\t'+pot_name+ '\n'
                else: # combine all the pieces to new name for pwm file
                    nline = pre +pot_name + '\n'
                modpwms.write(nline)
                
            elif gennamefile == False: # if not a name line, and we didn't select to just create a name file, just write what was present in origional pwm file
                modpwms.write(line)
        elif not gennamefile:
            modpwms.write('\n')


