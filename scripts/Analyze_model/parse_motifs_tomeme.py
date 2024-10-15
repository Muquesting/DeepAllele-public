'''
Reads motif files and enables manipulation of motifs, e.g. normalization
then write output in meme file. 
'''

import numpy as np
import sys, os
from DeepAllele.io import numbertype, readin_motif_files, write_meme_file
import argparse    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='parse_motifs_tomeme',
                    description='read motif file and modify pwms')
    parser.add_argument('pwmfile', type=str)
    parser.add_argument('--set', type=str, default = None, help='Set of logo names in pwmfile that should be kept')
    parser.add_argument('--strip', type=float, default = None, help = 'fraction of max abs value per position that should be kept and not removed from logo on the left and right side')
    parser.add_argument('--round', type=int, default = None, help='dezimal to round final motifs')
    
    parser.add_argument('--infocont', action='store_true', help='If True, motifs are converted to information content')
    parser.add_argument('--adjust_sign', action='store_true', help='If True sign will be adjusted to the sum of the abs max in the pwm')
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--exppwms', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--changenames', action='store_true')
    
    args = parser.parse_args()
    
    pwmfile = args.pwmfile
    pwms, names, nts = readin_motif_files(pwmfile)
    outname = os.path.splitext(pwmfile)[0]
    
    pwms = [pwm.T for pwm in pwms]
    
    if args.set is not None:
        setfile = args.set
        tset = np.genfromtxt(setfile, dtype = str)
        mask = np.where(np.isin(names, tset))[0]
        outname += '_'+ os.path.splitext(os.path.split(setfile)[1])[0].rsplit('_',1)[-1]
        pwms, names = [pwms[i] for i in mask], [names[i] for i in mask]
        
    if args.adjust_sign:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.sign(np.sum(pwm[np.argmax(np.absolute(pwm),axis = 0),np.arange(len(pwm[0]),dtype = int)]))*pwm
    
    if args.standardize:
        for p,pwm in enumerate(pwms):
            pwms[p] /= np.sqrt(np.mean(pwm**2))
    
    if args.exppwms:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.exp(pwm)
    
    if args.strip is not None:
        stripcut = args.strip
        for p,pwm in enumerate(pwms):
            pwmsum = np.sum(pwm,axis=0)
            mask = np.where(pwmsum >= stripcut*np.amax(pwmsum))[0]
            if mask[-1]-1 > mask[0]:
                pwms[p] = pwm[:,mask[0]:mask[-1]+1]
            else:
                print('No entries in pwm', names[p], pwmsum)
                print('Change stripcut')
                sys.exit()
    
    if args.norm:
        for p,pwm in enumerate(pwms):
            pwms[p] = pwm/np.sum(pwm,axis =0)
    
    if args.infocont:
        for p,pwm in enumerate(pwms):
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            pwms[p] = pwm
    
    if args.changenames:
        clusters = np.arange(len(names)).astype(str)
    else:
        clusters = names
    outname += '.meme'
    
    if args.round is not None:
        r = args.round
        for p,pwm in enumerate(pwms):
            pwms[p] = np.around(pwm, r)
    write_meme_file(pwms, clusters, ''.join(nts), outname)
    
    
    
    
    
