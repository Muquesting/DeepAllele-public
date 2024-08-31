'''
Reads motif files and enables manipulation of motifs, e.g. normalization
then write output in meme file. 
'''

import numpy as np
import sys, os
from DeepAllele.io import numbertype, readin_motif_files, write_meme_file
    
    
if __name__ == '__main__':
    
    pwmfile = sys.argv[1]
    pwms, names, nts = readin_motif_files(pwmfile)
    outname = os.path.splitext(pwmfile)[0]
    
    pwms = [pwm.T for pwm in pwms]
    
    if '--set' in sys.argv:
        setfile = sys.argv[sys.argv.index('--set')+1]
        tset = np.genfromtxt(setfile, dtype = str)
        mask = np.where(np.isin(names, tset))[0]
        outname += '_'+ os.path.splitext(os.path.split(setfile)[1])[0].rsplit('_',1)[-1]
        pwms, names = [pwms[i] for i in mask], [names[i] for i in mask]
        
    if '--adjust_sign' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.sign(np.sum(pwm[np.argmax(np.absolute(pwm),axis = 0),np.arange(len(pwm[0]),dtype = int)]))*pwm
    
    if '--standardize' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] /= np.sqrt(np.mean(pwm**2))
    
    if '--exppwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = np.exp(pwm)
    
    if '--strip' in sys.argv:
        stripcut = float(sys.argv[sys.argv.index('--strip')+1])
        for p,pwm in enumerate(pwms):
            pwmsum = np.sum(pwm,axis=0)
            mask = np.where(pwmsum >= stripcut*np.amax(pwmsum))[0]
            if mask[-1]-1 > mask[0]:
                pwms[p] = pwm[:,mask[0]:mask[-1]+1]
            else:
                print('No entries in pwm', names[p], pwmsum)
                print('Change stripcut')
                sys.exit()
    
    if '--norm' in sys.argv or '--normpwms' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwms[p] = pwm/np.sum(pwm,axis =0)
    
    if '--infocont' in sys.argv:
        for p,pwm in enumerate(pwms):
            pwm = np.log2((pwm+1e-16)/0.25)
            pwm[pwm<0] = 0
            pwms[p] = pwm
    
    if '--changenames' in sys.argv:
        clusters = np.arange(len(names)).astype(str)
    else:
        clusters = names
    outname += '.meme'
    
    if '--round' in sys.argv:
        r = int(sys.argv[sys.argv.index('--round')+1])
        for p,pwm in enumerate(pwms):
            pwms[p] = np.around(pwm, r)
    write_meme_file(pwms, clusters, ''.join(nts), outname)
    
    
    
    
    
