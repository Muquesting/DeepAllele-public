import numpy as np
import sys, os
import matplotlib as mpl
import logomaker
import pandas as pd

import matplotlib.pyplot as plt
from DeepAllele.motif_analysis import torch_compute_similarity_motifs, reverse, combine_pwms
from DeepAllele.io import numbertype, isint, readin_motif_files
from DeepAllele.plotlib import plot_pwms

    
    
if __name__ == '__main__':  
    pwmfile = sys.argv[1]
    infmt = os.path.splitext(pwmfile)[1]
    outname = os.path.splitext(pwmfile)[0]
    
    pwm_set, pwmnames, nts = readin_motif_files(pwmfile)

    if '--select' in sys.argv:
        select = sys.argv[sys.argv.index('--select')+1]
        if ',' in select:
            select = select.split(',')
        else:
            select = [select]
        for s, si in enumerate(select):
            if isint(si):
                select[s] = int(si)
            else:
                select[s] = list(pwmnames).index(si)
        pwm_set = pwm_set[select]
        pwmnames = pwmnames[select]
    
    bpwmnames = None
    if '--basepwms' in sys.argv:
        bpwm_set, bpwmnames, nts = readin_motif_files(sys.argv[sys.argv.index('--basepwms')+1])
        if '--clusterfile' in sys.argv: # Need to be in same order
            bpwm_cluster = np.genfromtxt(sys.argv[sys.argv.index('--clusterfile')+1], dtype = str)
            if not np.array_equal(bpwm_cluster[:,0], bpwmnames):
                print('clusterfile does not match basepwm file')
                sys.exit()
            bpwm_cluster = bpwm_cluster[:,1]
        elif ';' in ''.join(pwnnames):
            bpwm_clusters = -np.ones(len(bpwmnames)).astype(str)
            for p, pn in enumerate(pwmnames):
               bpwm_clusters[np.isin(bpwmnames, pn.split(';'))] = str(p)
            pwmnames = np.arange(len(pwmnames)).astype(str)
        else:
            print('No assignment given for basepwms')
            sys.exit()
        collect = 20
    
    log = False
    if '--infocont' in sys.argv:
        outname += 'ic'
        log = True
    
    for p, pwm in enumerate(pwm_set):
        name = pwmnames[p]
        pwm = pwm_set[p]
        
        outadd = ''
        if bpwmnames is not None:
            pwm = [pwm]
            outadd = 'wset'
            where = np.random.permutation(np.where(bpwm_cluster == name)[0])
            for f in where[:collect]:
                pwm.append(bpwm_set[f])
        
        fig = plot_pwms(pwm, log = log, showaxes = True)
        fig.savefig(outname+'.'+name+outadd+'.jpg', bbox_inches = 'tight', dpi = 300)
        print(outname+'.'+name+outadd+'.jpg')
        
        plt.show()


    
