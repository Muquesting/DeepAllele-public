import numpy as np
import sys, os
import matplotlib as mpl
import logomaker
import pandas as pd

import matplotlib.pyplot as plt
from DeepAllele.motif_analysis import torch_compute_similarity_motifs, reverse, combine_pwms
from DeepAllele.io import numbertype, isint, readin_motif_files
from DeepAllele.plotlib import plot_pwms

import argparse    
    
if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser(prog='plot_pwm_logos',
                    description='plot logos, or add logos that were aligned to generate logo')
    parser.add_argument('pwmfile', type=str)
    parser.add_argument('--select', type=str, default = None, help='Set of logo names in pwmfile that should be plotted')
    parser.add_argument('--basepwms', type=str, default = None, help = 'Meme file with pwms that were used to generate the pwms in pwmfile. 20 of these will be aligned and plotted with the pwm in pwmfile.')
    parser.add_argument('--clusterfile', type=str, default = None, help='Assignment of basepwms to pwms in pwmfile')
    parser.add_argument('--infocont', action='store_true', help='If True, logos are converted to information content')
    parser.add_argument('--reverse_complement', action='store_true', help='If True the logo will be reversed. Only works for ACGT order')
    
    args = parser.parse_args()
    
    pwmfile = args.pwmfile
    infmt = os.path.splitext(pwmfile)[1]
    outname = os.path.splitext(pwmfile)[0]
    
    pwm_set, pwmnames, nts = readin_motif_files(pwmfile)

    if args.select is not None:
        select = args.select
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
    if args.basepwms is not None:
        bpwm_set, bpwmnames, nts = readin_motif_files(args.basepwms)
        if args.clusterfile is not None: # Need to be in same order
            bpwm_cluster = np.genfromtxt(args.clusterfile, dtype = str)
            if not np.array_equal(bpwm_cluster[:,0], bpwmnames):
                print('clusterfile does not match basepwm file')
                sys.exit()
            bpwm_cluster = bpwm_cluster[:,1]
        elif ';' in ''.join(pwmnames):
            bpwm_clusters = -np.ones(len(bpwmnames)).astype(str)
            for p, pn in enumerate(pwmnames):
               bpwm_clusters[np.isin(bpwmnames, pn.split(';'))] = str(p)
            pwmnames = np.arange(len(pwmnames)).astype(str)
        else:
            print('No assignment given for basepwms')
            sys.exit()
        collect = 20
    
    log = False
    if args.infocont:
        outname += 'ic'
        log = True
    
    for p, pwm in enumerate(pwm_set):
        name = pwmnames[p]
        pwm = pwm_set[p]
        outadd = ''
        if args.reverse_complement:
            pwm=reverse(pwm)
            outadd += 'rev'
        if bpwmnames is not None:
            pwm = [pwm]
            outadd = 'wset'
            where = np.random.permutation(np.where(bpwm_cluster == name)[0])
            for f in where[:collect]:
                pwm.append(bpwm_set[f])
        
        fig = plot_pwms(pwm, log = log, showaxes = True, revcomp_matrix = True)
        fig.savefig(outname+'.'+name+outadd+'.jpg', bbox_inches = 'tight', dpi = 300)
        print(outname+'.'+name+outadd+'.jpg')
        
        plt.show()


    
