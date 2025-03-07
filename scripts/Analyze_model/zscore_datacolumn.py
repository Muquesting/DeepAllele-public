import numpy as np
import sys, os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='zscore_datacolumn',
                    description='Transforms data column by dividing through standard deviations and subtraing the mean')
    parser.add_argument('datafile', type=str)
    parser.add_argument('--mean', type=float, default = 0.)
    parser.add_argument('--compute_mean', action='store_true')
    parser.add_argument('--column', type = int, default = -1)
    parser.add_argument('--noheader', action='store_false')
    parser.add_argument('--prefix', type = str, default = 'seq_idx')

    args = parser.parse_args()
    
    f = np.genfromtxt(args.datafile, dtype = str)
    
    outname = os.path.splitext(args.datafile)[0]+'_zscore.txt'
    
    if args.noheader:
        header = f[0][[0, args.column]]
        f = f[1:]
    else:
        header = np.array(['seq_idx', 'z_score'])
    z = f[:,args.column].astype(float)
    
    if args.compute_mean:
        args.mean = np.mean(z)
    z -= args.mean
    # Compute std to defined mean
    std = np.sqrt(np.mean(z**2))
    z = z/std
    f[:,args.column] = z
    f = f[:,[0,args.column]].astype('<U200')
    
    if not args.prefix in f[0,0]:
        for i, fi in enumerate(f):
            f[i,0] = args.prefix+'_'+fi[0]

    np.savetxt(outname, f, fmt = '%s', header = ' '.join(header))
