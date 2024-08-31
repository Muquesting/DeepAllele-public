import numpy as np
import sys, os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='zscore_datacolumn',
                    description='Transforms data column by dividing through standard deviations and subtraing the mean')
    parser.add_argument('datafile', type=str)
    parser.add_argument('--mean', type=float, default = 0.)
    parser.add_argument('--compute_mean', action='store_false')
    parser.add_argument('--column', type = int, default = -1)
    args = parser.parse_args()
    
    f = np.genfromtxt(args.datafile, dtype = str)
    outname = os.path.splitext(args.datafile)[0]+'_zscore.txt'
    
    z = f[:,args.column].astype(float)
    
    if args.compute_mean:
        args.mean = np.mean(z)
    z -= args.mean
    # Compute std to defined mean
    z = z/np.sqrt(np.mean(z**2))
    f[:,args.column] = z
    f = f[:,[0,args.column]]
    np.savetxt(outname, f, fmt = '%s')
