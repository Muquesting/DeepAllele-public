import numpy as np
import sys, os


f = np.genfromtxt(sys.argv[1], dtype = str)
outname = os.path.splitext(sys.argv[1])[0]+'_seqpred.txt'

print(f'Saved in {outname}')
obj = open(outname, 'w')
useq = np.unique(f[:,0])
for s, seq in enumerate(useq):
    mask = f[:,0] == seq
    pred = np.sum(f[mask, 2].astype(float))
    obj.write(seq + ' ' + str(pred)+'\n')

