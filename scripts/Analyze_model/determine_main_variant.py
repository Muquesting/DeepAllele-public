import numpy as np
import sys, os


f = np.genfromtxt(sys.argv[1], dtype = str)
outname = os.path.splitext(sys.argv[1])[0]+'_mainvar.txt'

obj = open(outname, 'w')
useq = np.unique(f[:,0])
for s, seq in enumerate(useq):
    mask = f[:,0] == seq
    amax = np.argmax(np.absolute(f[mask, 2].astype(float)))
    pred = f[mask][amax]
    obj.write(' '.join(pred)+'\n')

