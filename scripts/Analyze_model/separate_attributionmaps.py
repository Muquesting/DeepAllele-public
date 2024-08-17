import numpy as np
import sys, os

labels = np.load(sys.argv[1])
atts = np.load(sys.argv[2])

outdir = os.path.split(sys.argv[2])[0]
if '--outdir' in sys.argv:
    outdir = sys.argv[sys.argv.index('--outdir')+1]

for l, label in enumerate(labels):
    np.save(outdir.strip('/')+'/seq_idx_'+str(l)+'_'+str(label)+'_'+os.path.split(sys.argv[2])[1], atts[l])
