import numpy as np
import sys, os
import pandas as pd

varfile = pd.read_csv(sys.argv[1])
data = []
for i in range(len(varfile)):

    data.append(['seq_idx_'+str(varfile['seq_idxs'][i]), str(varfile['variant_idxs'][i]), -(varfile['ratio_A'][i]+varfile['ratio_B'][i])/2])

np.savetxt('ISM_avg_variant_effect.txt', data, fmt = '%s')


