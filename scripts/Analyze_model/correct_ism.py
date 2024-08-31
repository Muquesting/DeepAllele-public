import numpy as np
import sys, os


if __name__ == '__main__':
    npy = np.load(sys.argv[1])
    print('Shape attributions', np.shape(npy))
    # Attributions are of shape = (N_data, L_data, n_bases=4, n_alleles = 2)
    npy = npy - np.mean(npy,axis=-2)[:,:,None, :]
    np.save(os.path.splitext(sys.argv[1])[0]+'.imp.npy', npy)


