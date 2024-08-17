import numpy as np
import sys, os

f1 = np.genfromtxt(sys.argv[1], dtype = str)
f2 = np.genfromtxt(sys.argv[2], dtype = str)

if not  np.array_equal(f1[:,0], f2[:,0]) or not np.array_equal(f1[:,1], f2[:,1]):
    n1 = np.array([s[0]+s[1] for s in f1])
    n2 = np.array([s[0]+s[1] for s in f2])

    sort1 = np.argsort(n1)
    sort2 = np.argsort(n2)
    f1 = f1[sort1]
    f2 = f2[sort2]

if np.array_equal(f1[:,0], f2[:,0]) and np.array_equal(f1[:,1], f2[:,1]):
    neffect = (f1[:,2].astype(float) + f2[:,2].astype(float))/2
    data = np.copy(f1)
    data[:,2] = neffect
    intersect_dirs = np.intersect1d(os.path.split(sys.argv[1])[0].split('/'), os.path.split(sys.argv[2])[0].split('/'))
    intersect_path = np.array(os.path.split(sys.argv[1])[0].split('/'))
    path = '/'.join(intersect_path[np.isin(intersect_path, intersect_dirs)])
    filename = os.path.split(sys.argv[1])[1].split('_')
    filename = filename[:1] + ['avg'] + filename[1:]
    filename = '_'.join(np.array(filename))
    np.savetxt(path+'/'+filename, data, fmt = '%s')
    print('Saved in', path+'/'+filename)


