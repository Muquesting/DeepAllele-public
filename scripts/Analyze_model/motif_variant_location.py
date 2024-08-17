import numpy as np
import sys, os
from functools import reduce

def readtomtom(ttfile):
    lines = open(ttfile).readlines()
    cluster, tfs, pvalues = [],[],[]
    for l, line in enumerate(lines):
        if l > 0:
            line = line.strip().split()
            cluster.append(line[0].split('_')[-1])
            tfs.append(line[1])
            pvalues.append(float(line[3]))
    return np.array(cluster), np.array(tfs), -np.log10(np.array(pvalues))

mainvar = np.genfromtxt(sys.argv[1], dtype = str) # file with main variant locations
motifclusters = np.genfromtxt(sys.argv[2], dtype = str) # clusters of motifs
motifalign = np.genfromtxt(sys.argv[3], dtype = str) # location of motif in alignments
seqnames = np.genfromtxt(sys.argv[4], dtype = str) # list of predictable sequences


if not np.array_equal(motifclusters[:,0], motifalign[:,0]):
    print('motifclusters and motiflocations dont contain same motifs')
    sys.exit()

# sequence names of motifclusters and motifalign
motifseq = np.array(['seq_idx_'+m.split('_')[2] for m in motifclusters[:,0]])

# reduce mainvar, motifclusters, and motifalign to sequences in seqnames
motmask = np.argsort(motifseq)[np.isin(np.sort(motifseq), seqnames)]
motifalign, motifclusters, motifseq = motifalign[motmask], motifclusters[motmask], motifseq[motmask]

mainvar = mainvar[np.argsort(mainvar[:,0])]
varmask = np.isin(mainvar[:,0], seqnames)
mainvar = mainvar[varmask]

# extracth motif location and cluster from files
motifloc = []
motifclust = []
for m, mn in enumerate(motifclusters[:,0]):
    loc = np.arange(int(motifalign[m,1].split(',')[0]), int(motifalign[m,1].split(',')[-1])+1, dtype = int)
    motifloc.append(loc)
    motifclust.append([motifclusters[m,1] for l in range(len(loc))])
motifloc = np.array(motifloc)
motifclust = np.array(motifclust)

#if not np.array_equal(mainvar[:,0], motifseq):
#    print('motifseqs and mainvar dont contain same motifs')
#    print(len(mainvar), len(motifseq))
#    sys.exit()


varloc = []
for var in mainvar[:,1]:
    if ':' in var:
        vl = var.split(':')
        vl = np.arange(int(vl[0]), int(vl[1])+1)
    else:
        vl = np.arange(int(var), int(var)+1)
    varloc.append(vl)
varloc = np.array(varloc)

varclust = -np.ones(len(mainvar))
for v in range(len(mainvar)):
    seq = mainvar[v,0]
    potmot = np.isin(motifseq, seq)
    moloc = motifloc[potmot]
    if len(moloc) > 0:
        inter = np.isin(np.concatenate(moloc), varloc[v])
        hitclust, hn = np.unique(np.concatenate(motifclust[potmot])[inter], return_counts = True)
        if len(hitclust) == 1:
            varclust[v] = hitclust[0]
        elif len(hitclust) > 1:
            varclust[v] = hitclust[np.argmax(hn)]
    
print('Total number investigated', len(varclust))
print('Outside motif %', round(np.sum(varclust == -1)*100/len(varclust),2))
print('Mean number of sequences in a motif cluster %', round(np.sum(varclust != -1)/len(np.unique(varclust)),1))
N=len(varclust)

# return a list with clusters that are affected by a variant
cl, cln = np.unique(varclust[varclust != -1], return_counts = True)
sort = np.argsort(-cln)
cl, cln = cl[sort].astype(int).astype('<U20'), (cln[sort]/N)*100

names = []
namenum = []
for s in range(len(cl)):
    #print(cl[s], round(cln[s],1))
    names.append('Cluster_'+str(cl[s]))
    namenum.append(round(cln[s],1))
# only put cluster names into the nameset if they have more than one sequence that they affect
if '--savemotiflist' in sys.argv:
    thresho = int(sys.argv[sys.argv.index('--savemotiflist')+1])
    names = np.array(names)[np.array(namenum) > thresho]
    np.savetxt(os.path.splitext(sys.argv[1])[0]+'_clusterset.txt', names, fmt = '%s')
    print(os.path.splitext(sys.argv[1])[0]+'_clusterset.txt')

add = ''
if '--TFenrichment' in sys.argv:
    tffile = sys.argv[sys.argv.index('--TFenrichment')+1]
    tfcl, tfnames, pvalues = readtomtom(tffile)
    tfmask = pvalues > -np.log10(0.05)
    tfcl, tfnames, pvalues = tfcl[tfmask], tfnames[tfmask], pvalues[tfmask]
    tfs = np.unique(tfnames)
    sigtfs = np.zeros(len(tfs))
    for t, tf in enumerate(tfs):
        tfc = tfcl[tfnames == tf]
        tfp = pvalues[tfnames == tf]
        for c, tc in enumerate(tfc):
            sigtfs[t] += cln[cl==tc] * tfp[c]
    sorttfs = np.argsort(-sigtfs)
    ncl = np.copy(cl)
    ncl = ncl.astype('<U100')
    ncl[:] = '-'
    for s in sorttfs:
        tfc = tfcl[tfnames == tfs[s]]
        mask = np.isin(cl, tfc)
        mask = (ncl == '-') * mask
        ncl[mask] = tfs[s]
    cl = np.unique(ncl)
    ncln = np.zeros(len(cl))
    for c, cc in enumerate(cl):
        ncln[c] = np.sum(cln[ncl == cc])
    print(cl, ncln)
    cln = ncln
    sort = np.argsort(-cln)
    cl, cln = cl[sort], cln[sort]
    add = 'TFset'


# reduce all cluster with less than 2 entries to Others
mask = cln > 1
cl = np.append(cl[mask], ['Other<1%'])
cln = np.append(cln[mask], np.sum(cln[~mask]))
cl = np.append(cl, ['Outside']) # append outside for -1, or for variants outside a motif
cln = np.append(cln, np.sum(varclust == -1)*100/N)
print(cl, cln)

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (3.,3.), dpi = 200)
ax = fig.add_subplot(111)
explode = np.zeros(len(cln))
explode[-1] = 0.1

colors = [np.array([0.4+0.5*i/len(cln),0.5-0.4*i/len(cln),0.7-0.5*i/len(cln),1]) for i in range(len(cln))]
colors[-1] = np.array([0.3,0.3,0.3,1])

ax.pie(cln, labels=cl, colors = colors, explode = explode) #autopct='%1.1f%%', explode = explode)
fig.savefig(os.path.splitext(sys.argv[1])[0]+'_'+add+'_clusterpie.jpg', dpi = 300, bbox_inches = 'tight')
print(os.path.splitext(sys.argv[1])[0]+'_'+add+'_clusterpie.jpg')
plt.show()



