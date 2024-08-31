import numpy as np
import sys, os
import matplotlib.pyplot as plt
import logomaker as lm
import pandas as pd
from .motif_analysis import align_onehot, torch_compute_similarity_motifs 
from matplotlib import cm
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib as mpl
from scipy.stats import gaussian_kde, pearsonr
from sklearn import linear_model
import matplotlib.patches as mpatches

def add_frames(att, locations, colors, ax):
    '''
    Adds frames around locations in colors
    
    Parameters
    ---------
    locations :
        list of tuples
    '''
    att = np.array(att)
    for l, loc in enumerate(locations):
        mina, maxa = np.amin(np.sum(np.ma.masked_greater(att[loc[0]:loc[1]+1],0),axis = 1)), np.amax(np.sum(np.ma.masked_less(att[loc[0]:loc[1]+1],0),axis = 1))
        x = [loc[0]-0.5, loc[1]+0.5]
        ax.plot(x, [mina, mina], c = cmap[colors[l]])
        ax.plot(x, [maxa, maxa], c = cmap[colors[l]])
        ax.plot([x[0], x[0]] , [mina, maxa], c = cmap[colors[l]])
        ax.plot([x[1], x[1]] , [mina, maxa], c = cmap[colors[l]])


def _plot_attribution(att, ax, nts = list('ACGT'), labelbottom=False, bottom_axis = False, ylabel = None, ylim = None, xticks = None, xticklabels = None):
    '''
    Plot single atribubtion from np.array
    '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(bottom_axis)
    ax.tick_params(bottom = labelbottom, labelbottom = labelbottom)
    att = pd.DataFrame(att, columns = nts)
    lm.Logo(att, ax = ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

def plot_attribution(seq, att, add_perbase = False, motifs = None, seq_based = True, nts = list('ACGT'), plot_effect_difference = True, plot_difference = True, unit=0.04, xticks = None, xticklabels = None):
    '''
    Plots attributions of two sequences and illustrates the variant effect sizes
    
    Parameters
    ----------
    seq : 
        one-hot encoded sequence of shape = (length, channels=4, n_seqs=2)
    att : 
        sequence attributions encoded sequence of shape = (length, channels=4, n_seqs=2)
    add_perbase :
        if True add heatmap with per base attributions for each sequence and difference
    motifs :
        list of lists (sequence_name, tuple of locations, allele, and color)
    seq_based:
        if True, only the attributions at a present base are shown. Otherwise, 
        also attributions from bases that are not present in the sequence. This
        can be helpful to spot motifs that would be preferred by the model.
    
    TODO: 
        Automatically set xlim
        Use matplotlib axgrid to make space for heatmaps smaller
    '''
    
    
    if add_perbase:
        pbatt = att
        pbvlim = np.amax(np.absolute(pbatt))
    
    if seq_based:
        att = seq * att
        
    # Align one-hot encodings to visualize on top of each other
    si1, si2, ti1, ti2 = align_onehot(seq)
    maxlen = np.amax(np.concatenate([ti1,ti2]))+1
    
    if xticks is not None:
        if xticklabels is None:
            xticklabels = xticks
        xticks = ti1[xticks]
    
    attA = np.zeros((maxlen,4))
    attB = np.zeros((maxlen,4))
    fracAB = np.zeros((maxlen,4))
    seqAB = np.zeros((maxlen,4))
    
    attA[ti1] = att[si1,:,0]
    attB[ti2] = att[si2,:,1]
    fracAB = attA - attB
    
    seqAB[ti1] = seq[si1,:,0]
    seqAB[ti2] -= seq[si2,:,1]

    if add_perbase:
        pbattA = np.zeros((maxlen,4))
        pbattB = np.zeros((maxlen,4))
        pbattA[ti1] = pbatt[si1,:,0]
        pbattB[ti2] = pbatt[si2,:,1]
        

    mina,minb = np.array(np.sum(np.ma.masked_greater(attA,0), axis = -1)), np.array(np.sum(np.ma.masked_greater(attB,0), axis = -1))
    
    maxa,maxb = np.array(np.sum(np.ma.masked_less(attA,0), axis = -1)), np.array(np.sum(np.ma.masked_less(attB,0), axis = -1))
    
    attlim = [min(np.amin(np.append(mina,minb)),0),np.amax(np.append(maxa,maxb))]
    
    ah = 25
    buff = 2
    gN = (3*(ah+buff)+3*(4+buff)*int(add_perbase))
    w, h = unit*maxlen, gN*unit
    fig = plt.figure(figsize = (w, h))
    spec = fig.add_gridspec(ncols=1, nrows=gN)
    
    ax0 =  fig.add_subplot(spec[:ah,:])
    _plot_attribution(attA, ax0, ylabel = 'B6', ylim = attlim)
    

    if motifs is not None:
        mask = motifs[:,-2] == 0 # Only consider motifs in sequence 0 here
        colors = motifs[mask,-1] # get the colors of these
        locations = [ti1[l] for l in motifs[mask,1]] # get the locs of these
        add_frames(attA, locations, colors, ax0)
    
    if add_perbase: # add another subplot with a heatmap for per base effecst
        ax0b =  fig.add_subplot(spec[ah+buff:ah+buff+4,:])
        ax0b.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False)
        ax0b.imshow(pbattA.T, vmin = -pbvlim, vmax = pbvlim, cmap = 'coolwarm_r', aspect = 'auto')

    ax1 =fig.add_subplot(spec[ah+buff+(4+buff)*int(add_perbase):2*ah+buff+(4+buff)*int(add_perbase),:])
    _plot_attribution(attB, ax1, ylabel = 'CAST', ylim = attlim)
    
    if motifs is not None:
        mask = motifs[:,-2] == 1
        colors = motifs[mask,-1]
        locations = [ti2[l] for l in motifs[mask,1]]
        add_frames(attB, locations, colors, ax1)

    if add_perbase:
        ax1b =  fig.add_subplot(spec[2*(ah+buff)+(4+buff)*int(add_perbase):2*(ah+buff)+((4+buff)+4)*int(add_perbase),:])
        ax1b.tick_params(bottom = False, labelbottom = False, labelleft = False, left = False)
        ax1b.imshow(pbattB.T, vmin = -pbvlim, vmax = pbvlim, cmap = 'coolwarm_r', aspect = 'auto')

    if plot_difference:
        # Plot the difference between the two attributions
        axf =  fig.add_subplot(spec[2*(ah+buff)+2*(4+buff)*int(add_perbase):2*(ah+buff)+ah+2*(4+buff)*int(add_perbase),:])
        
        if plot_effect_difference:
            sumeffect = np.sum(fracAB*np.absolute(seqAB), axis = 1)
            fraclim = [min(np.amin(sumeffect),attlim[0]), max(np.amax(sumeffect),attlim[1])]
            _plot_attribution(fracAB*np.absolute(seqAB),axf, labelbottom = not add_perbase, bottom_axis = not add_perbase, ylabel = 'Effect\nsize', ylim = fraclim, xticks = xticks, xticklabels=xticklabels)
        else:
            sumeffect = np.sum(fracAB, axis = 1)
            fraclim = [min(np.amin(sumeffect),attlim[0]), max(np.amax(sumeffect),attlim[1])]
            _plot_attribution(fracAB,axf, labelbottom = not add_perbase, bottom_axis = not add_perbase, ylabel = 'Effect\nsize', ylim = fraclim, xticks = xticks, xticklabels=xticklabels)

        if add_perbase:
            axfb =  fig.add_subplot(spec[3*(ah+buff)+2*(4+buff)*int(add_perbase):3*(ah+buff)+2*(4+buff)*int(add_perbase)+4,:])
            axfb.tick_params(bottom = True, labelbottom = True, labelleft = False, left = False)
            axfb.imshow(pbattA.T-pbattB.T, vmin = -pbvlim, vmax = pbvlim, cmap = 'coolwarm_r', aspect = 'auto')
            if xticks is not None:
                axfb.set_xticks(xticks)
            if xticklabels is not None:
                axfb.set_xticklabels(xticklabels)
    return fig




def plot_single_pwm(pwm, log = False, showaxes = False, channels = list('ACGT'), ax = None):
    '''
    Plots single PWM, determines figsize based on length of pwm
    pwm : 
        shape=(length_logo, channels)
    '''
    if ax is None:
        fig = plt.figure(figsize = (np.shape(pwm)[0]*unit,np.shape(pwm)[1]*unit), dpi = 300)
        ax = fig.add_subplot(111)
    
    lim = [min(0, -np.ceil(np.around(np.amax(-np.sum(np.ma.masked_array(pwm, pwm >0),axis = 1)),2))), 
           np.ceil(np.around(np.amax(np.sum(np.ma.masked_array(pwm, pwm <0),axis = 1)),2))]
    
    if log:
        pwm = np.log2((pwm+1e-16)/0.25)
        pwm[pwm<0] = 0
        lim = [0,2]
    
    lm.Logo(pd.DataFrame(pwm, columns = channels), ax = ax, color_scheme = 'classic')
    ax.set_ylim(lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not showaxes:
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelleft = False, left = False, labelbottom = False, bottom = False)
    ax.set_yticks(lim)
    return ax

def reverse(pwm):
    return pwm[::-1][:,::-1]

def plot_pwms(pwm, log = False, showaxes = False, unit = 0.4, channels= list('ACGT'), offsets = None, revcomp_matrix = None, align_to = 0):
    '''
    Aligns and plots multiple pwms
    use align_to to determine to which pwm the others should be aligned
    set align_to to 'combine' to combine list of pwms and add combined motif
    at position 0
    '''
    if isinstance(pwm, list):
        if offsets is None:
            ifcont = True
            min_sim = 4
            for pw in pwm:
                min_sim = min(min_sim, np.shape(pw)[0])
                if (pw<0).any():
                    ifcont = False
            correlation, offsets, revcomp_matrix = torch_compute_similarity_motifs(pwm, pwm, exact=True, return_alignment = True, metric = 'correlation', min_sim = min_sim, infocont = ifcont, reverse_complement = revcomp_matrix is not None)
            offsets = offsets[:,align_to] # use offsets  from first pwm
            revcomp_matrix = revcomp_matrix[:,align_to] # use reverse complement assignment from first pwms
        else:
            if revcomp_matrix is None:
                revcomp_matrix = np.zeros(len(offsets))
                
        pwm_len=np.array([len(pw) for pw in pwm]) # array of pwm lengths
        
        # compute offsets for each pwm so that they will be put into an array,
        #so that all pwms will be aligned 
        offleft = abs(min(0,np.amin(offsets))) 
        offright = max(0,np.amax(offsets + pwm_len-np.shape(pwm[0])[0]))

        nshape = list(np.shape(pwm[0]))
        nshape[0] = nshape[0] + offleft + offright # total length that is 
        #needed to fit all pwms into region when aligned to each other

        fig = plt.figure(figsize = (len(pwm) * nshape[0]*unit,3*unit*nshape[1]), dpi = 50)
        for p, pw in enumerate(pwm):
            ax = fig.add_subplot(len(pwm), 1, p + 1)
            if revcomp_matrix[p] == 1:
                pw = reverse(pw)
            # create empty array with nshape
            if not log:
                pw0 = np.zeros(nshape)
            else: 
                pw0 = np.ones(nshape)*0.25
            pw0[offleft + offsets[p]: len(pw) + offleft + offsets[p]] = pw
            pw = pw0
            plot_single_pwm(pw, log=log, showaxes = showaxes, channels = channels, ax = ax)
    else:
        
        fig = plt.figure(figsize = (np.shape(pwm)[0]*unit,3*np.shape(pwm)[1]*unit), dpi = 300)
        ax = fig.add_subplot(111)
        plot_single_pwm(pwm, log = log, showaxes = showaxes, channels = channels, ax = ax)
        
    return fig


def _check_symmetric_matrix(distmat):
    # Test if distmat is symmetric
    if np.shape(distmat)[0] != np.shape(distmat)[1]:
        print( 'Warning: not symmetric matrix: "sortx" set to None if given')
        return False
    elif np.any(np.abs(distmat - distmat.T) > 1e-4):
        print(np.where(np.abs(distmat - distmat.T) > 1e-4))
        print(np.abs(distmat - distmat.T)[np.abs(distmat - distmat.T) > 1e-4])
        print( f'Warning: not symmetric matrix: max difference between transpose elements is {np.amax(np.abs(distmat - distmat.T))} \n "sortx" set to None if given')
        return False
    else:
        return True

def _transform_similarity_to_distance(distmat):
    # checks if similarity matrix or distance matrix
    issimilarity = np.all(np.amax(distmat) == np.diag(distmat))
    heatmax, heatmin = np.amax(distmat), np.amin(distmat)
    simatrix = int(issimilarity) - (2.*int(issimilarity)-1.) * (distmat - heatmin)/(heatmax - heatmin)

    return simatrix


def plot_heatmap(heatmat, # matrix that is plotted with imshow
                 ydistmat = None, # matrix to compute sorty, default uses heatmat
                 xdistmat = None, # matrix to compute sortx, default uses hetamat
                 measurex = None, # if matrix is not a symmetric distance matrix
                 # measurex defines distannce metric to compute distances for linkage clustering 
                 measurey = None, # same as measurex just for y axic
                 sortx = None, # agglomerative clustering algorith used in likage, f.e average, or single
                 sorty = None, # same as above but for y axis
                 x_attributes = None, # additional heatmap with attributes of columns
                 y_attributes = None, # same as above for y axis
                 xattr_name = None, # names of attributes for columns
                 yattr_name = None, # names of attributes for rows
                 heatmapcolor = cm.BrBG_r, # color map of main matrix
                 xatt_color = None, # color map or list of colormaps for attributes
                 yatt_color = None, 
                 xatt_vlim = None, # vmin and vmas for xattributes, or list of vmin and vmax
                 yatt_vlim = None,
                 pwms = None, # pwms that are plotted with logomaker next to rows of matrix
                 infocont = True, # if True, the matrices will be plotted as information content
                 combine_cutx = 0., # NOT implemented, can be used to cut off 
                 # linkage tree at certain distance and reduce its resolution
                 combine_cuty = 0., 
                 color_cutx = 0., # cut off for coloring in linkage tree. 
                 color_cuty = 0., 
                 xdenline = None, # line drawn into linkage tree on x-axis
                 ydenline = None, 
                 plot_value = False, # if true the values are written into the cells of the matrix
                 vmin = None, # min color value 
                 vmax = None, 
                 grid = False, # if True, grey grid drawn around heatmap cells
                 xlabel = None, # label on x-axis
                 ylabel = None, # ylabel
                 xticklabels = None,
                 yticklabels  = None,
                 showdpi = None, # dpi value for plotting with plt.show()
                 dpi = None, # dpi value for savefig
                 figname = None, # if given, figure saved under this name
                 fmt = '.jpg', # format of saved figure
                 maxsize = 150, # largest size the figure can take along both axis
                 cellsize = 0.3, # size of a single cell in the heatmap
                 cellratio = 1., # ratio of cells y/x
                 noheatmap = False, # if True, only tree is plotted
                 row_distributions = None, # for each row in heatmap, add 
                 # a box or a bar plot with plot_distribution, 
                 row_distribution_kwargs = {} # kwargs fro plot_distribution
                 ):
    '''
    Plots a heatmap with tree on x and y
    Motifs can be added to the end of the tree
    Attributions of each column or row can be indicated by additoinal heatmap with different color code
    Other statistics, for example, barplot or boxplots can be added to the y-axis
    Heatmap can be blocked and only tree with motifs and other statistics can be shown
    TODO 
    Put dedrogram and pwm plot in function.
    '''
    
    # Determine size of heatmap
    if heatmat is None:
        Nx = 0
        Ny = np.shape(ydistmat)[0]
    else:
        Ny, Nx = np.shape(heatmat)[0], np.shape(heatmat)[1]
    # Use heatmat as default if xdistmat not specified
    if xdistmat is None:
        xdistmat = np.copy(heatmat)
    if ydistmat is None:
        ydistmat = np.copy(heatmat)
    # either provide similarity matrix as heatmap (measurex = None) or provide 
    # a similarity function from scipy.spatial.distance.pdist
    # If no measure is provided heatmap will be tested whether it is a distance
    # matrix and entries will be rescaled between 0,1 
    if not noheatmap:
        if measurex is not None:
            simatrixX = pdist(xdistmat.T, metric = measurex)
        elif xdistmat is not None:
            
            if not _check_symmetric_matrix(xdistmat):
                sortx = None
            
            if sortx is not None:        
                # checks if similarity matrix or distance matrix
                simatrixX = _transform_similarity_to_distance(xdistmat)
                simatrixX = simatrixX[np.triu_indices(len(simatrixX),1)]
        else:
            sortx = None
            simatrixX = None
                
    if measurey is not None:
        simatrixY = pdist(ydistmat, metric = measurey)
    
    elif ydistmat is not None:
            if not _check_symmetric_matrix(ydistmat):
                sorty = None
            
            if sorty is not None:        
                # checks if similarity matrix or distance matrix
                simatrixY = _transform_similarity_to_distance(ydistmat)
                simatrixY = simatrixY[np.triu_indices(len(simatrixY),1)]
    else:
        sorty = None
        simatrixY = None
    
    
    
    # Generate dendrogram for x and y
    #### NJ not yet included
    if sortx is not None and not noheatmap:
        Zx = linkage(simatrixX, sortx)
        #if combine_cutx > 0:
            #Zx = reduce_z(Zx, combine_cutx)

    if sorty is not None:
        Zy = linkage(simatrixY, sorty) 
        #if combine_cuty > 0:
            #Zy = reduce_z(Zy, combine_cuty)
    
    if not noheatmap and heatmat is not None:
        # Check if maxsize is exceeded and adjust parameters accordingly
        if cellsize*np.shape(heatmat)[1] > maxsize:
            xticklabels = None
            plot_value = False
            yattr_name = None
            cellsize = min(maxsize/(np.shape(heatmat)[0]*cellratio), 
                           maxsize/np.shape(heatmat)[1])
    
        if cellsize*np.shape(heatmat)[0] *cellratio > maxsize:
            yticklabels = None
            plot_value = False
            x_attr_name = None
            cellsize = min(maxsize/(np.shape(heatmat)[0]*cellratio), 
                           maxsize/np.shape(heatmat)[1])
    
    # Determine sizes for the elements that will have to be plotted in figure
    # Plan for extra space for attributes
    xextra = 0.
    if y_attributes is not None:
        y_attributes = np.array(y_attributes, dtype = object)
        xextra = np.shape(y_attributes)[1] + 0.25
    yextra = 0.
    if x_attributes is not None:
        x_attributes = np.array(x_attributes, dtype = object)
        yextra = np.shape(x_attributes)[0] + 0.25
    # Plan for extra space for dendrogram and pwms
    denx, deny, pwmsize, rowdistsize = 0, 0, 0, 0
    if sortx is not None and not noheatmap:
        denx = 10 + 0.25
    if sorty is not None:
        deny = 3+.25
    if pwms is not None:
        pwmsize = 3.25
    # Plan for extra space if row_distributions are added to heatmap
    if row_distributions is not None:
        rowdistsize = 6+ 0.25
    
    basesize = 0
    
    wfig = cellsize*(Nx+xextra+deny+pwmsize+rowdistsize+basesize)
    hfig = cellsize*cellratio*(Ny+yextra/cellratio+denx+basesize)
    
    fig = plt.figure(figsize = (wfig, hfig), dpi = showdpi)
    
    fullw = Nx+xextra+deny+pwmsize+rowdistsize+basesize
    fullh = Ny+yextra+denx+basesize
    # Determine all fractions of figure that will be assigned to each subplot
    left = 0.1
    bottom = 0.1
    width = 0.8
    height = 0.8
    before = width*(deny+pwmsize)/fullw
    after = width*(xextra+rowdistsize)/fullw
    beneath = height*yextra/fullh
    above = height*denx/fullh
    
    # Final width that heatmap will take
    wfac = width * Nx/fullw
    mfac = height * Ny/fullh
    
    if not noheatmap:
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_position([0.1+before,0.1+beneath, wfac, mfac])
        ax.tick_params(which = 'both', bottom = False, labelbottom = False,
                       left = False, labelleft = False)
    
    # plot dendrogram for x axis
    if sortx is not None and not noheatmap:
        axdenx = fig.add_subplot(711)
        axdenx.spines['top'].set_visible(False)
        axdenx.spines['right'].set_visible(False)
        axdenx.spines['bottom'].set_visible(False)
        axdenx.tick_params(which = 'both', bottom = False, labelbottom = False)
        axdenx.set_position([0.1+before,0.9 - above, wfac, 
                             height*(denx-0.25)/fullh])
        dnx = dendrogram(Zx, ax = axdenx, no_labels = True, 
                         above_threshold_color = 'k', 
                         color_threshold = color_cutx, orientation = 'top')
        
        sortx = dnx['leaves']
        heatmat = heatmat[:, sortx]
        if x_attributes is not None:
            x_attributes = x_attributes[:, sortx]
            
        if xticklabels is not None:
            xticklabels = xticklabels[sortx]
            
        if xdenline is not None:
            axdenx.plot([0,len(heatmat[0])*10], [xdenline, xdenline], color = 'r')
    elif heatmat is not None:
        sortx = np.arange(Nx, dtype = int)
    
    sys.setrecursionlimit(100000)    
    
    if sorty is not None:
        axdeny = fig.add_subplot(171)
        axdeny.spines['top'].set_visible(False)
        axdeny.spines['right'].set_visible(False)
        axdeny.spines['left'].set_visible(False)
        axdeny.tick_params(which = 'both', left = False, labelleft = False)
        axdeny.set_position([0.1,0.1+beneath, width*(deny-0.25)/fullw, mfac])
        dny = dendrogram(Zy, ax = axdeny, no_labels = True, 
                         color_threshold = color_cuty, above_threshold_color = 'k',
                         orientation = 'left', get_leaves = True)
        sorty = dny['leaves']
        if heatmat is not None:
            heatmat = heatmat[sorty]
        #axdeny.set_yticks(axdeny.get_yticks()[1:])

        if y_attributes is not None:
            y_attributes = y_attributes[sorty]
            
        if yticklabels is not None:
            yticklabels = yticklabels[sorty]
        if ydenline is not None:
            axdeny.plot([ydenline, ydenline], [0,len(heatmat)*10], color = 'r')
    elif heatmat is not None:
        sorty = np.arange(len(heatmat), dtype = int)
    
    
    # Plot PWMs if given
    if pwms is not None:
        if infocont:
            pwm_min, pwm_max = 0,2
        else:
            pwm_min, pwm_max = 0, int(np.ceil(np.amax([np.amax(np.sum(np.ma.masked_less(pwm,0),axis = -1)) for pwm in pwms])))
        lenpwms = np.array([len(pwm) for pwm in pwms])
        maxlenpwms = np.amax(lenpwms)
        for s, si in enumerate(sorty[::-1]):
            axpwm = fig.add_subplot(len(sorty),1,s+1)
            axpwm.set_position([0.1+before-pwmsize*width/fullw, 
                                0.1+beneath+mfac-mfac*(s+0.9)/len(sorty),
                                (pwmsize-0.25)*width/fullw, 
                                mfac/len(sorty) *0.8])
            pwm = pwms[si]
            if infocont:
                pwm = np.log2((pwms[si]+1e-16)/0.25)
                pwm[pwm<0] = 0
            ppwm = np.zeros((maxlenpwms,4))
            ppwm[(maxlenpwms-lenpwms[si])//2:(maxlenpwms-lenpwms[si])//2+lenpwms[si]] = pwm
            lm.Logo(pd.DataFrame(ppwm, columns = list('ACGT')),
                           ax = axpwm, color_scheme = 'classic')
            axpwm.set_ylim([pwm_min, pwm_max])
            
            axpwm.spines['top'].set_visible(False)
            axpwm.spines['right'].set_visible(False)
            axpwm.spines['left'].set_visible(False)
            axpwm.tick_params(labelleft = False, labelbottom = False, bottom = False)
            if noheatmap and row_distributions is None and yticklabels is not None:
                axpwm.tick_params(labelleft = False, labelright = True, 
                                  labelbottom = False, bottom = False)
                axpwm.set_yticks([(pwm_max+pwm_min)/2])
                axpwm.set_yticklabels(yticklabels[[-s-1]])
    
    # Plot Heatmap
    if not noheatmap:
        if vmin is None:
            vmin = np.amin(heatmat)
        if vmax is None:
            vmax = np.amax(heatmat)
        
        ax.imshow(heatmat, aspect = 'auto', cmap = heatmapcolor, vmin = vmin, 
                  vmax = vmax, origin = 'lower')
        ax.set_yticks(np.arange(len(heatmat)))
        ax.set_xticks(np.arange(len(heatmat[0])))
       
        # add colorbar
        axcol = fig.add_subplot(999)  
        print(vmin, vmax)
        axcol.set_position([0.1+before+wfac+width*0.25/fullw, 
                            0.1+beneath+mfac+height*0.25/fullh, 
                            width*5/fullw, 
                            height*1/fullh])
        axcol.tick_params(bottom = False, labelbottom = False, labeltop = True,
                          top = True, left = False, labelleft = False)
        axcol.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', 
                     cmap = heatmapcolor)
        axcol.set_xticks([0,101])
        
        colormapresolution = 1
        colormapresolution = ['Repressive', 'Activating']
        
        if isinstance(colormapresolution, int):
            axcol.set_xticklabels([round(vmin,colormapresolution), round(vmax,colormapresolution)], rotation = 60)
        elif isinstance(colormapresolution,list):
            axcol.set_xticklabels([colormapresolution[0], colormapresolution[-1]], rotation = 60)
            
        #Add text to heatmap if true
        if plot_value:
            # TODO add fuction to automate to scientific format, use 1 decimal
            # resolution in heatmap, add 10^i to colormap
            if np.amax(np.absolute(heatmat)) > 10:
                heattext = np.array(heatmat, dtype = int)
            elif np.amax(np.absolute(heatmat)) > 1:
                heattext = np.around(heatmat, 1)
            else:
                heattext = np.around(heatmat, 2)
            for c in range(len(heattext[0])):
                for d in range(len(heattext)):
                    ax.text(c,d,str(heattext[d,c]), color = 'k', ha = 'center', fontsize = 6)
        
        
        if grid:
            ax.set_yticks(np.arange(len(heatmat)+1)-0.5, minor = True)
            ax.set_xticks(np.arange(len(heatmat[0])+1)-0.5, minor = True)
            ax.grid(color = 'k', which = 'minor')

        # x_attributes are another heatmap that determines additiona features
        # of the columns
        if x_attributes is not None and not noheatmap:
            # transform the attributes into unique integers
            for x, xunique in enumerate(x_attributes):
                if xunique.dtype != float:
                    xunique = np.unique(xunique)
                    for s, xuni in enumerate(xunique):
                        x_attributes[x, x_attributes[x] == xuni] = s
            
            axatx = fig.add_subplot(717)
            axatx.spines['top'].set_visible(False)
            axatx.spines['bottom'].set_visible(False)
            axatx.spines['right'].set_visible(False)
            axatx.spines['left'].set_visible(False)
            axatx.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False, labelright = False)
            
            axatx.set_position([0.1+before,0.1, wfac, height*(yextra-0.25)/fullh])
            if isinstance(xatt_color,list):
                for xai, xac in enumerate(xatt_color):
                    mask = np.ones(np.shape(x_attributes))
                    mask[xai] = 0
                    mask = mask == 1
                    axatx.imshow(np.ma.masked_array(x_attributes,mask), aspect = 'auto', cmap = xac, vmin = np.amin(x_attributes[xai]), vmax = np.amax(x_attributes[xai]))
            else:
                axatx.imshow(x_attributes, aspect = 'auto', cmap = xatt_color)
            
            axatx.set_xticks(np.arange(len(heatmat[0])))        
            if xattr_name is not None:
                axatx.tick_params(labelright = True)
                axatx.set_yticks(np.arange(np.shape(x_attributes)[0]))
                axatx.set_yticklabels(xattr_name)
                
            # Determine which subplot gets the xlabels
            if xlabel is not None:
                axatx.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False)
                axatx.set_xlabel(xlabel)
            if xticklabels is not None:
                axatx.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
                axatx.set_xticklabels(xticklabels, rotation  = 90)
        
        elif xlabel is not None and not noheatmap:
            ax.set_xlabel(xlabel)
        elif xticklabels is not None and not noheatmap:
            ax.tick_params(which = 'both', bottom = True, labelbottom = True, left = False, labelleft = False)
            ax.set_xticklabels(xticklabels, rotation = 90)
                
        
        if y_attributes is not None:
            # Make y attributes integer if they are not float or int
            for y, yunique in enumerate(y_attributes.T):
                if yunique.dtype != float and yunique.dtype != int:
                    yunique = np.unique(yunique)
                    for s, yuni in enumerate(yunique):
                        y_attributes[y_attributes[:,y] == yuni,y] = s
            
            axaty = fig.add_subplot(177)
            axaty.spines['top'].set_visible(False)
            axaty.spines['bottom'].set_visible(False)
            axaty.spines['right'].set_visible(False)
            axaty.spines['left'].set_visible(False)
            axaty.tick_params(which = 'both', bottom = False, labelbottom = False, left = False, labelleft = False)
            
            axaty.set_position([0.1+before+wfac,0.1+beneath, width*(xextra-0.25)/fullw, mfac])
            if isinstance(yatt_color,list):
                if not (isinstance(yatt_vlim, list) and len(yatt_vlim) == len(yatt_color)):
                    yatt_vlim = [[None, None] for i in range(len(yatt_color))]
                for yai, yac in enumerate(yatt_color):
                    mask = np.ones(np.shape(y_attributes))
                    mask[:,yai] = 0
                    mask = mask == 1
                    axaty.imshow(np.ma.masked_array(y_attributes,mask), aspect = 'auto', cmap = yac, vmin = yatt_vlim[yai][0], vmax =yatt_vlim[yai][1],origin = 'lower')
            else:
                axaty.imshow(y_attributes, aspect = 'auto', cmap = yatt_color, origin = 'lower')
            if yattr_name is not None:
                axaty.tick_params(labeltop = True)
                axaty.set_xticks(np.arange(np.shape(y_attributes)[1]))
                axaty.set_xticklabels(yattr_name, rotation = 270)
            
            # Determine which subplot should have ticklabels
            axaty.set_yticks(np.arange(len(heatmat)))
            if ylabel is not None:
                axaty.tick_params(labelright = True)
                axaty.set_ylabel(ylabel)
            if yticklabels is not None:
                axaty.tick_params(labelright = True, right = True)
                #print('int', yticklabels)
                axaty.set_yticklabels(yticklabels)
        
        elif ylabel is not None and not noheatmap:
            ax.set_ylabel(ylabel)
        elif yticklabels is not None and not noheatmap:
            #print(yticklabels)
            ax.tick_params(which = 'both', bottom = False, labelbottom = True, left = False, labelleft = False, labelright = True)
            ax.set_yticklabels(yticklabels)
    
    # If given, add box or barplot to right side of the plot
    if row_distributions is not None:
        if not isinstance(row_distributions, list) and not isinstance(row_distributions, np.ndarray):
            row_distributions = list(matrix)
        axdy = fig.add_subplot(188)
        axdy.spines['top'].set_visible(False)
        axdy.spines['right'].set_visible(False)
        axdy.tick_params(which = 'both', left = False, labelleft = False, labelright = True, right = True)
        
        if y_attributes is not None:
            dwidth = mfac*np.shape(y_attributes)[1]*cellsize/wfig++mfac*0.25*cellsize/wfig
        else:
            dwidth = 0
        
        axdy.set_position([0.1+before+wfac+width*(xextra-0.25)/fullw, 0.1+beneath, width*(rowdistsize-0.25)/fullw, mfac])
        if sorty is not None:
            yticklabels = yticklabels[np.argsort(sorty)]
        plot_distribution(row_distributions, yticklabels, vert = False, labelside = 'opposite', ax = axdy, sort = sorty, outname = None, **row_distribution_kwargs)
    

    if figname is not None:
        if not noheatmap:
            figname += '_heatmap'
        fig.savefig(figname+fmt, dpi = dpi, bbox_inches = 'tight')
        print( 'SAVED', figname+fmt, dpi)
    else:
        plt.show()
    plt.close()
    return sortx, sorty



def approximate_density(x, bins = 20, sliding_windows = 4, miny=None, maxy = None):
    '''
    Generates density for bins
    '''
    if miny is None:
        miny = np.amin(x)
    if maxy is None:
        maxy = np.amax(x)
    # Determine bin size
    bsize = (maxy-miny)/bins
    
    dens = np.zeros(len(x))
    dcount = np.zeros(len(x))
    for m in range(sliding_windows):
        # create bins
        bins1 = np.linspace(miny-(m+1)*bsize/(sliding_windows+1), maxy-(m+1)*bsize/(sliding_windows+1),bins + 1)
        bins2 = np.linspace(miny+(m+1)*bsize/(sliding_windows+1), maxy+(m+1)*bsize/(sliding_windows+1),bins + 1)
        # determine entries in each bin
        density= np.array([np.sum((x >= bins1[b]) * (x<bins1[b+1])) for b in range(len(bins1)-1)])
        density2= np.array([np.sum((x >= bins2[b]) * (x<bins2[b+1])) for b in range(len(bins2)-1)])
        # scale to max 1
        density = density/np.amax(density)
        density2 = density2/np.amax(density2)
        # assign desities to dens
        for b in range(bins):
            dens[(x >= bins1[b]) * (x<bins1[b+1])] += density[b]
            dens[(x >= bins2[b]) * (x<bins2[b+1])] += density[b]
            dcount[(x >= bins1[b]) * (x<bins1[b+1])] += 1
            dcount[(x >= bins2[b]) * (x<bins2[b+1])] += 1
    # take average over all bins and sliding_windows
    dens = dens/dcount
    
    return dens

def _simple_swarmplot(data, positions, vert = True, unit = 0.4, colormin = None, colormax = None, color = None, cmap = None, connect_swarm = False, scattersort = 'top', scatter_size = None, ax = None):
    '''
    Creates a simple swarmplot with scatter plot with control over all aspects
    in the swarm, such as size, color, connections between distributions
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        if vert:
            fig = plt.figure(figsize = (len(data)*0.4, 3))
        else:
            fig = plt.figure(figsize = (3, len(data)*0.4))
        ax = fig.add_subplot(111)
    
    if colormin is None and not isinstance(color, str):
        colormin = np.amin(color)
    if colormax is None and not isinstance(color, str):
        colormax = np.amax(color)
    
    if connect_swarm and len(np.shape(data)) > 1:
        xposses = []
        randomshift = np.random.random(len(data[0])) # usese the same random 
        # shifts on x for all distributions
    
    for i, set1 in enumerate(data):
        set1 = np.array(set1)
        if scattersort == 'top':
            setsort = np.argsort(set1)
        else:
            setsort = np.argsort(-set1)
        
        if color is None:
            cmap = cm.twilight
            sccolor = np.ones(len(setsort))*0.25
        elif isinstance(color, str):
            sccolor = np.array([color for ci in range(len(setsort))])
        else:
            sccolor = (color-colormin)/(colormax-colormin)
            
        if scatter_size is None:
            scsize = 0.2*np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
        elif isinstance(scatter_size, float) or isinstance(scatter_size, int):
            scsize = scatter_size * np.ones(len(setsort))*plt.rcParams['lines.markersize'] ** 2.
        else:
            scsize = np.sqrt(scatter_size/3.)
            scsize = (((sizemax-sizemin)*(scsize - np.amin(scsize))/(np.amax(scsize) - np.amin(scsize))) + sizemin)
            scsize *= plt.rcParams['lines.markersize'] ** 2.
            
        
        dens = approximate_density(set1)
        if connect_swarm and len(np.shape(data)) > 1:
            randx = positions[i] + dens *width/2 * (randomshift-0.5)
        else:
            randx = positions[i] + dens * width * (np.random.random(len(setsort))-0.5) # + width/2 * simple_beeswarm(set1, nbins = 40) #
        
        if vert: 
            ax.scatter(randx[setsort], set1[setsort], cmap= cmap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
        else:
            ax.scatter(set1[setsort], randx[setsort], cmap= cmap, s = scsize[setsort], c = sccolor[setsort], alpha = scatter_alpha, vmin = 0, vmax = 1, lw = 0, zorder = 5)
        if connect_swarm and len(np.shape(data)) > 1:
            xposses.append(randx)
    
    if connect_swarm and len(np.shape(data)) > 1:
        xposses=np.array(xposses)
        for j, setj in enumerate(np.array(data).T):
            if vert:
                ax.plot(xposses[:,j], setj, color  = 'grey', alpha = 0.5, lw = 0.5)
            else:
                ax.plot(setj, xposses[:,j], color  = 'grey', alpha = 0.5, lw = 0.5)
    if return_fig:
        return fig
    
def _colorbar(cmap, ticklabels = None, vert = True, ratio = 3, tickpositions = None, ax = None):
    '''
    # Generates heatmap for cmap
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        if vert:
            fig = plt.figure(figsize = (1,ratio))
        else:
            fig = plt.figure(figsize = (ratio, 1))
        ax = fig.add_subplot(111)
    if tickpositions is not None:
        if tickpositions == 'left':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = False, top = False, left = True, labelleft = True, right=False, labelright = False)
        if tickpositions == 'right':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = False, top = False, left = False, labelleft = False, right=True, labelright = True)
        if tickpositions == 'bottom':
            ax.tick_params(bottom = True, labelbottom = True, labeltop = False, top = False, left = False, labelleft = False, right=False, labelright = False)
        if tickpositions == 'top':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = True, top = True, left = False, labelleft = False, right=False, labelright = False)
        else:
            ax.tick_params(bottom = False, labelbottom = False, labeltop = False, top = False, left = False, labelleft = False, right=False, labelright = False)
            
    ax.imshow(np.linspace(0,1,101).reshape(1,-1), aspect = 'auto', cmap = cmap)
    if ticklabels is not None:
        ax.set_xticks(np.linspace(0,101, len(ticklabels)))
        if vert:
            ax.set_yticklabels(ticklabels)
        else:
            ax.set_xticklabels(ticklabels, rotation = 90)
    
    if return_fig:
        return fig

def plot_distribution(
    data, # list of lists with values for boxplots, can be list of list of lists 
    # if multiple boxplots for each position are plotted
    modnames, # names of the distributions
    vert = True, # if vertical or horizontal 
    labelside = 'default', # default is bottom for vertical and left for horizontal
    ax = None, # if ax is None a figure object is produced
    sort = None, # order of data points
    split = 1, # splits data int split parts and plots them split boxplots 
    # for each position
    legend_labels = None, # labels for the legend, legend can be produced if 
    # split> 1, for boxplots at the same position with different facecolors
    legend_above = True, # if legend should be placed above the plot
    # TODO: make function to choose legend positions above, below, upper left,
    # upper right
    xwidth = 0.6, # width of every position in figure
    height = 4, # height of figure
    width = 0.8, # width of boxplot
    show_mean = False, 
    showfliers = False, 
    showcaps = True, 
    facecolor = None, # color or list of colors if split > 1
    mediancolor = None, # color or list of colors if split > 1
    grid = True, 
    swarm = False, # adds swarmplot to boxplot
    barplot = False, # create barplot instead of boxplot, either use single
    # value input of function uses mean
    scatter_color = 'grey', 
    scatter_colormap = cm.jet, 
    scatter_alpha = 0.8, 
    scatter_size = 0.5, 
    connect_swarm = False, # can connect the dots in the swarm plot between distributions
    scattersort = 'top', # sorts scatter plot dots by value
    ylim = None, 
    ylabel = None, 
    sizemax = 2, # max size for scatters
    sizemin = 0.25, # min size for scatters
    colormin = None, 
    colormax = None, 
    dpi = 200, 
    savedpi = 200, 
    outname = None, # Name of figure, if given, figure will be saved
    fmt = 'jpg'):
    
    if sort is not None:
        positions = np.argsort(sort)
        modnames = np.array(modnames)[sort]
    else:
        positions = np.arange(len(data))
    
    fcolor = None # color for every boxplot at every position derived from
    # facecolors
    # Adjust parameters to split option
    if split > 1: # split boxplots will be plotted at each position
        if len(data) == split: # if data was given as list of lists for each split
            data = [m for ma in data for m in ma]
        
        if width * split >1: # adjust width of individual boxplots
            width = width/split
        
        # determine new positions of boxplots
        positions = []
        for s in range(split):
            if sort is None:
                positions.append(np.arange(int(len(data)/split)) + width*s - (split*width/2) + width/2)
            else:
                positions.append(np.argsort(sort) + width*s - (split*width/2) + width/2)
        positions = np.concatenate(positions)
        
        # create array with colors for each boxplot
        if isinstance(facecolor, list):
            if len(facecolor) == split:
                fcolor = [facecolor[c] for c in range(split) for j in range(int(len(data)/split))]
            else:
                fcolor = [facecolor[c] for c in range(len(data))]
        
        # same for median color
        if mediancolor is not None:
            if isinstance(mediancolor, list):
                if len(mediancolor) == split:
                    mediancolor = [mediancolor[c] for c in range(split) for j in range(int(len(data)/split))]
                else:
                    mediancolor = [mediancolor[c] for c in range(len(data))]
            else:
                mediancolor = [mediancolor for c in range(len(data))]
            
    # if median color is not None, need to replicate into list
    if mediancolor is not None:
        if not isinstance(mediancolor, list):
            mediancolor = [mediancolor for mc in range(len(data))]
    
    return_ax = False # if ax given: function returns manipulated subplot
    if ax is None:
        if vert:
            fig = plt.figure(figsize = (len(modnames)*xwidth, height), dpi = dpi)
        else:
            fig = plt.figure(figsize = (height, len(modnames)*xwidth), dpi = dpi)
        ax = fig.add_subplot(111)
        ax.set_position([0.1,0.1,0.8,0.8])
    else: 
        return_ax = True
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ylabel is not None:
        if vert:
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(ylabel)
        
    data = list(data)
    
    if swarm: # add fancy swarmplot with different option on top of boxplot
        _simple_swarmplot(data, positions, vert = vert, colormin = colormin, colormax = colormat, color = scatter_color, cmap = scatter_colormap, connect_swarm = connect_swarm, scattersort = scattersort, scatter_size = scatter_size, ax = ax)
            
        # generate colorbar
        if ((scatter_color is not None) and (not isinstance(scatter_color, str))):
            axcol = fig.add_subplot(911)
            axcol.set_position([0.6,0.925,0.3, 0.05])
            _colorbar(scatter_colormap, ticklabels = [round(colormin,1), round((colormin+colormax)/2,1), round(colormax,1)] , tickpositions = 'top', ax = axcol)
            
    
    if barplot:
        if fcolor is not None:
            barcolor = fcolor
        else:
            barcolor = 'grey'
        if vert:
            bplot = ax.bar(positions, np.mean(data,axis = 1), width = width*0.9, color = barcolor, linewidth = 1)
        else:
            if len(np.shape(data)) > 1:
                data = np.mean(data, axis = 1)
            bplot = ax.barh(positions, data, height = width*0.9, color = barcolor, linewidth = 1)
            ax.set_ylim([np.amin(positions)-0.5, np.amax(positions)+0.5])
        
        # create a legend()
        if isinstance(facecolor, list) and legend_labels is not None:
            handles = []
            for f, fcol in enumerate(facecolor):
                patch = mpatches.Patch(color=fcol, label=legend_labels[f])
                handles.append(patch)
            ax.legend(handles = handles)
    else:
        if facecolor is None or fcolor is not None:
            boxplotcolor = (0,0,0,0)
        else:
            boxplotcolor = facelolor
        
        bplot = ax.boxplot(data, positions = positions, vert = vert, showcaps=showcaps, patch_artist = True, boxprops={'facecolor':boxplotcolor}, showfliers=showfliers, whiskerprops={'linewidth':1}, widths = width,zorder = 4)
    
        if fcolor is not None:
            for patch, color in zip(bplot['boxes'], fcolor):
                patch.set_facecolor(color)
                fc = patch.get_facecolor()
                patch.set_facecolor(mpl.colors.to_rgba(fc, 0.7))
            # create legend()
            if isinstance(facecolor, list) and legend_labels is not None:
                handles = []
                for f, fcol in enumerate(facecolor):
                    patch = mpatches.Patch(color=fcol, label=legend_labels[f])
                    handles.append(patch)
                if legend_above:
                    ax.legend(handles = handles,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=3)
                else:
                    ax.legend(handles = handles)
                
                
        if mediancolor is not None:
            for mx, median in enumerate(bplot['medians']):
                median.set_color(mediancolor[mx])
    
    if ylim is not None:
        if vert:
            ax.set_ylim(ylim)
        else:
            ax.set_xlim(ylim)
    
    if show_mean:
        if vert:
            ax.plot(np.sort(positions), [np.mean(data[s]) for s in np.argsort(positions)], color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
        else:
            ax.plot([np.mean(data[s]) for s in np.argsort(positions)],np.sort(positions), color = 'r', marker = 's', zorder = 6, markersize = 3, ls = '--', lw = 0.5)
    
    if vert:
        if labelside =='opposite':
            ax.tick_params(bottom = False, labelbottom = False, labeltop = True)
        ax.set_xticks(np.arange(len(modnames)))
        ax.set_xticklabels(modnames, rotation = 90)
        if grid:
            ax.grid(axis = 'y')
    else:
        if labelside =='opposite':
            ax.tick_params(left = False, labelleft = False, labelright = True)
        ax.set_yticks(np.arange(len(modnames)))
        ax.set_yticklabels(modnames)
        if grid:
            ax.grid(axis = 'x')
    
    if return_ax:
        return ax
    
    if outname is None:
        #fig.tight_layout()
        plt.show()
    else:
        fig.savefig(outname+'_distribution.'+fmt, dpi = savedpi, bbox_inches = 'tight')


def piechart(percentages, labels = None, colors = None, cmap = 'tab10', cmap_range=[0,1], explode_size = None, explode_indices = None, labels_on_side = False, explode_color = None, ax = None):
    '''
    Plots piechart with some options
    '''
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize = (3.,3.), dpi = 200)
        ax = fig.add_subplot(111)
    
    if labels is None:
        labels = np.arange(len(percentages)).astype(str)
        
    if colors is None:
        colors = plt.get_cmap(cmap)(np.linspace(cmap_range[0],cmap_range[1], len(percentages)))
        
    explode = None
    if explode_indices:
        explode = np.zeros(len(percentages))
        # Have Outside entries stick out
        if explode_size is None:
            explode_size = 0.1
        explode[explode_indices] = explode_size
        if explode_color is not None:
            if isinstance(explode_color, str) and not isinstance(colors[0], str):
                explode_color = mpl.colors.to_rgba(explode_color)
            colors[explode_indices] = explode_color
    
    if labels_on_side:
        wedges, texts = ax.pie(percentages, colors = colors, explode = explode) 
        bbox_props = dict(boxstyle="square,pad=0.", fc="w", ec=None, lw=0.)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)
    else:
        ax.pie(percentages, labels=labels, colors = colors, explode = explode)
    if return_fig:
        return fig



