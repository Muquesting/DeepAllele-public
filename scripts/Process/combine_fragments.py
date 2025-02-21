# Combine fragments of different files
import numpy as np
import pandas as pd
import argparse

def add_frid(fpath_df):
    """ Add fragment id as new column and index
    """
    fpath_df['chr'] = fpath_df['chr'].astype(str)
    fpath_df['start'] = fpath_df['start'].astype(str)
    fpath_df['stop'] = fpath_df['stop'].astype(str)
    fpath_df['cellbc'] = fpath_df['cellbc'].astype(str)

    frid = fpath_df['chr'].values + '-' + fpath_df['start'].values + '-' + fpath_df['stop'].values + '-' + fpath_df['cellbc'].values

    fpath_df.loc[:,'frid']=frid

    fpath_df.set_index('frid',inplace=True,drop=False)

    return(fpath_df)

def combine_fragments(fpath1,fpath2,outpath):
    """ Combines two fragments files together, combining any common fragments
    fpath1: path to fragment file 1 (.bed, .bed.gz)
    fpath2: path to fragment file 2 (.bed, .bed.gz)
    outpath: path to output combined fragment file (.bed)
    """

    print('Reading in fragments files')
    print('Fragment file 1 = {}'.format(fpath1))
    print('Fragment file 2 = {}'.format(fpath2))

    # read in fragments
    fpath_df1 = pd.read_csv(fpath1,sep='\t',index_col=None,header=None)
    fpath_df1.columns = ['chr','start','stop','cellbc','read_ct']

    fpath_df2 = pd.read_csv(fpath2,sep='\t',index_col=None,header=None)
    fpath_df2.columns = ['chr','start','stop','cellbc','read_ct']

    len1=fpath_df1.shape[0]
    len2=fpath_df2.shape[0]

    print('Getting fragment indices')

    fpath_df1 = add_frid(fpath_df1)
    fpath_df2 = add_frid(fpath_df2)

    print('Combining common fragments')
    # get common fragments
    frid1 = fpath_df1.frid.to_numpy()
    frid2 = fpath_df2.frid.to_numpy()
    matching_frid = np.intersect1d(frid1,frid2)
    matching_frid = matching_frid.tolist()

    print('{} common fragments found'.format(len(matching_frid)))

    fpath_df1_matching = fpath_df1.loc[matching_frid,:]
    fpath_df2_matching = fpath_df2.loc[matching_frid,:]

    # combine common
    fpath_df_matching_combo = fpath_df1_matching
    fpath_df_matching_combo.loc[:,'read_ct'] = fpath_df1_matching.loc[:,'read_ct'] + fpath_df2_matching.loc[:,'read_ct']

    # combine all
    fpath_df1.drop(matching_frid, inplace=True)
    fpath_df2.drop(matching_frid, inplace=True)

    # concatenate
    final_df = pd.concat([fpath_df1, fpath_df2,fpath_df_matching_combo],sort=False)

    print('Final number of rows = {}'.format(final_df.shape[0]))
    expected_num = len1+len2-len(matching_frid)
    print('Expected number of rows = {}'.format(expected_num))
    # keep only required columns
    final_df=final_df.iloc[:,0:5]

    print('Writing combined fragments to {}'.format(outpath))
    # write to file
    final_df.to_csv(path_or_buf=outpath,sep='\t',index=False,header=False)

    print('All Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--file1", help="Path to fragments file 1",type=str)
    parser.add_argument("-B", "--file2", help="Path to fragments file 2",type=str)
    parser.add_argument("-O", "--outpath", help="Path to output fragments file",type=str)
    argt = parser.parse_args()

    combine_fragments(argt.file1,argt.file2,argt.outpath)