# Split fragments from single cell bam files form both genomes
import numpy as np
import pandas as pd
import argparse

def split_fragments(fpath):
    '''Splits scATAC fragments file into 2
    For each fragment, samples using binomial distribution, p = 0.5 to split reads into one of two files
    Writes new fragments files after splitting

    fpath = path to fragment file to split

    '''
    print('Reading fragments file from {}'.format(fpath))
    fpath_df = pd.read_csv(fpath,sep='\t',index_col=None,header=None)
    fpath_df.columns = ['chr','start','stop','cellbc','read_ct']
    fpath_df_read_ct = fpath_df['read_ct'].to_numpy()

    print('splitting fragments file')
    fpath_df_read_ct_bin = np.random.binomial(fpath_df_read_ct,0.5)
    fpath_df['split1']= fpath_df_read_ct_bin
    fpath_df['split2']= fpath_df['read_ct']-fpath_df['split1']
    fpath_df_split1 = fpath_df.loc[:,['chr','start','stop','cellbc','split1']]
    fpath_df_split2 = fpath_df.loc[:,['chr','start','stop','cellbc','split2']]
    fpath_df_split1 = fpath_df_split1[fpath_df_split1.split1 != 0]
    fpath_df_split2 = fpath_df_split2[fpath_df_split2.split2 != 0]
    split1_path = fpath.split('bed.gz')[0]+'split1.bed'
    split2_path = fpath.split('bed.gz')[0]+'split2.bed'

    print('checking to make sure that counts match, should be 0: {}',
          format(fpath_df['read_ct'].sum()-fpath_df_split1['split1'].sum()-fpath_df_split2['split2'].sum()))

    print('checking to make sure that prob correct, should be 0.5: {}',
      format(fpath_df_split1['split1'].sum()/fpath_df['read_ct'].sum()))

    print('writing new split fragment files')

    fpath_df_split1.to_csv(path_or_buf=split1_path,sep='\t',index=False,header=False)

    fpath_df_split2.to_csv(path_or_buf=split2_path,sep='\t',index=False,header=False)

    print('All Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--filename", help="Path to fragments file",type=str)
    argt = parser.parse_args()

    split_fragments(argt.filename)