# Insert variants and generate fasta file
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import Bio
from Bio import SeqIO
from Bio import Seq
from copy import deepcopy
from tqdm import tqdm
from Bio.SeqRecord import SeqRecord
import argparse

def load_fasta(fasta_path):
    """ Load in fasta file to SeqIO dictionary
    Args:
    fasta_path: path to input fasta to modify
    """

    record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    record_dict_mod = deepcopy(record_dict)

    return record_dict,record_dict_mod

def snp_load_process(snp_intersect_file,filter_pass=False):
    """ Load in bedtools intersect output of snp vcf and peak bed file
    Args:
    snp_intersect_file: path to intersect tab-delimited file
    filter_pass: (bool) if True, then only includes those variants with filter set to PASS
    """

    snp_int = pd.read_csv(snp_intersect_file,sep='\t',header=None)

    if filter_pass:
        print('filtering to only variants with PASS quality')
        snp_int_pass = snp_int[snp_int.loc[:,6]=='PASS']
    else:
        print('Including ALL variants')
        snp_int_pass = snp_int

    snp_int_pass = snp_int_pass.iloc[:,[0,1,2,3,4,10,11,12,13,14]]
    snp_int_pass.columns = ['variant_chr','variant_pos','variant_ID','REF',
                        'ALT','OCR_chr','OCR_start','OCR_end','OCR_ID','overlap']

    # in case of multiple posssible alt alleles, take the first
    snp_int_pass.loc[:,'ALT'] = snp_int_pass.loc[:,'ALT'].apply(lambda x: x.split(',')[0])
    snp_int_pass.loc[:,'REF'] = snp_int_pass.loc[:,'REF'].apply(lambda x: x.split(',')[0])

    # convert the variant position from absolute 1-index to 0-index relative to OCR start
    snp_int_pass.loc[:,'variant_pos_relative'] = snp_int_pass.loc[:,'variant_pos']-snp_int_pass.loc[:,'OCR_start']-1


    snp_int_pass = snp_int_pass.loc[snp_int_pass['variant_pos_relative']>0,:]

    print('Number of variants = {}'.format(snp_int_pass.shape[0]))

    return(snp_int_pass)

def indel_load_process(indel_intersect_file,filter_pass=False):
    """ Load in bedtools intersect output of indel vcf and peak bed file
    Args:
    indel_intersect_file: path to intersect tab-delimited file
    filter_pass: (bool) if True, then only includes those variants with filter set to PASS
    """
    indel_int = pd.read_csv(indel_intersect_file,sep='\t',header=None)

    if filter_pass:
        print('filtering to only variants with PASS quality')
        indel_int_pass = indel_int[indel_int.loc[:,6]=='PASS']
    else:
        print('Including ALL variants')
        indel_int_pass = indel_int

    indel_int_pass = indel_int_pass.iloc[:,[0,1,2,3,4,10,11,12,13,14]]
    indel_int_pass.columns = ['variant_chr','variant_pos','variant_ID','REF',
                        'ALT','OCR_chr','OCR_start','OCR_end','OCR_ID','overlap']

    # in case of multiple posssible alt alleles, take the first
    indel_int_pass.loc[:,'ALT'] = indel_int_pass['ALT'].apply(lambda x: x.split(',')[0])
    indel_int_pass.loc[:,'REF'] = indel_int_pass['REF'].apply(lambda x: x.split(',')[0])

    # convert the variant position from absolute 1-index to 0-index relative to OCR start
    indel_int_pass.loc[:,'variant_pos_relative'] = indel_int_pass.loc[:,'variant_pos']-indel_int_pass.loc[:,'OCR_start']-1
    indel_int_pass.loc[:,'REF_LEN'] = indel_int_pass.loc[:,'REF'].apply(len)
    indel_int_pass.loc[:,'ALT_LEN'] = indel_int_pass.loc[:,'ALT'].apply(len)

    ref_len = indel_int_pass.loc[:,'REF_LEN'].values
    alt_len = indel_int_pass.loc[:,'ALT_LEN'].values
    indel = ['INS' if ref_len[i]<alt_len[i] else 'DEL' for i in np.arange(len(ref_len))]
    indel_int_pass.loc[:,'indel'] = indel

    indel_int_pass.loc[:,'var_rel_end'] = indel_int_pass.loc[:,'variant_pos_relative']+indel_int_pass.loc[:,'REF_LEN']-1

    indel_int_pass = indel_int_pass.loc[indel_int_pass['variant_pos_relative']>0,:]

    print('Number of variants = {}'.format(indel_int_pass.shape[0]))

    return(indel_int_pass)

def modify_snps(snp_int_pass,record_dict_mod):
    """ Modify sequences in dictionary with snps
    Args:
    snp_int_pass: processed snp's output
    record_dict_mod: processed dictionary output

    Run this before running the indel pipeline
    """
    snp_int_pass.index = np.arange(snp_int_pass.shape[0])
    print('modifying snps')

    ## first do snps
    # replaces position with alternative allele
    for j in tqdm(np.arange(snp_int_pass.shape[0])):
        peak_id = snp_int_pass.loc[j,'OCR_ID']
        new_seq = list(record_dict_mod[peak_id])
        new_seq[snp_int_pass.loc[j,'variant_pos_relative']] = snp_int_pass.loc[j,'ALT']
        new_seq = ''.join(new_seq)
        record_dict_mod[peak_id]=Seq.Seq(new_seq)

    return(record_dict_mod)

def modify_indels(indel_int_pass,record_dict_mod):
    """ Modify sequences in dictionary with indels
    Args:
    indel_int_pass: processed indels output
    record_dict_mod: processed dictionary output

    Run this after running the snp pipeline
    """
    # first create dictionary to hold position ids
    print('creating position dictionary')
    posnDict = {}
    for key, val in tqdm(record_dict_mod.items()):
        posnDict.update(dict.fromkeys([str(key)], np.arange(len(val))))

    indel_int_pass.index = np.arange(indel_int_pass.shape[0])
    print('modifying indels')

    for j in tqdm(np.arange(indel_int_pass.shape[0])):
        peak_id = indel_int_pass.loc[j,'OCR_ID']
        new_seq = list(record_dict_mod[peak_id])
        new_posn = posnDict[peak_id]

        # for insertions, adds the additional bases (first one is same as reference)
        # adds corresponding -10 at those positions in position index for tracking later
        indel_type = indel_int_pass.loc[j,'indel']
        if indel_type == 'INS':
            alt_posn = indel_int_pass.loc[j,'variant_pos_relative']
            if np.sum(new_posn==alt_posn) > 0:
                var_posn_og = np.where(new_posn==alt_posn)[0][0]+1
                alt_variant = indel_int_pass.loc[j,'ALT']
                alt_variant = list(alt_variant)
                alt_variant = alt_variant[1:]
                new_seq[var_posn_og:var_posn_og] = alt_variant
                new_posn = list(new_posn)
                new_posn[var_posn_og:var_posn_og] = list(-10*np.ones(len(alt_variant)))
                new_posn=np.array(new_posn)
                # assign modified
                posnDict[peak_id] = new_posn
                record_dict_mod[peak_id]=Seq.Seq(''.join(new_seq))

        # for deletions where replacemetn is only 1 base, just deletes from sequence and position
        # for deletions where replacement is > 1 base, first does insertion of additional bases and then deletion of the reference
        if indel_type == 'DEL':
            if indel_int_pass.loc[j,'ALT_LEN'] > 1:
                alt_posn = indel_int_pass.loc[j,'variant_pos_relative']
                if np.sum(new_posn==alt_posn) > 0:
                    var_posn_og = np.where(new_posn==alt_posn)[0][0]
                    new_seq[var_posn_og] = indel_int_pass.loc[j,'ALT'][0]

                    # insert additional characters
                    var_posn_og = var_posn_og+1

                    alt_variant = list(indel_int_pass.loc[j,'ALT'][1:])
                    new_seq[var_posn_og:var_posn_og] = alt_variant
                    new_posn = list(new_posn)
                    new_posn[var_posn_og:var_posn_og] = list(-10*np.ones(len(alt_variant)))

                    # define deletion positions
                    start = indel_int_pass.loc[j,'variant_pos_relative']
                    stop = indel_int_pass.loc[j,'var_rel_end']
                    del_posn = np.arange(start+1,stop+1)

                    # find these in the index dictionary
                    del_posn_indices = np.isin(new_posn,del_posn)
                    # remove in posn and sequence
                    new_posn = [new_posn[x] for x in np.arange(len(new_posn)) if not del_posn_indices[x]]
                    new_seq = [new_seq[x] for x in np.arange(len(new_seq)) if not del_posn_indices[x]]

                    # assign modified
                    new_posn=np.array(new_posn)
                    posnDict[peak_id] = new_posn
                    record_dict_mod[peak_id]=Seq.Seq(''.join(new_seq))

            else:
                alt_posn = indel_int_pass.loc[j,'variant_pos_relative']
                if np.sum(new_posn==alt_posn) > 0:
                    var_posn_og = np.where(new_posn==alt_posn)[0][0]
                    new_seq[var_posn_og] = indel_int_pass.loc[j,'ALT'][0]

                    # define deletion positions
                    start = indel_int_pass.loc[j,'variant_pos_relative']
                    stop = indel_int_pass.loc[j,'var_rel_end']
                    del_posn = np.arange(start+1,stop+1)

                    # find these in the index dictionary
                    del_posn_indices = np.isin(new_posn,del_posn)
                    # remove in posn and sequence
                    new_posn = [new_posn[x] for x in np.arange(len(new_posn)) if not del_posn_indices[x]]
                    new_seq = [new_seq[x] for x in np.arange(len(new_seq)) if not del_posn_indices[x]]

                    # assign modified
                    new_posn=np.array(new_posn)
                    posnDict[peak_id] = new_posn
                    record_dict_mod[peak_id]=Seq.Seq(''.join(new_seq))


    return record_dict_mod

def write_fasta(fn,record_dict_mod_snp_indel):
    """ Write processed fasta to file
    Args:
    fn: output file name for fasta
    record_dict_mod_snp_indel: processed dictionary output (post snp and indel pipeline)
    """
    with open(fn, "w") as outfile:
        for key in tqdm(record_dict_mod_snp_indel.keys()):
            if type(record_dict_mod_snp_indel[key])==Bio.SeqRecord.SeqRecord:
                sequence_write = str(record_dict_mod_snp_indel[key].seq)
            else:
                sequence_write = str(record_dict_mod_snp_indel[key])

            seq_id = str(key)
            outfile.write('>{}\n'.format(seq_id))
            outfile.write('{}\n'.format(sequence_write))

    print('done writing to {}!'.format(fn))

if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", help="Path to input fasta")
    parser.add_argument("-s", "--snp", help="Path to intersected snp file")
    parser.add_argument("-d", "--indel", help="Path to intersected indel file")
    parser.add_argument("-o", "--out", help="Path to output processed fasta file")
    parser.add_argument('--filter_pass', action='store_true', help="If included, only processes variants passing all filters")

    args = parser.parse_args()

    # usage: python insert_variants.py -f [fasta_file] -s [snp_file] -d [indel_file] -o [output_fasta_path] [optional --filter_pass]

    print('loading fasta')
    record_dict,record_dict_mod_pre = load_fasta(args.fasta)

    print('processing SNPs')
    snp_int_pass = snp_load_process(args.snp,filter_pass=args.filter_pass)
    print('processing indels')
    indel_int_pass = indel_load_process(args.indel,filter_pass=args.filter_pass)

    print('Start modifying sequences')
    record_dict_mod_snp = modify_snps(snp_int_pass,record_dict_mod_pre)
    record_dict_mod_snp_indel = modify_indels(indel_int_pass,record_dict_mod_snp)

    print('Writing output fasta')
    write_fasta(args.out,record_dict_mod_snp_indel)