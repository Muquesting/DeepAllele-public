#!/usr/bin/env python3
"""
ChIP-seq preprocessing script for DeepAllele project
Transforms ChIP-seq data into a structured HDF5 file format for model training
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from DeepAllele import model, data, tools
from DeepAllele.tools import pearson_r
from DeepAllele.data import DF2array

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ChIP-seq data')
    parser.add_argument('--sequence-path', required=True,
                        help='Base path for sequence FASTA files')
    parser.add_argument('--c57-pwk-chip', required=True,
                        help='Path to C57-PWK ChIP data file')
    parser.add_argument('--c57-spret-chip', required=True,
                        help='Path to C57-SPRET ChIP data file')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save processed files')
    parser.add_argument('--seq-length', type=int, default=551,
                        help='Length for one-hot encoding sequences')
    return parser.parse_args()

def process_pwk_data(args):
    """Process PWK/B6 ChIP-seq data"""
    print("Processing PWK/B6 ChIP-seq data...")
    
    # Load ChIP-seq data
    df_C57_PWK = pd.read_csv(args.c57_pwk_chip, sep='\t')
    df_C57_PWK['peak_name'] = df_C57_PWK['ID']
    
    # Load sequence data
    B6_sequence_DF = data.load_fasta(args.sequence_path + 'b6seqs.fa')
    B6_sequence_DF.rename(columns={"sequences": "B6_sequence"}, inplace=True)
    B6_sequence_DF['peak_name'] = B6_sequence_DF['peak_name'].apply(lambda x: x.split('_')[0])
    print(f"B6 unique peaks: {B6_sequence_DF['peak_name'].nunique()}")
    
    PWK_sequence_DF = data.load_fasta(args.sequence_path + 'PWKseqs.fa')
    PWK_sequence_DF.rename(columns={"sequences": "PWK_sequence"}, inplace=True)
    PWK_sequence_DF['peak_name'] = PWK_sequence_DF['peak_name'].apply(lambda x: x.split('_')[0])
    
    # Merge data
    peak_data = df_C57_PWK[['peak_name', 'F1_FPC_PU1_C57', 'F1_FPC_PU1_PWK']]
    peak_data = peak_data.merge(PWK_sequence_DF, on="peak_name", how='inner')
    peak_data = peak_data.merge(B6_sequence_DF, on="peak_name")
    
    print(f"Final dataset has {peak_data['peak_name'].nunique()} peaks")
    
    # Process data
    cast_sequence_np = DF2array(peak_data, "PWK_sequence", length=args.seq_length)
    B6_sequence_np = DF2array(peak_data, "B6_sequence", length=args.seq_length)
    B6_counts = np.log(peak_data['F1_FPC_PU1_C57']+1)
    PWK_counts = np.log(peak_data['F1_FPC_PU1_PWK']+1)
    ratio = B6_counts - PWK_counts
    
    # Save to HDF5
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "sequence_datasets_chip_PWK_B6.hdf5")
    
    f = h5py.File(output_file, "w")
    f.attrs['data_type'] = 'ChIP-seq PWK/B6'
    f.attrs['sequence_length'] = args.seq_length
    
    f["Cast_sequence"] = cast_sequence_np
    f["B6_sequence"] = B6_sequence_np
    f["Cast_counts"] = PWK_counts
    f["B6_counts"] = B6_counts
    f["ratio"] = ratio
    
    # Convert peak names to string array suitable for HDF5
    str_dtype = h5py.string_dtype(encoding='utf-8') 
    f['peak_name'] = np.array(peak_data['peak_name'].astype(str), dtype=str_dtype)
    
    f.close()
    print(f"PWK/B6 data saved to: {output_file}")
    return output_file

def process_spret_data(args):
    """Process SPRET/B6 ChIP-seq data"""
    print("Processing SPRET/B6 ChIP-seq data...")
    
    # Load ChIP-seq data
    df_C57_SPRET = pd.read_csv(args.c57_spret_chip, sep='\t')
    df_C57_SPRET['peak_name'] = df_C57_SPRET['ID']
    
    # Load sequence data
    B6_sequence_DF = data.load_fasta(args.sequence_path + 'b6seqs.fa')
    B6_sequence_DF.rename(columns={"sequences": "B6_sequence"}, inplace=True)
    B6_sequence_DF['peak_name'] = B6_sequence_DF['peak_name'].apply(lambda x: x.split('_')[0])
    
    SPRET_sequence_DF = data.load_fasta(args.sequence_path + 'SPRETseqs.fa')
    SPRET_sequence_DF.rename(columns={"sequences": "SPRET_sequence"}, inplace=True)
    SPRET_sequence_DF['peak_name'] = SPRET_sequence_DF['peak_name'].apply(lambda x: x.split('_')[0])
    
    # Merge data
    peak_data = df_C57_SPRET[['peak_name', 'F1_FSC_PU1_C57', 'F1_FSC_PU1_SPRET']]
    peak_data = peak_data.merge(SPRET_sequence_DF, on="peak_name")
    peak_data = peak_data.merge(B6_sequence_DF, on="peak_name")
    
    print(f"Final dataset has {peak_data['peak_name'].nunique()} peaks")
    
    # Process data
    cast_sequence_np = DF2array(peak_data, "SPRET_sequence", length=args.seq_length)
    B6_sequence_np = DF2array(peak_data, "B6_sequence", length=args.seq_length)
    B6_counts = np.log(peak_data['F1_FSC_PU1_C57']+1)
    SPRET_counts = np.log(peak_data['F1_FSC_PU1_SPRET']+1)
    ratio = B6_counts - SPRET_counts
    
    # Save to HDF5
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "sequence_datasets_chip_SPRET_B6.hdf5")
    
    f = h5py.File(output_file, "w")
    f.attrs['data_type'] = 'ChIP-seq SPRET/B6'
    f.attrs['sequence_length'] = args.seq_length
    
    f["Cast_sequence"] = cast_sequence_np
    f["B6_sequence"] = B6_sequence_np
    f["Cast_counts"] = SPRET_counts
    f["B6_counts"] = B6_counts
    f["ratio"] = ratio
    
    # Convert peak names to string array suitable for HDF5
    str_dtype = h5py.string_dtype(encoding='utf-8') 
    f['peak_name'] = np.array(peak_data['peak_name'].astype(str), dtype=str_dtype)
    
    f.close()
    print(f"SPRET/B6 data saved to: {output_file}")
    return output_file

def main():
    args = parse_args()
    
    print(f"ChIP-seq preprocessing with sequence length: {args.seq_length}")
    print(f"Using sequence path: {args.sequence_path}")
    print(f"Using C57-PWK ChIP file: {args.c57_pwk_chip}")
    print(f"Using C57-SPRET ChIP file: {args.c57_spret_chip}")
    
    # Process PWK data
    pwk_output = process_pwk_data(args)
    
    # Process SPRET data
    spret_output = process_spret_data(args)
    
    print("ChIP-seq preprocessing completed successfully!")
    print(f"Output files: \n- {pwk_output}\n- {spret_output}")

if __name__ == "__main__":
    main()


