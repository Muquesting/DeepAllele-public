#!/usr/bin/env python3
"""
ATAC-seq preprocessing script for DeepAllele project
Transforms ATAC-seq data into a structured HDF5 file format for model training
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import scanpy as sc

from DeepAllele import tools, data
from DeepAllele.tools import pearson_r

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ATAC-seq data')
    parser.add_argument('--input-csv', required=True,
                        help='Path to input CSV file with ATAC-seq data')
    parser.add_argument('--data-path', required=True,
                        help='Base path to data directory')
    parser.add_argument('--b6-seq-path', required=True,
                        help='Path to B6 sequences FASTA file')
    parser.add_argument('--cast-seq-path', required=True,
                        help='Path to CAST/shifted sequences FASTA file')
    parser.add_argument('--sc-data-path', 
                        help='Path to single-cell data directory (optional)')
    parser.add_argument('--output', required=True,
                        help='Path to output HDF5 file')
    parser.add_argument('--peak-info',
                        help='Path to peak info file (optional)')
    return parser.parse_args()

def load_and_process_bulk_data(args):
    """Load and process the bulk ATAC-seq data"""
    print("Loading bulk ATAC-seq data...")
    
    # Check if file exists and is readable
    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
        
    # Load data
    print(f"Reading file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Rename the first column to peak_name if needed
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'peak_name'}, inplace=True)
    
    print(f"Loaded data with {len(df)} peaks and {len(df.columns)} columns")
    
    # Process the data for B6 and CAST strains
    print("Computing summary columns...")
    
    # Get counts for B6 and CAST
    df['sum.B6'] = df[df.columns[df.columns.str.contains('B6')]].sum(axis=1)
    df['sum.CAST'] = df[df.columns[df.columns.str.contains('CAST')]].sum(axis=1)
    
    # Get counts for each replicate
    for i in range(1, 5):
        df[f'rep{i}.B6'] = df[df.columns[df.columns.str.contains(f'rep{i}.B6')]].sum(axis=1)
        df[f'rep{i}.CAST'] = df[df.columns[df.columns.str.contains(f'rep{i}.CAST')]].sum(axis=1)
    
    # Process cell types
    cell_type_list = ['Treg', 'Tcon', 'resting_Treg', 'resting_Tcon', 'activated_Treg', 'activated_Tcon']
    
    for cell_type in cell_type_list:
        b6_columns = [col for col in df.columns if cell_type in col and 'B6' in col]
        cast_columns = [col for col in df.columns if cell_type in col and 'CAST' in col]
        
        if b6_columns:
            df[f'{cell_type}.B6'] = df[b6_columns].sum(axis=1)
        if cast_columns:
            df[f'{cell_type}.CAST'] = df[cast_columns].sum(axis=1)
    
    # Calculate log ratios
    columns_name = df.columns.tolist()
    columns_name_B6 = [x for x in columns_name if 'B6' in x]
    
    for batch_name in columns_name_B6:
        name = batch_name.split('.')[0]
        df[f'{name}.ratio'] = np.log((1+df[f'{name}.B6'].astype(float)) / (1+df[f'{name}.CAST'].astype(float)))
    
    return df

def load_sequences_and_merge(args, df):
    """Load sequence data from FASTA files and merge with peak data"""
    print("Loading sequence data...")
    
    # Load sequence data as shown in the notebook cells
    b6_sequence_df = data.load_fasta(args.b6_seq_path)
    b6_sequence_df.rename(columns={"sequences": "B6_sequence"}, inplace=True)
    
    cast_sequence_df = data.load_fasta(args.cast_seq_path)
    cast_sequence_df.rename(columns={"sequences": "cast_sequence"}, inplace=True)
    
    print(f"Loaded {len(b6_sequence_df)} B6 sequences and {len(cast_sequence_df)} CAST sequences")
    
    # Create a numeric index column if it doesn't exist
    if 'peak_name' not in df.columns:
        df['peak_name'] = df.index
    
    # Ensure peak_name in all dataframes is string type for consistent merging
    b6_sequence_df['peak_name'] = b6_sequence_df['peak_name'].astype(str)
    cast_sequence_df['peak_name'] = cast_sequence_df['peak_name'].astype(str)
    df['peak_name'] = df['peak_name'].astype(str)
    
    # Merge with peak data
    merged_df = df.merge(cast_sequence_df, on="peak_name", how="inner")
    merged_df = merged_df.merge(b6_sequence_df, on="peak_name", how="inner")
    
    print(f"After merging: {len(merged_df)} peaks with sequence data")
    
    return merged_df

def load_single_cell_data(args, peak_data):
    """Load and merge single-cell ATAC data if available"""
    if not args.sc_data_path:
        print("Skipping single-cell data (path not provided)")
        return peak_data
    
    try:
        print("Loading single-cell ATAC data...")
        
        # Load single-cell data as shown in the notebook
        b6_ad = sc.read_h5ad(os.path.join(args.sc_data_path, "Treg/B6_Treg.h5ad"))
        cast_ad = sc.read_h5ad(os.path.join(args.sc_data_path, "Treg/cast_Treg.h5ad"))
        
        # Process single-cell data
        total_counts_b6 = np.array(b6_ad.X.sum(axis=0))
        total_counts_cast = np.array(cast_ad.X.sum(axis=0))
        
        total_ratio = np.log((1+total_counts_b6) / (1+total_counts_cast))
        b6_ad.var["sc.ratio"] = total_ratio[0]
        
        treg_peak_data = b6_ad.var.copy()
        treg_peak_data["sc.B6"] = total_counts_b6.T
        treg_peak_data["sc.CAST"] = total_counts_cast.T
        
        # Ensure consistent peak_name format
        treg_peak_data.reset_index(inplace=True)
        treg_peak_data.rename(columns={'index': 'peak_name'}, inplace=True)
        treg_peak_data['peak_name'] = treg_peak_data['peak_name'].astype(str)
        
        # Merge with main peak data
        peak_data['peak_name'] = peak_data['peak_name'].astype(str)
        peak_data = peak_data.merge(treg_peak_data, on="peak_name", how="inner")
        
        print(f"After merging single-cell data: {len(peak_data)} peaks")
        return peak_data
        
    except Exception as e:
        print(f"Warning: Could not load single-cell data: {e}")
        return peak_data

def load_peak_info(args, peak_data):
    """Load peak information and create peak_name with chr-position format"""
    if not args.peak_info:
        print("No peak info file provided, skipping peak name enhancement")
        return peak_data
    
    try:
        print(f"Loading peak info from: {args.peak_info}")
        # Read the peak info file
        peak_info = pd.read_csv(args.peak_info, header=None, sep='\t')
        
        # Name columns based on typical BED format
        if peak_info.shape[1] >= 3:
            peak_info.columns = ['chr', 'start', 'end'] + \
                              [f'col{i}' for i in range(4, peak_info.shape[1] + 1)]
            
            # Ensure chromosome has 'chr' prefix
            peak_info['chr'] = peak_info['chr'].astype(str).apply(
                lambda x: f"chr{x}" if not x.startswith('chr') else x
            )
            
            # Create simple peak_name with chr-position format
            peak_info['peak_name'] = peak_info.apply(
                lambda row: f"peak-{row['chr']}-{row['start']}-{row['end']}",
                axis=1
            )
            
            # Create a mapping from index to peak_name
            peak_name_map = dict(enumerate(peak_info['peak_name']))
            
            # Apply mapping to peak_data
            if 'peak_name' in peak_data.columns:
                # Store original peak names
                peak_data['original_peak_index'] = peak_data['peak_name'].copy()
                
                # Map index to new peak name
                peak_data['peak_name'] = peak_data['peak_name'].apply(
                    lambda x: peak_name_map.get(int(x) if isinstance(x, str) and x.isdigit() else x, 
                                               f"peak-unknown-{x}")
                )
            
            print(f"Created simple peak names with chr-position format")
            
            # Display sample peak names for verification
            print(f"Sample peak names: {peak_data['peak_name'].head(3).tolist()}")
            
        else:
            print(f"Peak info file has insufficient columns: {peak_info.shape[1]}, expected at least 3")
            
    except Exception as e:
        print(f"Warning: Could not process peak info file: {e}")
    
    return peak_data

def save_to_hdf5(args, peak_data):
    """Save processed data to HDF5 format"""
    print("Converting sequences to one-hot arrays...")
    
    # Convert sequences to one-hot encodings
    cast_sequence_np = data.DF2array(peak_data, "cast_sequence")
    b6_sequence_np = data.DF2array(peak_data, "B6_sequence")
    
    print(f"One-hot shapes: B6={b6_sequence_np.shape}, CAST={cast_sequence_np.shape}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    print(f"Saving to HDF5 file: {args.output}")
    
    # Save to HDF5
    with h5py.File(args.output, "w") as f:
        # Add metadata
        f.attrs['data_type'] = 'ATAC-seq'
        f.attrs['sequence_length'] = cast_sequence_np.shape[1]
        
        # Add sequence data
        f["Cast_sequence"] = cast_sequence_np
        f["B6_sequence"] = b6_sequence_np
        
        # Add peak names
        str_dtype = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("peak_name", data=np.array(peak_data["peak_name"].astype(str)), dtype=str_dtype)
        
        # Add all numerical columns
        for column in peak_data.columns:
            if column == "peak_name":
                continue
            if column.endswith("_sequence"):
                continue
            if column.startswith("length_"):
                continue
                
            # If B6 or CAST in column name, use log transformation
            if 'B6' in column or 'CAST' in column:
                try:
                    # Check if already logged
                    if not any(peak_data[column] < 0):  # Non-negative values might need logging
                        f[column] = np.array(np.log(1+peak_data[column]), dtype=float)
                    else:
                        f[column] = np.array(peak_data[column], dtype=float)
                except Exception as e:
                    print(f"Warning: Could not process column {column}: {e}")
            else:
                try:
                    f[column] = np.array(peak_data[column], dtype=float)
                except Exception as e:
                    print(f"Warning: Could not process column {column}: {e}")
    
    print(f"Successfully saved to {args.output}")

def main():
    args = parse_args()
    
    # Step 1: Load and process bulk ATAC-seq data
    peak_data = load_and_process_bulk_data(args)
    
    # Step 2: Load sequences and merge
    peak_data = load_sequences_and_merge(args, peak_data)
    
    # Step 3: Load and apply peak information (new step)
    peak_data = load_peak_info(args, peak_data)
    
    # Step 4: Load single-cell data if available
    peak_data = load_single_cell_data(args, peak_data)
    
    # Step 5: Save to HDF5
    save_to_hdf5(args, peak_data)
    
    print("ATAC-seq preprocessing completed successfully!")

if __name__ == "__main__":
    main()



