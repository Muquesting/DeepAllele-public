import pandas as pd
import numpy as np
import pysam
from tqdm import tqdm
import h5py
from DeepAllele import tools
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess RNA-seq data')
    parser.add_argument('--vcf-file', required=True,
                        help='Path to VCF file')
    parser.add_argument('--gene-meta', required=True,
                        help='Path to gene metadata file')
    parser.add_argument('--genome-file', required=True,
                        help='Path to genome FASTA file')
    parser.add_argument('--cast-expr', required=True,
                        help='Path to CAST expression file')
    parser.add_argument('--b6-expr', required=True,
                        help='Path to B6 expression file')
    parser.add_argument('--output', required=True,
                        help='Path to output HDF5 file')
    parser.add_argument('--window-size', type=int, default=5000,
                        help='Window size around TSS')
    parser.add_argument('--padding', type=int, default=100,
                        help='Extra padding for sequence length (default: 100)')
    return parser.parse_args()

def DF2array(DF, sequence_name, length=None):
    """
    Transform sequences to one-hot encoded arrays: [n_seq, seq_length, 4].
    If length is None, it will use the maximum sequence length.
    """
    # Determine maximum length of sequences if not provided
    if length is None:
        max_length = max(len(s) for s in DF[sequence_name])
        print(f"Using maximum sequence length: {max_length}")
        length = max_length
    
    n_seq = len(DF[sequence_name])
    sequence_np = np.zeros((n_seq, length, 4))
    
    for i, s in tqdm(enumerate(DF[sequence_name]), total=n_seq, desc=f"One-hot {sequence_name}"):
        sequence_np[i, 0: len(s), :] = tools.onehot_encoding(s)
    return sequence_np

def fetch_variant_sequences(gene_name, gene_meta_file, vcf_file_path, genome_file_path, window_size=50000, verbose=False):
    """
    Fetch and modify sequences around the TSS of a given gene
    based on variants from a VCF file.
    """
    select_gene_info = gene_meta_file[gene_meta_file['gene_name'] == gene_name]
    if select_gene_info.empty:
        raise ValueError(f"No gene information found for gene name: {gene_name}")

    # Determine TSS based on strand
    if select_gene_info['strand'].values[0] == '+':
        tss_pos = select_gene_info['start'].values[0]
    else:
        tss_pos = select_gene_info['end'].values[0]

    chr_ = select_gene_info['seqname'].values[0]

    # Make sure we don't fetch a negative position
    pos_start = max(1, tss_pos - window_size // 2)
    pos_end   = tss_pos + window_size // 2

    with pysam.VariantFile(vcf_file_path) as vcf, pysam.FastaFile(genome_file_path) as genome:
        # Some VCFs might drop 'chr' prefix; adjust if needed
        contig = chr_.replace('chr', '')  
        
        # Note: Python is 0-based in fetch
        # genome.fetch(chrom, start, end) fetches [start, end) (end non-inclusive)
        seq = genome.fetch(chr_, pos_start - 1, pos_end - 1).upper()
        B6_seq = seq

        shift = 0
        for record in vcf.fetch(contig, pos_start, pos_end):
            position = record.pos - pos_start + shift
            ref_allele = record.ref.upper()
            alt_allele = record.alts[0].upper()

            seq_before_variant = seq[:position]
            seq_after_variant  = seq[position + len(ref_allele):]

            # Check if reference matches the sequence
            if seq[position:position + len(ref_allele)] != ref_allele:
                if verbose:
                    print(f"Warning: mismatch in {gene_name} at position {position}")
                continue

            # Replace with alt
            seq  = seq_before_variant + alt_allele + seq_after_variant
            shift += len(alt_allele) - len(ref_allele)

        CAST_seq = seq
        return B6_seq, CAST_seq

# Main function to organize workflow
def main():
    # Parse command line arguments
    args = parse_args()
    
    # ---------------------------
    # 1. Define file paths from arguments
    # ---------------------------
    vcf_file_path = args.vcf_file
    gene_meta_file_path = args.gene_meta
    genome_file_path = args.genome_file
    cast_expr_file_path = args.cast_expr
    b6_expr_file_path = args.b6_expr
    output_h5_file = args.output
    window_size = args.window_size
    
    print(f"Processing with window size: {window_size}")
    print(f"Output will be saved to: {output_h5_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_file)), exist_ok=True)
    
    # ---------------------------
    # 2. Read metadata, genome, and VCF
    # ---------------------------
    gene_meta_file = pd.read_csv(gene_meta_file_path)
    genome = pysam.FastaFile(genome_file_path)
    vcf_file = pysam.VariantFile(vcf_file_path)
    
    # ---------------------------
    # 3. Read in expression data
    # ---------------------------
    CAST_data = pd.read_csv(cast_expr_file_path, sep='\t')
    B6_data   = pd.read_csv(b6_expr_file_path,   sep='\t')

    print("CAST data columns:\n", CAST_data.columns)
    print("B6 data columns:\n",   B6_data.columns)
    # Example columns:
    # CAST: ['# Gene', 'B_Fo_Sp_IL4_CAST', 'B_Fo_Sp_PBS_CAST', 'MF_PC_IL4_CAST', ...]
    # B6:   ['# Gene', 'B_Fo_Sp_IL4_B6',   'B_Fo_Sp_PBS_B6',   'MF_PC_IL4_B6',   ...]

    # ---------------------------
    # 4. Restrict to overlapping genes
    # ---------------------------
    gene_list = set(CAST_data['# Gene']) & set(B6_data['# Gene']) & set(gene_meta_file['gene_name'])
    gene_list = list(gene_list)
    print("Number of overlapping genes:", len(gene_list))

    # Filter out only those genes
    CAST_data = CAST_data[CAST_data['# Gene'].isin(gene_list)]
    B6_data   = B6_data[B6_data['# Gene'].isin(gene_list)]

    # ---------------------------
    # 5. Merge B6 and CAST expression
    #    Remove the suffixes parameter since columns already end with _B6/_CAST
    # ---------------------------
    expr_df = pd.merge(
        B6_data,
        CAST_data,
        on="# Gene",
        how="inner"
    )
    # Rename the gene column to something simpler
    expr_df.rename(columns={"# Gene": "gene_name"}, inplace=True)

    # ---------------------------
    # 6. Loop through each gene to get sequences
    # ---------------------------
    list_genes, list_B6_seq, list_CAST_seq = [], [], []

    for gene_name in tqdm(gene_list, desc="Fetching sequences"):
        try:
            B6_seq, CAST_seq = fetch_variant_sequences(
                gene_name, 
                gene_meta_file, 
                vcf_file_path, 
                genome_file_path,
                window_size=window_size,
                verbose=False
            )
            list_genes.append(gene_name)
            list_B6_seq.append(B6_seq)
            list_CAST_seq.append(CAST_seq)
        except Exception as e:
            print(f"Skipping gene {gene_name} due to error: {e}")
            continue

    seq_df = pd.DataFrame({
        "gene_name": list_genes,
        "B6_sequence": list_B6_seq,
        "CAST_sequence": list_CAST_seq
    })

    # Merge sequence data with expression data
    final_df = pd.merge(expr_df, seq_df, on="gene_name", how="inner")

    # Right before writing final_df to HDF5, define peak_name:
    gene_meta_cols = gene_meta_file.set_index("gene_name")[["seqname", "start", "end"]]
    final_df["peak_name"] = final_df["gene_name"].apply(
        lambda g: f"{g}-chr{gene_meta_cols.loc[g, 'seqname'].replace('chr','')}"
                f"-{gene_meta_cols.loc[g, 'start']}"
                f"-{gene_meta_cols.loc[g, 'end']}"
        if g in gene_meta_cols.index else g
    )

    # -----------
    # 6) Compute sum columns, ratio, etc. (same logic as before)
    # -----------
    b6_cols   = [c for c in final_df.columns if c.endswith("_B6")   and c not in ["gene_name"]]
    cast_cols = [c for c in final_df.columns if c.endswith("_CAST") and c not in ["gene_name"]]

    final_df["sum.B6"]   = final_df[b6_cols].sum(axis=1)
    final_df["sum.CAST"] = final_df[cast_cols].sum(axis=1)
    final_df["sum.ratio"]    = final_df["sum.B6"] - final_df["sum.CAST"] # because it's log-transformed data

    # Calculate sequence arrays with dynamic length determination
    print("Generating one-hot encoded sequences...")
    # Determine the actual max sequence length or use window_size + padding
    sequence_length = max(
        max(len(s) for s in final_df["B6_sequence"]),
        max(len(s) for s in final_df["CAST_sequence"])
    )
    print(f"Maximum sequence length: {sequence_length}")
    
    # Use either max length or window size, whichever is larger (without padding)
    encoding_length = max(sequence_length, window_size)
    
    # One-hot encode sequences
    B6_sequence_np = DF2array(final_df, "B6_sequence", length=encoding_length)
    CAST_sequence_np = DF2array(final_df, "CAST_sequence", length=encoding_length)
    
    # -----------
    # 8) Write to HDF5
    # -----------
    print(f"Writing to HDF5 file: {output_h5_file}")
    f = h5py.File(output_h5_file, "w")
    
    # Store metadata about the processing
    f.attrs['window_size'] = window_size
    f.attrs['sequence_length'] = encoding_length

    # 8A) Store the one-hot arrays (unchanged)
    f.create_dataset("B6_sequence", data=B6_sequence_np, compression="gzip", shuffle=True)
    f.create_dataset("Cast_sequence", data=CAST_sequence_np, compression="gzip", shuffle=True)

    # 8B) Store gene names
    str_dtype = h5py.string_dtype(encoding='utf-8')
    f.create_dataset("gene_name", data=final_df["gene_name"].astype(object), dtype=str_dtype)
    f.create_dataset("peak_name", data=final_df["peak_name"].astype(str), dtype=str_dtype)

    # 8C) Store individual condition data
    # Get condition names without _B6/_CAST suffix
    conditions = set([col.replace('_B6', '').replace('_CAST', '') 
                    for col in b6_cols + cast_cols])

    for condition in conditions:
        b6_col = f"{condition}_B6"
        cast_col = f"{condition}_CAST"
        
        if b6_col in final_df.columns and cast_col in final_df.columns:
            # Store raw counts
            f.create_dataset(f"{condition}.B6", data=final_df[b6_col].to_numpy(dtype=float))
            f.create_dataset(f"{condition}.CAST", data=final_df[cast_col].to_numpy(dtype=float))
            
            # Calculate and store ratio for this condition
            ratio = final_df[b6_col] - final_df[cast_col] # because it's log-transformed data, so subtraction is equivalent to division
            f.create_dataset(f"{condition}.ratio", data=ratio.to_numpy(dtype=float))

    # 8D) Store sum columns (optional, but keeping for backward compatibility)
    f.create_dataset("sum.B6", data=final_df["sum.B6"].to_numpy(dtype=float))
    f.create_dataset("sum.CAST", data=final_df["sum.CAST"].to_numpy(dtype=float))
    f.create_dataset("sum.ratio", data=final_df["sum.ratio"].to_numpy(dtype=float))

    # Print available batch_ids for reference
    conditions = [k.split('.')[0] for k in f.keys() if '.ratio' in k]
    print("\nAvailable batch_ids for load_h5:")
    print(conditions)

    f.close()
    print("Done! Created HDF5 file with data formatted for load_h5 function at", output_h5_file)

if __name__ == "__main__":
    main()




