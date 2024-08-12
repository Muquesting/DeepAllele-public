from DeepAllele import data
from DeepAllele import tools
from DeepAllele.data import DF2array
import h5py
import scanpy as sc
import pandas as pd
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="preprocess OCR sequences and counts data to HDF5 file"
    )

    parser.add_argument(
        "--in_dir", required=True, help="Input directory with sequences data"
    )

    parser.add_argument(
        "--in_fa", required=True, help="Input fasta folder with sequences data"
    )

    parser.add_argument("--maternal_fa", required=True, help="Maternal sequences data")

    parser.add_argument("--paternal_fa", required=True, help="Paternal sequences data")

    parser.add_argument(
        "--out_dir", required=True, help="Output directory with preprocessed data"
    )

    parser.add_argument(
        "--out_file", required=True, help="Output file with preprocessed data"
    )

    args = parser.parse_args()
    data_path = args.in_dir
    fa_path = args.in_fa
    output_path = args.out_dir
    output_file = args.out_file
    maternal_fa = fa_path + args.maternal_fa
    paternal_fa = fa_path + args.paternal_fa
    # TODO: decouple the path such as the .fa files

    # load anndata data
    B6_Treg_ad = sc.read_h5ad(data_path + "counts/Treg/B6_Treg.h5ad")
    cast_Treg_ad = sc.read_h5ad(data_path + "counts/Treg/cast_Treg.h5ad")
    peak_info = pd.read_csv(
        data_path + "peaks_info_updated_2021_12_16.txt", sep="\t", header=None
    )
    # rename the columns of chr, star, end, the peak id
    peak_info.columns = ["chr", "start", "end", "peak_name"]

    # Load OCR sequences into DataFrame
    B6_sequence_DF = data.load_fasta(maternal_fa)
    cast_sequence_DF = data.load_fasta(paternal_fa)

    B6_sequence_DF.rename(columns={"sequences": "B6_sequence"}, inplace=True)
    cast_sequence_DF.rename(columns={"sequences": "cast_sequence"}, inplace=True)

    # calculate the max length of the OCR sequences
    max_len = max(
        B6_sequence_DF["B6_sequence"].apply(len).max(),
        cast_sequence_DF["cast_sequence"].apply(len).max(),
    )

    total_counts_B6 = np.array(B6_Treg_ad.X.sum(axis=0))
    total_counts_cast = np.array(cast_Treg_ad.X.sum(axis=0))

    # add 1 to avoid log(0)
    total_counts_B6 += 1
    total_counts_cast += 1

    total_ratio = np.log(total_counts_B6 / total_counts_cast)
    B6_Treg_ad.var["total_ratio"] = total_ratio[0]

    peak_data = B6_Treg_ad.var.copy()
    peak_data["B6_counts"] = total_counts_B6.T
    peak_data["cast_counts"] = total_counts_cast.T

    peak_data = peak_data.merge(peak_info, on="peak_name", how="inner")
    peak_data = peak_data.merge(cast_sequence_DF, on="peak_name", how="inner")
    peak_data = peak_data.merge(B6_sequence_DF, on="peak_name", how="inner")

    # Filter NaN peaks
    peak_data_filter = peak_data[
        (peak_data["total_ratio"] != np.inf)
        & (peak_data["total_ratio"] != -np.inf)
        & (1 - np.isnan(np.array(peak_data["total_ratio"])))
    ]

    cast_sequence_np = DF2array(peak_data_filter, "cast_sequence", max_len)
    B6_sequence_np = DF2array(peak_data_filter, "B6_sequence", max_len)

    B6_counts = np.log(np.array(peak_data_filter["B6_counts"].tolist(), dtype=float))

    cast_counts = np.log(
        np.array(peak_data_filter["cast_counts"].tolist(), dtype=float)
    )

    ratio = np.array(peak_data_filter["total_ratio"].tolist(), dtype=float)
    chr = np.array(peak_data_filter["chr"])

    # Save data as np.array
    tools.mkdir(output_path)

    f = h5py.File(output_path + output_file, "w")
    f["Cast_sequence"] = cast_sequence_np
    f["B6_sequence"] = B6_sequence_np
    f["Cast_counts"] = cast_counts
    f["B6_counts"] = B6_counts
    f["ratio"] = ratio
    f["chr"] = chr

    f.close()
