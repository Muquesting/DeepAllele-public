import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io
from scipy.sparse import csc_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse


def process_mtx(data_path, output_path):
    """[summary]
    1. load the mtx counts matrix into anndata formation
    2. save the h5ad files
    Args:
        data_path ([str]): [data path]
        output_path ([str]): [output path]
    """

    # load peak list and barcode list
    peak_list = pd.read_csv(data_path + "counts/ocr_list.txt", header=None)
    barcode_list = pd.read_csv(
        data_path + "counts/Treg/barcodes_list_Treg.txt", header=None
    )
    print("Finish peak and barcode list loading")

    # save the count matrix as anndata type

    B6_Treg_ad = sc.AnnData(io.mmread(data_path + "counts/Treg/b6_counts_Treg.mtx").T)
    B6_Treg_ad.var["peak_name"] = peak_list[0].tolist()
    B6_Treg_ad.obs["barcodes"] = barcode_list[0].tolist()

    cast_Treg_ad = sc.AnnData(
        io.mmread(data_path + "counts/Treg/cast_counts_Treg.mtx").T
    )
    cast_Treg_ad.var["peak_name"] = peak_list[0].tolist()
    cast_Treg_ad.obs["barcodes"] = barcode_list[0].tolist()

    print("Finish counts matrix loading")

    # convert the data type to CSC_matrix
    # save the data
    B6_Treg_ad.X = csc_matrix(B6_Treg_ad.X)
    B6_Treg_ad.write(output_path + "B6_Treg.h5ad")

    cast_Treg_ad.X = csc_matrix(cast_Treg_ad.X)
    cast_Treg_ad.write(output_path + "cast_Treg.h5ad")

    cast_Treg_ad.X = csc_matrix(cast_Treg_ad.X)
    cast_Treg_ad.write(output_path + "cast_Treg.h5ad")

    print("Finish saving")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="counts matrix preprocess")

    parser.add_argument(
        "--in_dir", required=True, help="Input directory with counts matrix"
    )
    # TODO add out_dir
    # parser.add_argument(
    #     "--out_dir", required=True, help="Output directory with trained model"
    # )
    args = parser.parse_args()
    data_path = args.in_dir

    output_path = data_path + "counts/Treg/"
    process_mtx(data_path=data_path, output_path=output_path)
