
from DeepAllele import tools, data
import pandas as pd
import numpy as np
import h5py
import os

def merge_peak_data_with_sequence(peak_data_df: pd.DataFrame, b6_sequence_df: pd.DataFrame, cast_sequence_df: pd.DataFrame,
                                  output_dir: str, output_name: str, on: str = "peak_name") -> None:
    """
    Merge peak data with B6 and CAST sequence data and save to an HDF5 file.

    Args:
        peak_data: pandas DataFrame containing the counts/ratio of peaks in B6 and CAST.
        b6_sequence: pandas DataFrame containing the sequence of peaks in B6.
        cast_sequence: pandas DataFrame containing the sequence of peaks in CAST.
        output_dir: directory to save the output file.
        output_name: name of the output file.
        on: column to join the dataframes on (default is "peak_name").

    Returns:
        None
    """
    # Merge dataframes
    peak_data_df = pd.merge(peak_data_df, b6_sequence_df, on=on)
    peak_data_df = pd.merge(peak_data_df, cast_sequence_df, on=on)

    # Convert sequences to numpy arrays
    cast_sequence_np = data.DF2array(peak_data_df, "cast_sequence")
    b6_sequence_np = data.DF2array(peak_data_df, "B6_sequence")

    tools.mkdir(output_dir)
    data_columns = [c for c in peak_data_df.columns if c not in ["B6_sequence", "cast_sequence", "length_x", "length_y"]]

    # Save data to HDF5 file
    with h5py.File(os.path.join(output_dir, output_name), "w") as f:
        f.create_dataset("Cast_sequence", data=cast_sequence_np)
        f.create_dataset("B6_sequence", data=b6_sequence_np)
        for column in data_columns:
            # Check if the column contains non-numeric values
            if peak_data_df[column].dtype == np.object:
                column_data = peak_data_df[column].values.astype("S")
            else:
                column_data = np.array(peak_data_df[column], dtype=float)
            f.create_dataset(column, data=column_data)




