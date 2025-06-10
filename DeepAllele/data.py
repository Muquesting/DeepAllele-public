import os
from statistics import variance
from Bio import SeqIO
import pandas as pd
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from DeepAllele import tools

# TODO: speed up the DF2array function using apply function
def DF2array(DF, sequence_name, length=330):
    """[summary]
    Transform sequences(DataFrame files) to np.array
    Args:
        DF ([type]): [description]
        sequence_name ([type]): [description]
        length (int, optional): [description]. Defaults to 330.

    Returns:
        [np.array]: Onehot encoding of sequences array[n_seq, seq_length, 4]
    """
    n_seq = len(DF[sequence_name])
    sequence_np = np.zeros((n_seq, length, 4))
    for i, s in tqdm(enumerate(DF[sequence_name])):
        sequence_np[i, 0 : len(s), :] = tools.onehot_encoding(s, length)

    return sequence_np


def load_fasta(path):
    """[summary]
    load fasta file into DataFrame
    Args:
        path ([string]): [the path of fasta file]

    Returns:
        [DataFrame]: [description]
    """

    with open(path) as fasta_file:  # Will close handle cleanly
        identifiers = []
        sequences = []
        lengths = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
            identifiers.append(seq_record.id)
            sequences.append(str(seq_record.seq.upper()))
            lengths.append(len(seq_record.seq))
    Qfasta = pd.DataFrame(
        dict(peak_name=identifiers, sequences=sequences, length=lengths)
    )
    return Qfasta


def load_h5(
    path,
    split_ratio=0.8,
    batch_size=32,
    shuffle=True,
    batch_id=None,
    seed=42,
    split_by_chrom=False,
    val_chroms=None,
):
    """[summary]
    load h5 file into dataloader
    Args:
        path ([string]): [the path of h5 file]
        split_ratio ([float]): [the ratio of train and validation]
        batch_size (int, optional): [description]. Defaults to 32.
        shuffle (bool, optional): [description]. Defaults to False.
        batch_id ([string], optional): [the batch id of the data]. Defaults to None. or list of string
        seed (int, optional): [random seed]. Defaults to 42.
        split_by_chrom (bool, optional): [whether to split by chromosome]. Defaults to False.
        val_chroms (list, optional): [chromosomes to use for validation]. Defaults to ["16", "17", "18"].

    Returns:
        train_loader, val_loader, train_peak_name, val_peak_name
    """
    # Set default validation chromosomes if not specified
    if val_chroms is None:
        val_chroms = ["16", "17", "18"]
        
    f = h5py.File(path, "r")
    Cast_sequence = f["Cast_sequence"][:]
    B6_sequence = f["B6_sequence"][:]

    if batch_id is not None and len(batch_id) > 0:
        if isinstance(batch_id, str):
            batch_id = [batch_id]
        # concate for multiple batch
        for batch in batch_id:
            print(batch)

        ratio = np.stack([f[batch + ".ratio"][:] for batch in batch_id], axis=-1)
        B6_counts = np.stack([f[batch + ".B6"][:] for batch in batch_id], axis=-1)
        Cast_counts = np.stack([f[batch + ".CAST"][:] for batch in batch_id], axis=-1)
        # ratio = f[batch_id + ".ratio"][:]
        # B6_counts = f[batch_id + ".B6"][:]
        # Cast_counts = f[batch_id + ".CAST"][:]
        x_all = torch.from_numpy(np.stack([B6_sequence, Cast_sequence], -1)).float()
        y_all = torch.from_numpy(
            np.concatenate([B6_counts, Cast_counts, ratio], -1)
        ).float()
    else:
        ratio = f["ratio"][:]
        B6_counts = f["B6_counts"][:]
        Cast_counts = f["Cast_counts"][:]

        x_all = torch.from_numpy(np.stack([B6_sequence, Cast_sequence], -1)).float()
        y_all = torch.from_numpy(np.stack([B6_counts, Cast_counts, ratio], -1)).float()

    # check if there is peak_name in the keys
    try:
        peak_name_dataset = f["peak_name"][:]
        # decode to string
        peak_name_dataset = [
            peak_name.decode("utf-8") for peak_name in peak_name_dataset
        ]
        peak_name_dataset = np.array(peak_name_dataset)

        if split_by_chrom:
            print("splitting train and val by chrom")
            chroms = []
            for peak_name in peak_name_dataset:
                chrom = peak_name.split("chr")[1].split("-")[0]
                chroms.append(chrom)
            chroms = np.array(chroms)
            
            # Create validation index using the specified validation chromosomes
            val_index = np.zeros(len(chroms), dtype=bool)
            for chrom in val_chroms:
                val_index = val_index | (chroms == chrom)
                
            train_index = np.logical_not(val_index)
            print('splitting finished: using chromosomes {} for validation'.format(val_chroms))
            print('there are {} samples in val and {} samples in train'.format(val_index.sum(), train_index.sum()))

        else:
            print("not splitting train and val by chrom")
            n_sample = x_all.shape[0]
            n_train_sample = int(n_sample * split_ratio)
            train_index, val_index = random_split(
                range(n_sample),
                [n_train_sample, n_sample - n_train_sample],
                generator=torch.Generator().manual_seed(seed),
            )

        # train peak name, val peak name with fast indexing
        train_peak_name = peak_name_dataset[train_index]
        val_peak_name = peak_name_dataset[val_index]
        print("decode the peak name successfully")

    except:
        print("No peak_name in the hdf5 file")
        train_peak_name = None
        val_peak_name = None

        # if not split_by_chrom:
        n_sample = x_all.shape[0]
        n_train_sample = int(n_sample * split_ratio)
        train_index, val_index = random_split(
            range(n_sample),
            [n_train_sample, n_sample - n_train_sample],
            generator=torch.Generator().manual_seed(seed),
        )

    train_dataset = TensorDataset(x_all[train_index], y_all[train_index])
    val_dataset = TensorDataset(x_all[val_index], y_all[val_index])

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )
    # never shuffle the validation data
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=32
    )

    return trainloader, valloader, train_peak_name, val_peak_name


def load_h5_single(
    path,
    split_ratio=0.8,
    batch_size=32,
    shuffle=True,
    batch_id=None,
    seed=42,
    Genome=["B6", "Cast"],
    split_by_chrom=False,
    val_chroms=None,
):
    """[summary]
    load h5 file into dataloader
    Args:
        path ([string]): [the path of h5 file]
        split_ratio ([float]): [the ratio of train and validation]
        batch_size (int, optional): [description]. Defaults to 32.
        shuffle (bool, optional): [description]. Defaults to False.
        batch_id ([string], optional): [the batch id of the data]. Defaults to None. or list of string
        seed (int, optional): [random seed]. Defaults to 42.
        Genome (list, optional): [list of genome names]. Defaults to ["B6", "Cast"].
        split_by_chrom (bool, optional): [whether to split by chromosome]. Defaults to False.
        val_chroms (list, optional): [chromosomes to use for validation]. Defaults to ["16", "17", "18"].

    Returns:
        train_loader, val_loader, train_peak_name, val_peak_name
    """

    f = h5py.File(path, "r")
    Cast_sequence = f["Cast_sequence"][:]
    B6_sequence = f["B6_sequence"][:]

    if batch_id is not None and len(batch_id) > 0:
        if isinstance(batch_id, str):
            batch_id = [batch_id]
        # concate for multiple batch
        for batch in batch_id:
            print(batch)

        ratio = np.stack([f[batch + ".ratio"][:] for batch in batch_id], axis=-1)
        # print(ratio.shape)
        B6_counts = np.stack([f[batch + ".B6"][:] for batch in batch_id], axis=-1)
        Cast_counts = np.stack([f[batch + ".CAST"][:] for batch in batch_id], axis=-1)

    else:
        ratio = f["ratio"][:]
        B6_counts = f["B6_counts"][:]
        Cast_counts = f["Cast_counts"][:]
        # add the dim for the channel to make it consistent with the batch data
        B6_counts = np.expand_dims(B6_counts, axis=-1)
        Cast_counts = np.expand_dims(Cast_counts, axis=-1)
        ratio = np.expand_dims(ratio, axis=-1)

    try:
        peak_name_dataset = f["peak_name"][:]
        # decode to string
        peak_name_dataset = [
            peak_name.decode("utf-8") for peak_name in peak_name_dataset
        ]
        peak_name_dataset = np.array(peak_name_dataset)

        # print(peak_name_dataset)

        # new ------
        if split_by_chrom:
            print("splitting by chrom")

            chroms = []
            for peak_name in peak_name_dataset:
                chrom = peak_name.split("chr")[1].split("-")[0]
                chroms.append(chrom)
            chroms = np.array(chroms)

            # Set default validation chromosomes if not specified
            if val_chroms is None:
                val_chroms = ["16", "17", "18"]
            
            # Create validation index using the specified validation chromosomes
            val_index = np.zeros(len(chroms), dtype=bool)
            for chrom in val_chroms:
                val_index = val_index | (chroms == chrom)

            train_index = np.logical_not(val_index)
            print(
                "splitting finished: using chromosomes {} for validation, {} samples in val and {} samples in train".format(
                    val_chroms, val_index.sum(), train_index.sum()
                )
            )

        else:
            print("not splitting train and val by chrom")
            n_sample = ratio.shape[0]
            n_train_sample = int(n_sample * split_ratio)
            train_index, val_index = random_split(
                range(n_sample),
                [n_train_sample, n_sample - n_train_sample],
                generator=torch.Generator().manual_seed(seed),
            )

        # train peak name, val peak name with fast indexing
        train_peak_name = peak_name_dataset[train_index]
        val_peak_name = peak_name_dataset[val_index]
        print("decode the peak name successfully")
        # return trainloader, valloader, train_peak_name, val_peak_name

    except:
        print("No peak_name in the hdf5 file")
        train_peak_name = None
        val_peak_name = None

        # if not split_by_chrom:
        # n_sample = x_all.shape[0]
        n_sample = ratio.shape[0]

        n_train_sample = int(n_sample * split_ratio)
        train_index, val_index = random_split(
            range(n_sample),
            [n_train_sample, n_sample - n_train_sample],
            generator=torch.Generator().manual_seed(seed),
        )

    for genome in Genome:
        sequence = f[genome + "_sequence"][:]
        if batch_id is not None and len(batch_id) > 0:
            counts = np.stack(
                [f[batch + "." + genome.upper()] for batch in batch_id], axis=-1
            )
        else:
            counts = f[genome + "_counts"][:]
            counts = np.expand_dims(counts, axis=-1)

        if genome == Genome[0]:
            x_all = torch.from_numpy(sequence).float()
            y_all = torch.from_numpy(counts).float()

            x_train = x_all[train_index]
            y_train = y_all[train_index]

        else:
            x_all = torch.from_numpy(sequence).float()
            y_all = torch.from_numpy(counts).float()

            # cat the data
            x_train = torch.cat((x_train, x_all[train_index]), dim=0)
            y_train = torch.cat((y_train, y_all[train_index]), dim=0)

    # load the validation data
    # B6_counts = np.stack([f[batch + ".B6"][:] for batch in batch_id], axis=-1)
    # Cast_counts = np.stack([f[batch + ".CAST"][:] for batch in batch_id], axis=-1)

    x_val = torch.from_numpy(B6_sequence[val_index]).float()
    y_val = torch.from_numpy(B6_counts[val_index]).float()

    x_val = torch.cat(
        (x_val, torch.from_numpy(Cast_sequence[val_index]).float()), dim=0
    )
    y_val = torch.cat((y_val, torch.from_numpy(Cast_counts[val_index]).float()), dim=0)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )
    # never shuffle the validation data
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=32
    )

    return trainloader, valloader, train_peak_name, val_peak_name



def unfold_loader(loader):
    seqs_list = []
    labels_list = []

    for batch_idx, (seqs, labels) in enumerate(loader):
        seqs_list.append(seqs)
        labels_list.append(labels)

    seqs = torch.cat(seqs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    # return numpy array
    return seqs.numpy(), labels.numpy()
