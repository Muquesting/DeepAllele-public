from statistics import variance
from Bio import SeqIO
import pandas as pd
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from DeepAllele import tools

# added 8/24
def get_peak_chroms(peak_labels):
    peak_data = pd.read_csv(
        "/data/tuxm/project/F1-ASCA/data/raw_data/peaks_info_updated_2021_12_16.txt",
        delimiter="\t",
        header=None,
    )
    peak_data.columns = ["chr", "start", "end", "peak_name"]
    peak_data.set_index("peak_name", inplace=True)
    chroms = []
    for label in peak_labels:
        # print(label)

        if label in peak_data.index:
            # print('in data')
            if isinstance(peak_data.loc[label]["chr"], str):
                # print(peak_data.loc[label]['chr'])
                chrom = peak_data.loc[label]["chr"].split("r")[1]
                chroms.append(chrom)
            else:  # its nan
                chroms.append("nan_chr_label")

        else:
            chroms.append("not_in_dataset")

    chroms = np.array(chroms)
    return chroms


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
):
    """[summary]
    load h5 file into dataloader
    Args:
        path ([string]): [the path of h5 file]
        split_ratio ([float]): [the ratio of train and validation]
        batch_size (int, optional): [description]. Defaults to 32.
        shuffle (bool, optional): [description]. Defaults to False.
        batch_id ([string], optional): [the batch id of the data]. Defaults to None. or list of string

    Returns:
        train_loader, val_loader, train_peak_name, val_peak_name
    """
    f = h5py.File(path, "r")
    Cast_sequence = f["Cast_sequence"][:]
    B6_sequence = f["B6_sequence"][:]

    if batch_id is not None:
        if isinstance(batch_id, str):
            batch_id = [batch_id]
        # concate for multiple batch
        for batch in batch_id:
            print(batch)

        ratio = np.stack([f[batch + ".ratio"][:] for batch in batch_id], axis=-1)
        # print(ratio.shape)
        B6_counts = np.stack([f[batch + ".B6"][:] for batch in batch_id], axis=-1)
        Cast_counts = np.stack([f[batch + ".CAST"][:] for batch in batch_id], axis=-1)
        # ratio = f[batch_id + ".ratio"][:]
        # B6_counts = f[batch_id + ".B6"][:]
        # Cast_counts = f[batch_id + ".CAST"][:]
        x_all = torch.from_numpy(np.stack([B6_sequence, Cast_sequence], -1)).float()
        y_all = torch.from_numpy(
            np.concatenate([B6_counts, Cast_counts, ratio], -1)
        ).float()
        # print(x_all.shape, y_all.shape)
    else:
        ratio = f["ratio"][:]
        B6_counts = f["B6_counts"][:]
        Cast_counts = f["Cast_counts"][:]

        x_all = torch.from_numpy(np.stack([B6_sequence, Cast_sequence], -1)).float()
        y_all = torch.from_numpy(np.stack([B6_counts, Cast_counts, ratio], -1)).float()

    # moved --------

    # check if there is peak_name in the keys
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
            print("splitting train and val by chrom")

            if (
                path
                == "/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5"
            ):
                print("ATAC seq dataset")

                chroms = get_peak_chroms(peak_name_dataset)

                print(chroms[:20])
                # val_index = (chroms == '1') | (chroms == '19')
                val_index = (chroms == "16") | (chroms == "17") | (chroms == "18")

                not_val_index = np.logical_not(val_index)
                remove_index = np.logical_not(
                    (chroms == "nan_chr_label") | (chroms == "not_in_dataset")
                )
                train_index = remove_index & not_val_index
                print("finish splitting train and val by chrom")

            else:
                print("not ATAC seq dataset")
                chroms = []
                for peak_name in peak_name_dataset:
                    chrom = peak_name.split("-")[1].split("r")[1]
                    chroms.append(chrom)
                chroms = np.array(chroms)

                # print(len(chroms))

                # val_index = (chroms == '1') | (chroms == '19')
                val_index = (chroms == "16") | (chroms == "17") | (chroms == "18")
                train_index = np.logical_not(val_index)

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
        # return trainloader, valloader, train_peak_name, val_peak_name

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

    # print(x_all.shape)
    # print(train_index.shape)
    # print(val_index.shape)

    train_dataset = TensorDataset(x_all[train_index], y_all[train_index])
    val_dataset = TensorDataset(x_all[val_index], y_all[val_index])

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )

    return trainloader, valloader, train_peak_name, val_peak_name


# TODO: add the peak name to the data


def load_data(
    path,
    split_ratio,
    batch_size=32,
    metacell=False,
    shuffle=False,
    batch_id=None,
    color_feature=None,
    null=False,
    repeat=None,
):
    # TODO: add the function to load the data to predict metacell level
    # TODO: Decouple the bulk data and single cell data, meta cell
    f = h5py.File(path, "r")
    Cast_sequence = f["Cast_sequence"][:]
    B6_sequence = f["B6_sequence"][:]
    # peak_name = f["peak_name"][:]

    if batch_id is not None:
        # for the bulk seq data the log is not calculated so we have to calculate it
        ratio = f[batch_id + ".ratio"][:]
        # Cast_counts = np.log(f[batch_id + ".CAST"][:])
        # B6_counts = np.log(f[batch_id + ".B6"][:])
        Cast_counts = np.log(f[batch_id + ".CAST"][:] + 1)
        B6_counts = np.log(f[batch_id + ".B6"][:] + 1)
    else:
        ratio = f["ratio"][:]
        Cast_counts = f["Cast_counts"][:]
        B6_counts = f["B6_counts"][:]
    if repeat is not None:
        if repeat == "B6":
            x_all = torch.from_numpy(np.stack([B6_sequence, B6_sequence], -1)).float()
        elif repeat == "Cast":
            x_all = torch.from_numpy(
                np.stack([Cast_sequence, Cast_sequence], -1)
            ).float()
    else:
        x_all = torch.from_numpy(np.stack([B6_sequence, Cast_sequence], -1)).float()

    y_all = torch.from_numpy(np.stack([B6_counts, Cast_counts, ratio], -1)).float()

    if null:
        # shuffle the data y_all
        # y_all = y_all[np.random.permutation(y_all.shape[0])]
        print(y_all.shape)
        # shuffle the data x_all dim = 2
        indices = torch.argsort(torch.rand(*x_all.shape), dim=1)
        x_all = torch.gather(x_all, dim=1, index=indices)
        # x_all = x_all[:, np.random.permutation(x_all.shape[1]), :, :]
        print(x_all.shape)
    Sequence_dataset = TensorDataset(x_all, y_all)  # create your datset

    n_sample = x_all.shape[0]
    n_train_sample = int(n_sample * split_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        Sequence_dataset,
        [n_train_sample, n_sample - n_train_sample],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )
    validloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )

    if color_feature is not None:
        # TODO support the list of color_feature
        feature = f[color_feature][:]
        train_feature, val_feature = torch.utils.data.random_split(
            feature,
            [n_train_sample, n_sample - n_train_sample],
            generator=torch.Generator().manual_seed(
                42
            ),  # keep same with the data split
        )
        return trainloader, validloader, train_feature, val_feature

    return trainloader, validloader


def load_h5_single(
    path,
    split_ratio=0.8,
    batch_size=32,
    shuffle=True,
    batch_id=None,
    seed=42,
    Genome=["B6", "Cast"],
    split_by_chrom=False,
):
    """[summary]
    load h5 file into dataloader
    Args:
        path ([string]): [the path of h5 file]
        split_ratio ([float]): [the ratio of train and validation]
        batch_size (int, optional): [description]. Defaults to 32.
        shuffle (bool, optional): [description]. Defaults to False.
        batch_id ([string], optional): [the batch id of the data]. Defaults to None. or list of string

    Returns:
        train_loader, val_loader, train_peak_name, val_peak_name
    """

    f = h5py.File(path, "r")
    Cast_sequence = f["Cast_sequence"][:]
    B6_sequence = f["B6_sequence"][:]

    if batch_id is not None:
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

            if (
                path
                == "/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5"
            ):
                print("ATAC seq dataset")

                chroms = get_peak_chroms(peak_name_dataset)

                # val_index = (chroms == '1') | (chroms == '19')
                val_index = (chroms == "16") | (chroms == "17") | (chroms == "18")

                not_val_index = np.logical_not(val_index)
                remove_index = np.logical_not(
                    (chroms == "nan_chr_label") | (chroms == "not_in_dataset")
                )
                train_index = remove_index & not_val_index

            else:
                print("not ATAC seq dataset")
                chroms = []
                for peak_name in peak_name_dataset:
                    chrom = peak_name.split("-")[1].split("r")[1]
                    chroms.append(chrom)
                chroms = np.array(chroms)

                # print(len(chroms))

                # val chroms = ['1', '19']
                # val_index = (chroms == '1') | (chroms == '19')
                val_index = (chroms == "16") | (chroms == "17") | (chroms == "18")

                train_index = np.logical_not(val_index)

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
        if batch_id is not None:
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
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )

    return trainloader, valloader, train_peak_name, val_peak_name


def load_single_data(
    path,
    split_ratio,
    input_sequence_list=["Cast"],
    batch_size=32,
    shuffle=False,
):
    # TODO add comments to the function

    f = h5py.File(path, "r")
    for input_sequence in input_sequence_list:
        sequence = f[input_sequence + "_sequence"][:]
        counts = f[input_sequence + "_counts"][:]
        if input_sequence == input_sequence_list[0]:
            x_all = torch.from_numpy(sequence)
            y_all = torch.from_numpy(counts)
        else:
            x_all = torch.cat((x_all, torch.from_numpy(sequence)), 1)
            y_all = torch.cat((y_all, torch.from_numpy(counts)), 1)
    # x_all = torch.from_numpy(sequence)
    # y_all = torch.from_numpy(counts)
    Sequence_dataset = TensorDataset(x_all, y_all)  # create your datset
    n_sample = x_all.shape[0]
    n_train_sample = int(n_sample * split_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        Sequence_dataset,
        [n_train_sample, n_sample - n_train_sample],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )
    validloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )

    return trainloader, validloader


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
