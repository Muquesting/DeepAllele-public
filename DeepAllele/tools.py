import numpy as np
import os
from Bio import pairwise2
from scipy.stats import pearsonr, spearmanr
import multiprocessing
from tqdm import tqdm
from DeepAllele import model

def onehot_encoding(
    sequence: str,
    length: int,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value=0.25,
    dtype=np.float32,
) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def mkdir(Path):
    if not os.path.exists(Path):
        os.makedirs(Path)


def reversed_onehot(onehot_encoding):
    # reverse the onehot encoding to [A,C,G,T]
    onehot = np.argmax(onehot_encoding, axis=1)
    seqs = np.array(["A", "C", "G", "T"])[onehot]
    # print(onehot_encoding.max(axis=1))
    seqs[onehot_encoding.max(axis=1) == 0] = "-"
    seqs = "".join(seqs)
    return seqs


def alignments_onehot(seq1, seq2):
    seq1 = reversed_onehot(seq1)
    seq2 = reversed_onehot(seq2)
    print('B6:',seq1)
    print('CAST:',seq2)

    alignments = pairwise2.align.globalxs(seq1, seq2, -0.5, -0.1)[0]

    # print(alignments)
    seq1 = onehot_encoding(alignments.seqA, len(alignments.seqA))
    seq2 = onehot_encoding(alignments.seqB, len(alignments.seqB))
    return seq1, seq2


def transfer(weights):
    return np.exp(weights) / np.exp(weights).sum(axis=0, keepdims=True)


def amplify_pwm(pwm, amplify_factor=1.0):
    return transfer(np.log(pwm) * amplify_factor)


def write_meme_file(pwm, output_file_path, amplify_factor=1.0):
    """[summary]
    write the pwm to a meme file
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        output_file_path ([type]): [description]
    """
    n_filters = pwm.shape[0]
    if n_filters == 0:
        print("The number of filters is 0")
        return
    print(pwm.shape)
    if pwm.shape[2] == 4:
        print("Convert the shape of the pwm to n_filters * 4 * motif_length")
        pwm = np.transpose(pwm, (0, 2, 1))
    # check the sum of the pwm
    if (np.sum(pwm, axis=1).max() != 1.0) or (np.sum(pwm, axis=1).min() != 1.0):
        print("The sum of the pwm is not 1")
        print("Normalize the pwm")
        # normalize the pwm  to make the sum of the pwm equal to 1 for axis=1
        pwm = pwm / np.sum(pwm, axis=1, keepdims=True)

    l_filters = pwm.shape[2]
    meme_file = open(output_file_path, "w")
    meme_file.write("MEME version 4 \n")
    meme_file.write("ALPHABET= ACGT \n")
    meme_file.write("strands: + -\n")

    print("Saved PWM File as : {}".format(output_file_path))

    for i in range(0, n_filters):
        if np.sum(pwm[i, :, :]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s \n" % i)
            meme_file.write(
                "letter-probability matrix: alength= 4 w= %d \n"
                % np.count_nonzero(np.sum(pwm[i, :, :], axis=0))
            )

        for j in range(0, l_filters):
            if np.sum(pwm[i, :, j]) > 0:
                meme_file.write(
                    str(pwm[i, 0, j])
                    + "\t"
                    + str(pwm[i, 1, j])
                    + "\t"
                    + str(pwm[i, 2, j])
                    + "\t"
                    + str(pwm[i, 3, j])
                    + "\n"
                )

    meme_file.close()


def pearson_r(x, y, type_r="pearson"):
    filter_flag = np.isfinite(x) & np.isfinite(y)
    if type_r == "pearson":
        return pearsonr(x[filter_flag], y[filter_flag])[0]
    elif type_r == "spearman":
        return spearmanr(x[filter_flag], y[filter_flag])[0]
    raise ValueError("type_r must be pearson or spearman")


def tomtom_annotation(
    meme_file, target_meme_file, output_dir, output_name, evalue_threshold=0.1, dist="ed"
):
    cmd = "tomtom -o {}/{}  -evalue -thresh {} {} {} -dist {}".format(
        output_dir, output_name, evalue_threshold, meme_file, target_meme_file, dist
    )
    os.system(cmd)
    print("finish tomtom annotation for {}".format(meme_file))


def tomtom_annotation_parallel(
    meme_file_list, target_meme_file, output_dir, n_jobs=1, evalue_threshold=0.1, dist="ed"
):
    mkdir(output_dir)
    pool = multiprocessing.Pool(n_jobs)
    for meme_file in meme_file_list:
        output_name = meme_file.split("/")[-2]  # think about the name
        pool.apply_async(
            tomtom_annotation,
            args=(
                meme_file,
                target_meme_file,
                output_dir,
                output_name,
                evalue_threshold,
                dist,
            ),
        )
    pool.close()
    pool.join()

def get_pwm(act_seqs):
    return act_seqs.mean(axis=0)

def concate_genome(seq, axis=(0,-1), keepdims=False):
    # split the sequence based on the second axis and store the results in a list
    split_seq = np.split(seq, seq.shape[axis[1]], axis=axis[1])
    # concatenate the list of sequences along the first axis
    concate_seq = np.concatenate(split_seq, axis=axis[0])
    # keep the shape of the sequence or not
    if keepdims:
        return concate_seq
    else:
        return concate_seq.squeeze(axis=axis[1])



def extract_motifs_activation_seqs(
    activations, seqs, kernel_length, motif_length=15, activation_threshold=0.5, multiple_genome=False
):
    """[summary]
    extract the motifs from the activations and seqs
    Args:
        activations ([np.array]): n_sequences * n_filters * l_sequences [optional if multiple_genome=True] * n_genomes
        seqs ([np.array]): n_sequences * l_sequences * 4 [optional if multiple_genome=True] * n_genomes
        kernel_length ([int]): the length of the kernel
        motif_length ([int], optional): the length of the motif. Defaults to 15.
        activation_threshold ([float], optional): the threshold of the activation. Defaults to 0.5.
    Returns:
        [np.array]: n_filters * n_motifs * 4 * motif_length
    """
    if multiple_genome:
        # convert the activations and seqs from [n_sequences, n_filters, l_sequences,  n_genomes] to [n_sequences * n_genomes, n_filters, l_sequences]
        activations = concate_genome(activations, axis=(0,-1), keepdims=False)
        seqs = concate_genome(seqs, axis=(0,-1), keepdims=False)


    print("activations.shape: {}".format(activations.shape))
    print("seqs.shape: {}".format(seqs.shape))
    n_filters = activations.shape[1]
    n_sequences = seqs.shape[0]
    l_sequences = seqs.shape[1]

    activation_threshold = activation_threshold * np.amax(activations, axis=(0, 2))

    print("activation_threshold shape: {}".format(activation_threshold.shape)) 
    print("n_filters: {}".format(n_filters))
    print("n_sequences: {}".format(n_sequences))
    print("l_sequences: {}".format(l_sequences))

    # extract the motifs
    pwm = np.zeros((n_filters, motif_length, 4))
    n_activations = np.zeros(n_filters)
    offset = int((kernel_length) / 2) + int((motif_length-kernel_length)/2)
    for filter in tqdm(range(n_filters)):
        n_act = 0
        act_seqs = []
        for seq in range(n_sequences):
            indices = np.where(
                activations[seq, filter, :] > activation_threshold[filter]
            )
            for seq_start in indices[0]:
                seq_end = seq_start + motif_length

                if (seq_end < l_sequences)&(seq_start>=offset):
                    act_seqs.append(seqs[seq, seq_start-offset:seq_end-offset, :])
                    n_act += 1
        n_activations[filter] = n_act
        act_seqs = np.array(act_seqs)
        pwm[filter, :, :] = get_pwm(act_seqs)


    return pwm, n_activations


def information_content(pwm, pseudocount=0.001, background=[0.25, 0.25, 0.25, 0.25]):
    """[summary]
    calculate the information content of the pwm
    Args:
        pwm ([np.array]): n_filters * 4 * motif_length
        pseudocount ([float], optional): pseudocount. Defaults to 0.001.
        background ([list], optional): background. Defaults to [0.25, 0.25, 0.25, 0.25].
    Returns:
        [np.array]: n_filters * motif_length
    """
    raise NotImplementedError


def load_saved_model(ckpt_path,mh_or_sh,map_location = None): 
    
    if map_location is not None:
        if isinstance(map_location, int):
            if map_location >= 0:
                map_location = f'cuda:{map_location}'
            else:
                map_location = 'cpu'

    if mh_or_sh == 'mh': 
        curr_model = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(ckpt_path, map_location=map_location)
    elif mh_or_sh == 'sh': 
        curr_model = model.SingleHeadResidualCNN.load_from_checkpoint(ckpt_path, map_location=map_location)
    curr_model.eval()
    return curr_model