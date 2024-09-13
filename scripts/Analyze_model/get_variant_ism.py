import numpy as np 
import pandas as pd
import os
import argparse
from Bio import pairwise2
from collections import defaultdict
import sys
import torch
from DeepAllele import tools
from get_predictions import get_predictions

def first_non_dash_index(string):
    for index, char in enumerate(string):
        if char != '-':
            return index
    return -1  # if all characters are '-'

def last_non_dash_index(string):
    last_index = -1
    for index, char in enumerate(string):
        if char != '-':
            last_index = index
    return last_index

def remove_gaps_and_pad(seq_to_pad):
    # for removing gaps from aligned seqs to pass through the model 
    # takes (1,len,4), returns (1,len,4) 
    final_len = seq_to_pad.shape[1]
    mask = np.all(seq_to_pad[:, :, :] == 0, axis=2).reshape(-1)
    filtered_seq = seq_to_pad[:, ~mask, :]
    pad_diff = final_len - filtered_seq.shape[1]
    pad_width = [(0, 0), (0, pad_diff), (0, 0)]
    padded_seq = np.pad(filtered_seq, pad_width, mode='constant', constant_values=0)
    return padded_seq

def save_var_info(save_dir, seqs_path):    
    seqs_all = np.load(seqs_path)    
    seq_idxs = []
    non_matching_indices = []
    string0_at_non_matching = []
    string1_at_non_matching = []
    # go through seqs and get info for any vars present 
    for seq_idx in range(len(seqs_all)): 
        if seq_idx%1000 == 0: 
            print(seq_idx)
        string0, string1, onehot0, onehot1 = alignments_onehot(seqs_all[seq_idx,:,:,0],seqs_all[seq_idx,:,:,1])
        j=0
        while (j<len(string0)):
            if string0[j] != string1[j]:
                seq_idxs.append(seq_idx) 
                if string0[j] == '-' or string1[j] == '-': # find when the dashes end on this string             
                    if string0[j] == '-':
                        after_j_non_dash = first_non_dash_index(string0[j+1:])
                        next_non_string_idx = after_j_non_dash + j+1
                    elif string1[j] == '-': 
                        after_j_non_dash = first_non_dash_index(string1[j+1:])
                        next_non_string_idx = after_j_non_dash + j+1
                        
                    if after_j_non_dash == -1: 
                        # there is an indel before the end of the seq: this var goes until both strings are -
                        # or, its just the last variant, the rest is - 
                        end_of_var = max(max(last_non_dash_index(string1)+1,last_non_dash_index(string0)+1),j+1) # add the +1 for indexing
                    else: 
                        end_of_var = next_non_string_idx
                    
                    if end_of_var==j+1: # len == 1
                        non_matching_indices.append(str(j))
                        string0_at_non_matching.append(string0[j])
                        string1_at_non_matching.append(string1[j])
                        j+=1
                    else: # indel 
                        non_matching_indices.append(str(j)+':'+str(end_of_var))
                        string0_at_non_matching.append(string0[j:end_of_var])
                        string1_at_non_matching.append(string1[j:end_of_var])
                        j = end_of_var
                else: 
                    non_matching_indices.append(str(j))
                    string0_at_non_matching.append(string0[j])
                    string1_at_non_matching.append(string1[j])
                    j+=1
            else: 
                j+=1
    var_info_df = pd.DataFrame()
    var_info_df['seq_idxs']=seq_idxs
    var_info_df['variant_idxs'] = non_matching_indices
    var_info_df['A_sequence'] = string0_at_non_matching # in this case, A is B6, B is CAST 
    var_info_df['B_sequence'] = string1_at_non_matching
    var_info_df.to_csv(save_dir + 'variant_info.csv')

def alignments_onehot(seq1, seq2):
    seq1 = tools.reversed_onehot(seq1)
    seq2 = tools.reversed_onehot(seq2)
    alignments = pairwise2.align.globalxs(seq1, seq2, -0.5, -0.1)[0]
    seq1 = tools.onehot_encoding(alignments.seqA, len(alignments.seqA))
    seq2 = tools.onehot_encoding(alignments.seqB, len(alignments.seqB))
    return alignments[0], alignments[1], seq1, seq2 

def save_aligned_seqs(save_dir,seqs_path):        
    seqs_all = np.load(seqs_path)    
    seq_len = seqs_all.shape[1]
    genome_0_seqs = []
    genome_1_seqs = []
    for seq_idx in range(len(seqs_all)): 
        print(seq_idx)
        alignments_0, alignments_1, seq1, seq2 = alignments_onehot(seqs_all[seq_idx,:,:,0], seqs_all[seq_idx,:,:,1])
        genome_0_seqs.append(seq1[:seq_len,:])
        genome_1_seqs.append(seq2[:seq_len,:])
    genome_0_seqs = np.stack(genome_0_seqs, axis=0)
    genome_1_seqs = np.stack(genome_1_seqs, axis=0)
    genome_0_seqs = genome_0_seqs[..., np.newaxis]
    genome_1_seqs = genome_1_seqs[..., np.newaxis]
    comb_aligned = np.concatenate((genome_0_seqs, genome_1_seqs), axis=-1)
    np.save(save_dir + 'aligned_seqs', comb_aligned)

def get_ism(save_dir, seqs_path, ckpt_path, device):                     
    os.makedirs(save_dir,exist_ok=True)
    
    # first, get predictions for all ref seqs 
    if 'mh_predictions.txt' not in os.listdir(save_dir): 
        print('getting predictions for ref seqs')
        get_predictions(save_dir, ckpt_path, seqs_path,device=device)
    preds =  pd.read_csv(f'{save_dir}mh_predictions.txt',index_col=0,sep='\t')
    
    # if they have not already been saved, save variant info and aligned seqs to use for variant ism     
    if 'variant_info.csv' not in os.listdir(save_dir):
        print('saving variant info')
        save_var_info(save_dir, seqs_path)   

    if 'aligned_seqs.npy' not in os.listdir(save_dir):
        print('saving aligned seqs')
        save_aligned_seqs(save_dir,seqs_path)      
    
    device = torch.device('cuda:'+str(device) if torch.cuda.is_available() else "cpu")
    model = tools.load_saved_model(ckpt_path, mh_or_sh='mh')
    model.eval()
    model.to(device)
    var_info = pd.read_csv(save_dir + 'variant_info.csv',index_col=0)
    aligned_seqs = np.load(save_dir + 'aligned_seqs.npy')
    genome_labels = ['A','B']

    for genome_label in genome_labels:
        var_info[f'ratio_{genome_label}']=np.zeros(len(var_info))
        var_info[f'count_{genome_label}']=np.zeros(len(var_info))
    
    for genome_to_insert_idx in range(2): 
        print(f'genome_idx={genome_to_insert_idx}')
        other_genome_idx = 1-genome_to_insert_idx
        genome_label = genome_labels[genome_to_insert_idx]
        other_genome_label = genome_labels[other_genome_idx]
        
        for i in var_info.index:
            print(f'var_info_idx={i}')
            seq_idx = var_info.loc[i]['seq_idxs']
            print(f'seq_idx={seq_idx}')
                        
            count_ref_pred = preds.iloc[seq_idx][f'count_{genome_label}']
            ratio_ref_pred = preds.iloc[seq_idx]['ratioHead']
            
            curr_to_insert = aligned_seqs[[seq_idx],:,:,:].copy() # the aligned seqs at this seq idx
            var_position=var_info.loc[i]['variant_idxs']
            
            print(var_position)
            
            if len(var_position.split(':'))==1: # SNP
                start_insert = int(var_position)
                end_insert = int(var_position)+1
            else: # INDEL
                start_insert = int(var_position.split(':')[0])
                end_insert = int(var_position.split(':')[1])
            onehot_to_insert = tools.onehot_encoding(var_info.loc[i][f'{other_genome_label}_sequence'],len(var_info.loc[i][f'{other_genome_label}_sequence'])) # insert what's present in the other genome 
            
            curr_to_insert[:,start_insert:end_insert,:,genome_to_insert_idx] = onehot_to_insert

            # remove gaps and pad for both genomes in curr_to_insert 
            curr_to_insert[:,:,:,genome_to_insert_idx] = remove_gaps_and_pad(curr_to_insert[:,:,:,genome_to_insert_idx])
            curr_to_insert[:,:,:,other_genome_idx] = remove_gaps_and_pad(curr_to_insert[:,:,:,other_genome_idx])
            
            curr_to_insert = torch.from_numpy(curr_to_insert).to(device, dtype=torch.float)  
            out = model(curr_to_insert).cpu().detach().numpy()
            
            var_info.loc[i,f'ratio_{genome_label}'] = out[0,2] - ratio_ref_pred
            var_info.loc[i,f'count_{genome_label}'] = out[0,genome_to_insert_idx] - count_ref_pred
            
    var_info.to_csv(f'{save_dir}variant_ism_res.csv')
    
if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_dir")
    parser.add_argument("--seqs_path")
    parser.add_argument("--ckpt_path") 
    parser.add_argument("--device",default=0,type=int) 

    args = parser.parse_args()

    get_ism(args.save_dir, args.seqs_path, args.ckpt_path, args.device)                    

        
        
        
