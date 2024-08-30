import numpy as np 
import pandas as pd
import os
import time
import h5py
import argparse
from captum.attr import DeepLift
from captum.attr import IntegratedGradients
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from deeplift.dinuc_shuffle import dinuc_shuffle

sys.path.insert(0, '/homes/gws/aspiro17/DeepAllele/')  # adding the path
from DeepAllele import model, data, tools, surrogate_model
import DeepAllele
sys.path.pop(0)  

def load_saved_model(ckpt_path,mh_or_sh): 
    if mh_or_sh == 'mh': 
        curr_model = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(ckpt_path)
    elif mh_or_sh == 'sh': 
        curr_model = model.SingleHeadResidualCNN.load_from_checkpoint(ckpt_path)
    curr_model.eval()
    return curr_model

def save_seqs_obs(save_dir, save_label, hdf5_path, batch_id='sum', split_by_chrom=True, train_or_val='val'): 
    if batch_id=='':
        batch_id=None
    trainloader, valloader, train_feature, val_feature = data.load_h5(hdf5_path, 0.9, 32, batch_id=batch_id,split_by_chrom=split_by_chrom, shuffle=False)
    
    if train_or_val=='train':
        loader = trainloader
    elif train_or_val=='val':
        loader = valloader

    seqs_all = []
    obs_all = []
    for bid, (seqs, labels) in enumerate(valloader):
        seqs_all.append(seqs.cpu().numpy())
        obs_all.append(labels.cpu().numpy())
    seqs_all=np.concatenate(seqs_all)
    obs_all=np.concatenate(obs_all)
    
    np.save(f'{save_dir}{save_label}_{batch_id}_{train_or_val}_seqs', seqs_all)
    np.save(f'{save_dir}{save_label}_{batch_id}_{train_or_val}_obs', obs_all)

    
def get_predictions(save_dir, ckpt_path, seqs_path, save_label='', mh_or_sh='mh',device=0): 
    os.makedirs(save_dir,exist_ok=True)
    seqs_all = np.load(seqs_path)    
    model = load_saved_model(ckpt_path, mh_or_sh)
    trainer = pl.Trainer(gpus=[device])
    
    dataset = TensorDataset(seqs_all, np.zeros(len(seqs_all))) # y is placeholder 
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32
    )
    res = trainer.predict(model, loader)
    res=torch.cat(res).numpy()
    save_dir = os.path.join(save_dir, atac_or_chip) + '/'
    np.save(f'{save_dir}{save_label}_{mh_or_sh}_predictions',res)
    
    
if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir")
    parser.add_argument("--save_label")
    parser.add_argument("--hdf5_path")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--seqs_path")
    parser.add_argument("--mh_or_sh",default='mh')
    parser.add_argument("--device",default=0,type=int)
    parser.add_argument("--batch_id",default='')
    parser.add_argument("--which_fn")
    parser.add_argument("--split_by_chrom",default=1,type=int)
    parser.add_argument("--train_or_val",default='val')

    args = parser.parse_args()
    
    if args.which_fn=='save_seqs_obs_labels':
        save_seqs_obs_labels(args.save_dir, args.save_label, args.hdf5_path, batch_id=args.batch_id, split_by_chrom=args.split_by_chrom, train_or_val=args.train_or_val)
    if args.which_fn=='get_predictions':
        get_predictions(args.save_dir, args.ckpt_path, args.seqs_path, save_label=args.save_label, mh_or_sh=args.mh_or_sh,device=args.device)