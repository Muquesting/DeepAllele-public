import numpy as np 
import pandas as pd
import os
import h5py
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from DeepAllele import data, tools

def save_seqs_obs_labels(save_dir, save_label, hdf5_path, batch_id='sum', split_by_chrom=True, train_or_val='val'): 
    if batch_id=='':
        batch_id=None
    trainloader, valloader, train_feature, val_feature = data.load_h5(hdf5_path, 0.9, 32, batch_id=batch_id,split_by_chrom=split_by_chrom, shuffle=False)
    
    if train_or_val=='train':
        loader = trainloader
        feats = train_feature
    elif train_or_val=='val':
        loader = valloader
        feats = val_feature

    seqs_all = []
    obs_all = []
    for bid, (seqs, labels) in enumerate(valloader):
        seqs_all.append(seqs.cpu().numpy())
        obs_all.append(labels.cpu().numpy())
    seqs_all=np.concatenate(seqs_all)
    obs_all=np.concatenate(obs_all)
    
    np.save(f'{save_dir}{save_label}_{batch_id}_{train_or_val}_seqs', seqs_all)
    np.save(f'{save_dir}{save_label}_{batch_id}_{train_or_val}_obs', obs_all)
    np.save(f'{save_dir}{save_label}_{batch_id}_{train_or_val}_seq_labels', feats)

    
def get_predictions(save_dir, ckpt_path, seqs_path, save_label='', mh_or_sh='mh',device=0,batch_size=32,num_workers=32): 
    os.makedirs(save_dir,exist_ok=True)
    seqs_all = np.load(seqs_path)    
    model = tools.load_saved_model(ckpt_path, mh_or_sh)
    trainer = pl.Trainer(gpus=[device])
    placeholder_y = np.zeros((len(seqs_all),3))
    dataset = TensorDataset(torch.from_numpy(seqs_all), torch.from_numpy(placeholder_y)) # y is placeholder 
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    res = trainer.predict(model, loader)
    res=torch.cat(res).numpy()
    
    res_df = pd.DataFrame()
    res_df['count_A'] = res[:,0]
    res_df['count_B'] = res[:,1]
    res_df['ratioHead'] = res[:,2]
    res_df['ratio_count_A-B'] = res[:,0]-res[:,1]
    res_df.index.name = 'seq_idx'
    res_df.to_csv(f'{save_dir}{save_label}_{mh_or_sh}_predictions.txt', sep='\t', index=True)    
    
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