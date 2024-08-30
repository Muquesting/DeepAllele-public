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
from DeepAllele import model, data, tools, surrogate_model
import DeepAllele

CHIP_SEQ_LEN = 551
ATAC_SEQ_LEN = 330


def load_saved_model(ckpt_path,mh_or_sh): 
    if mh_or_sh == 'mh': 
        curr_model = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(ckpt_path)
    elif mh_or_sh == 'sh': 
        curr_model = model.SingleHeadResidualCNN.load_from_checkpoint(ckpt_path)
    curr_model.eval()
    return curr_model


def get_surrogate_model(ckpt_path):
    # Captum DeepLift does not allow for reuse of ReLU model, which is part of deepallele's architecture.  
    #  To deal with this, we create a surrogate model for interpretation with the same weights as a provided deepallele model, 
    #  but no resued ReLU. This surrogate model has identical predicton performance to the original. 
    
    # load original deepallele model from ckpt 
    orig = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(ckpt_path)
    surrogate = surrogate_model.SeparateMultiHeadResidualCNN_DeepliftSurrogate(
        kernel_number=orig.hparams['kernel_number'],
        kernel_length=orig.hparams['kernel_length'],
        filter_number=orig.hparams['filter_number'],
        kernel_size=orig.hparams['kernel_size'],
        pooling_size=orig.hparams['pooling_size'],
        input_length=orig.hparams['input_length'],
        conv_layers=orig.hparams['conv_layers'],
        hidden_size=orig.hparams['hidden_size'],
    )
    # transfer weights
    surrogate.conv0_b6.load_state_dict(orig.conv0.state_dict())
    for i in range(len(surrogate.convlayers_b6)):
        surrogate.convlayers_b6[i].load_state_dict(orig.convlayers[i].state_dict())

    surrogate.conv0_cast.load_state_dict(orig.conv0.state_dict())
    for i in range(len(surrogate.convlayers_cast)):
        surrogate.convlayers_cast[i].load_state_dict(orig.convlayers[i].state_dict())

    surrogate.fc0_b6.load_state_dict(orig.fc0.state_dict())
    for i in range(len(surrogate.fclayers_b6)):
        surrogate.fclayers_b6[i].load_state_dict(orig.fclayers[i].state_dict())

    surrogate.fc0_cast.load_state_dict(orig.fc0.state_dict())
    for i in range(len(surrogate.fclayers_b6)):
        surrogate.fclayers_cast[i].load_state_dict(orig.fclayers[i].state_dict())

    surrogate.counts_out_b6.load_state_dict(orig.counts_out.state_dict())
    surrogate.counts_out_cast.load_state_dict(orig.counts_out.state_dict())

    surrogate.ratio_out.load_state_dict(orig.ratio_out.state_dict())
    for i in range(len(surrogate.ratio_fclayers)):
        surrogate.ratio_fclayers[i].load_state_dict(orig.ratio_fclayers[i].state_dict())
    surrogate.eval()
    return surrogate


def get_attributions(x, model, baseline,attrib_type='deeplift',target_idx=2,multiply_by_inputs=False,n_steps=100,internal_batch_size=200): 
    # Note: x and baseline should be the same dimensions: for ex, [12,551,4]
    
    if attrib_type=='deeplift': 
        attrib_funct = DeepLift(model.eval(),multiply_by_inputs=multiply_by_inputs)
        attributions = attrib_funct.attribute(x, baseline, target=target_idx, return_convergence_delta=False)
        attributions = attributions.detach().cpu().numpy()

    if attrib_type=='ig': 
        ig = IntegratedGradients(model)
        attributions = ig.attribute(x,baselines=baseline,return_convergence_delta=False,target=target_idx,n_steps=n_steps,internal_batch_size=internal_batch_size)
        attributions = attributions.detach().cpu().numpy()
    
    if attrib_type=='grad': 
        input_seq = x.clone().requires_grad_(True)
        model_output = model(input_seq)[0]
        model_output[target_idx].backward(retain_graph=True)
        attributions = input_seq.grad.clone().cpu().numpy()
        
    return attributions 
    
def get_deeplift_res(save_dir, ckpt_path, seqs_path, save_label='', mh_or_sh='mh', num_shuffles=10, device=0, baseline_type='uniform', attrib_type='deeplift',subtract_means=True):
    
    os.makedirs(save_dir,exist_ok=True)
    seqs_all = np.load(seqs_path)    
    model = load_saved_model(ckpt_path, mh_or_sh)
    if mh_or_sh == 'mh': 
         model = get_surrogate_model(ckpt_path)
    model.to(device)

    if baseline_type =='b6-b6': 
        all_seqs_res = np.ones((seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq).to(device)
            b6_seq = seqs_all[[seq_idx],:,:,0]
            baseline = torch.Tensor(np.stack((b6_seq, b6_seq), axis=-1)).to(device)
            if mh_or_sh=='mh':
                deeplift_res =  get_attributions(tensor_seq,model,baseline, attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,:] = deeplift_res
            elif mh_or_sh=='sh':
                deeplift_res_0 = get_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0, attrib_type=attrib_type)
                deeplift_res_1 = get_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,0] = deeplift_res_0
                all_seqs_res[seq_idx,:,:,1] = deeplift_res_1
  
    if baseline_type =='uniform': 
        all_seqs_res = np.ones((seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            start = time.time()
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq).to(device)
            baseline = (torch.ones_like(tensor_seq)*0.25).to(device)
            if mh_or_sh=='mh':
                deeplift_res = get_attributions(tensor_seq,model,baseline,attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,:] = deeplift_res
            elif mh_or_sh=='sh':
                deeplift_res_0 = get_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0,attrib_type=attrib_type)
                deeplift_res_1 = get_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,0] = deeplift_res_0
                all_seqs_res[seq_idx,:,:,1] = deeplift_res_1
                
    elif baseline_type=='dinuc_shuffled': 
        all_seqs_res = np.ones((num_shuffles, seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            start = time.time()
            print(seq_idx)
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq).to(device)
            for shuffle_idx in range(num_shuffles):
                shuffled_0 = dinuc_shuffle(curr_seq[0,:,:,0])
                shuffled_1 = dinuc_shuffle(curr_seq[0,:,:,1])
                shuffled_0 = np.expand_dims(shuffled_0, axis=-1)
                shuffled_1 = np.expand_dims(shuffled_1, axis=-1)
                baseline = np.concatenate((shuffled_0, shuffled_1), axis=-1)
                baseline = torch.Tensor(np.expand_dims(baseline, axis=0)).to(device)
                if mh_or_sh=='mh':
                    deeplift_res = get_attributions(tensor_seq,model,baseline,attrib_type=attrib_type)
                    all_seqs_res[shuffle_idx,seq_idx,:,:,:] = deeplift_res
                elif mh_or_sh=='sh':
                    deeplift_res_0 = get_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0,attrib_type=attrib_type)
                    deeplift_res_1 = get_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type)
                    all_seqs_res[shuffle_idx,seq_idx,:,:,0] = deeplift_res_0
                    all_seqs_res[shuffle_idx,seq_idx,:,:,1] = deeplift_res_1
    
    if subtract_means: 
        mean = np.mean(all_seqs_res, axis=2)
        all_seqs_res = all_seqs_res - mean[:, :, np.newaxis, :] 

    np.save(f'{save_dir}{save_label}_deeplift_attribs', all_seqs_res) 

    
def get_ism_res(save_dir, ckpt_path, seqs_path, save_label='', mh_or_sh='mh', device=0, batch_size=200):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seqs_all = np.load(seqs_path)

    model = load_saved_model(ckpt_path, mh_or_sh)
    
    ism_res = np.zeros(seqs_all.shape)    
    trainer = pl.Trainer(gpus=[device])

    for seq_idx in range(seqs_all.shape[0]):
        print(seq_idx)
        start = time.time()
        curr_seq = seqs_all[[seq_idx],:,:,:]
        tensor_seq = torch.Tensor(curr_seq)
        ref_ratio = list(model(tensor_seq)[0].cpu().detach().numpy())[2]
        
        # collect alt seqs 
        zero_idxs = torch.where(tensor_seq[0,:,:,:]== 0) #tensor_seq[0,:,:,:] is [lenx4x2]
        pos_idxs = zero_idxs[0]
        nuc_idxs = zero_idxs[1]
        genome_idxs =  zero_idxs[2]
        
        # init alt seqs 
        alt_seqs = torch.clone(tensor_seq[0,:,:,:].expand([len(pos_idxs)] + list(tensor_seq[0,:,:,:].size()))) # alt seqs init as current seq with shape [num_altsxlenx4x2]
        
        # fill in alt seqs based on zero idxs 
        for alt_seq_idx in range(alt_seqs.shape[0]): 
            alt_seqs[alt_seq_idx,pos_idxs[alt_seq_idx],:,genome_idxs[alt_seq_idx]] = 0 # set all nucs to 0 
            alt_seqs[alt_seq_idx,pos_idxs[alt_seq_idx],nuc_idxs[alt_seq_idx],genome_idxs[alt_seq_idx]] = 1 # set curr nuc to 1 
        
        # predict for alt_seqs 
        dataset = TensorDataset(alt_seqs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=32)
        
        pred = trainer.predict(model,loader)
        pred = torch.cat(pred).detach().numpy()[:,2] # get ratio 

        # put in correct order in res 
        for alt_seq_idx in range(alt_seqs.shape[0]): 
            ism_res[seq_idx,pos_idxs[alt_seq_idx], nuc_idxs[alt_seq_idx], genome_idxs[alt_seq_idx]] = pred[alt_seq_idx] - ref_ratio
            
        end = time.time()
        print('one seq time')
        print(end - start)

        if seq_idx%1000 == 0: # don't save every time to improve speed 
            np.save(f'{save_dir}{save_label}_ism_res', ism_res) 
            
    # save again at the end 
    np.save(f'{save_dir}{save_label}_ism_res', ism_res) 



if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir")
    parser.add_argument("--save_label")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--seqs_path")
    parser.add_argument("--mh_or_sh",default='mh')
    parser.add_argument("--num_shuffles",default=10,type=int)
    parser.add_argument("--device",default=0,type=int)
    parser.add_argument("--baseline_type",default='uniform')
    parser.add_argument("--attrib_type",default='deeplift')
    parser.add_argument("--batch_size",default=200,type=int)
    parser.add_argument("--batch_id",default='')

    parser.add_argument("--which_fn")

    args = parser.parse_args()
    
    if args.which_fn=='get_deeplift_res':
        get_deeplift_res(args.save_dir, args.ckpt_path, args.seqs_path, save_label=args.save_label, mh_or_sh=args.mh_or_sh, num_shuffles=args.num_shuffles, device=args.device, baseline_type=args.baseline_type, attrib_type=args.attrib_type)
    if args.which_fn=='get_ism_res':
        get_ism_res(args.save_dir, args.ckpt_path, args.seqs_path, save_label=args.save_label, mh_or_sh=args.mh_or_sh, device=args.device, batch_size=args.batch_size)

    