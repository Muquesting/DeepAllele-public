import numpy as np 
import pandas as pd
import os
import time
import argparse
from captum.attr import DeepLift
from captum.attr import IntegratedGradients
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from deeplift.dinuc_shuffle import dinuc_shuffle
from DeepAllele import model, tools, surrogate_model

def get_surrogate_model(ckpt_path):
    """
    Captum DeepLift does not allow for reuse of ReLU module, which is part of Deepallele's architecture. 
    To deal with this, we create a surrogate model for interpretation with the same weights as a provided deepallele model, 
    but no resued ReLU. This surrogate model has identical predicton performance to the original. 

    Parameters: 
    - ckpt_path: String path to ckeckpoint containing model paramters for DeepAllele (multi-head) model. 

    Returns: Surrogate model with identical performance to the one found at ckpt_path, but no reuse of ReLU module. 
    """
    # load original model from ckpt 
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


def get_gradient_based_attributions(x, model, baseline,attrib_type='deeplift',target_idx=2,multiply_by_inputs=False,n_steps=100,internal_batch_size=200): 
    """
    Get gradient-based model attributions for a given input sequence and model. 

    Parameters: 
    - x: Tensor model input of shape [batch_size, input length, 4] (4 is for the 4 nucleotides of the one-hot encoded sequence). 
    - model: DeepAllele model (either single-head or multi-head). 
    - baseline: Tensor baseline (only used if attrib_type is 'deeplift' or 'ig'). Baseline should be of the same shape as x: [batch_size, input length, 4]. 
    - attrib_type: String attribution type, of {'deeplift', 'ig', 'grad'}. 
    - target_idx: Integer index of the model output to use for getting attributions. 
    - multiply_by_inputs: Boolean input to DeepLift indicating whether local (if multiply_by_inputs=False) or global (if multiply_by_inputs=True) should be returned. 
    - n_steps: Integer input to IntegratedGradients. 
    - internal_batch_size: Integer input to IntegratedGradients. 

    Returns: Numpy array of attributions of shape [batch_size, input length, 4]. 
    """

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
    
def save_gradient_based_attributions(save_dir, ckpt_path, seqs_path, mh_or_sh='mh', num_shuffles=10, device=0, baseline_type='uniform', attrib_type='deeplift',subtract_means=True):
    """
    Save gradient-based model attributions for a given path to input sequences and model. 

    Parameters: 
    - save_dir: String path to directory in which to save attributions. 
    - ckpt_path: String path to DeepAllele model checkpoint. 
    - seqs_path: String path to sequences (saved as Numpy array). Array is of shape [n seqs, seq length, 4, 2]. The last dimension is 2 because the array contains sequences from 2 genomes. 
    - mh_or_sh: DeepAllele model type: mh is "multi-head" (SeparateMultiHeadResidualCNN), sh is "single-head" (SingleHeadResidualCNN). 
    - num_shuffles: Integer number of times to shuffle sequences to get baseline input. Only relevant if baseline_type=dinuc_shuffled. 
    - device: Integer of GPU to use. To use CPU, use device<0. 
    - baseline_type: Baseline to use, of {"seq_idx_0", "uniform", "dinuc_shuffled"}. Only relevant if attrib_type!="grad". "seq_idx_0" to use only the first of the paired sequence input as baseline (if "mh"), uniform to use 
        a sequence of 0.25, dinuc_shuffled to use shuffled sequence. 
    - attrib_type: String attribution type, of {'deeplift', 'ig', 'grad'}. 
    - subtract_means: Boolean, whether to zero-center the attributions. 

    Saves: Numpy array of attributions (of the same shape as the sequences found at seqs_path). 
    """
    
    os.makedirs(save_dir,exist_ok=True)
    seqs_all = np.load(seqs_path) 
    print(f'sequences shape: {seqs_all.shape}')
        
    model = tools.load_saved_model(ckpt_path, mh_or_sh)
    if mh_or_sh == 'mh': 
         model = get_surrogate_model(ckpt_path)

    if device>=0: 
        print(f'Using GPU {device}')
        model.to(device)
    else: 
        print('Using CPU')
        model.to('cpu')  

    if baseline_type =='seq_idx_0': 
        all_seqs_res = np.ones((seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            print(seq_idx)
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq)
            b6_seq = seqs_all[[seq_idx],:,:,0]
            baseline = torch.Tensor(np.stack((b6_seq, b6_seq), axis=-1))
            if device>=0: 
                tensor_seq=tensor_seq.to(device)
                baseline=baseline.to(device)
            if mh_or_sh=='mh':
                deeplift_res =  get_gradient_based_attributions(tensor_seq,model,baseline, attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,:] = deeplift_res
            elif mh_or_sh=='sh':
                deeplift_res_0 = get_gradient_based_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0, attrib_type=attrib_type)
                deeplift_res_1 = get_gradient_based_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,0] = deeplift_res_0
                all_seqs_res[seq_idx,:,:,1] = deeplift_res_1
  
    if baseline_type =='uniform': 
        all_seqs_res = np.ones((seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            print(seq_idx)
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq)
            baseline = (torch.ones_like(tensor_seq)*0.25)
            if device>=0: 
                tensor_seq=tensor_seq.to(device)
                baseline=baseline.to(device)
            if mh_or_sh=='mh':
                res = get_gradient_based_attributions(tensor_seq,model,baseline,attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,:] = res
            elif mh_or_sh=='sh':
                res_0 = get_gradient_based_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0,attrib_type=attrib_type)
                res_1 = get_gradient_based_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type)
                all_seqs_res[seq_idx,:,:,0] = res_0
                all_seqs_res[seq_idx,:,:,1] = res_1
                
    elif baseline_type=='dinuc_shuffled': 
        all_seqs_res = np.ones((num_shuffles, seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            print(seq_idx)
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq)
            if device>=0: 
                tensor_seq=tensor_seq.to(device)
            for shuffle_idx in range(num_shuffles):
                shuffled_0 = dinuc_shuffle(curr_seq[0,:,:,0])
                shuffled_1 = dinuc_shuffle(curr_seq[0,:,:,1])
                shuffled_0 = np.expand_dims(shuffled_0, axis=-1)
                shuffled_1 = np.expand_dims(shuffled_1, axis=-1)
                baseline = np.concatenate((shuffled_0, shuffled_1), axis=-1)
                baseline = torch.Tensor(np.expand_dims(baseline, axis=0))
                if device>=0: 
                    baseline=baseline.to(device)
                if mh_or_sh=='mh':
                    res = get_gradient_based_attributions(tensor_seq,model,baseline,attrib_type=attrib_type)
                    all_seqs_res[shuffle_idx,seq_idx,:,:,:] = res
                elif mh_or_sh=='sh':
                    res_0 = get_gradient_based_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0,attrib_type=attrib_type)
                    res_1 = get_gradient_based_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type)
                    all_seqs_res[shuffle_idx,seq_idx,:,:,0] = res_0
                    all_seqs_res[shuffle_idx,seq_idx,:,:,1] = res_1
    if subtract_means: 
        print('subtracting mean from each position')
        mean = np.mean(all_seqs_res, axis=2)
        all_seqs_res = all_seqs_res - mean[:, :, np.newaxis, :] 
    np.save(f'{save_dir}{mh_or_sh}_{attrib_type}_attribs', all_seqs_res) 

    
def save_ism_attributions(save_dir, ckpt_path, seqs_path, mh_or_sh='mh', device=0, batch_size=200,subtract_means=True):
    """
    Save ISM attributions for a given path to input sequences and model. 

    Parameters: 
    - save_dir: String path to directory in which to save attributions. 
    - ckpt_path: String path to DeepAllele model checkpoint. 
    - seqs_path: String path to sequences (saved as Numpy array). Array is of shape [n seqs, seq length, 4, 2]. The last dimension is 2 because the array contains sequences from 2 genomes. 
    - mh_or_sh: DeepAllele model type: mh is "multi-head" (SeparateMultiHeadResidualCNN), sh is "single-head" (SingleHeadResidualCNN). 
    - batch_size: Integer batch size for prediction. 
    - subtract_means: Boolean, whether to zero-center the attributions. 
    - device: Integer of GPU to use. To use CPU, use device<0. 

    Saves: Numpy array of ISM attributions (of the same shape as the sequences found at seqs_path). 
    
    """
    os.makedirs(save_dir, exist_ok=True)
    seqs_all = np.load(seqs_path)
    print(f'sequences shape: {seqs_all.shape}')

    model = tools.load_saved_model(ckpt_path, mh_or_sh)
    ism_res = np.zeros(seqs_all.shape)    

    if device>=0: 
        print(f'Using GPU {device}')
    else: 
        print('Using CPU')

    trainer = pl.Trainer(
    devices=[device] if device>=0 else None,  
    accelerator="gpu" if device>=0  else "cpu", 
    logger=False
)

    for seq_idx in range(seqs_all.shape[0]):
        print(seq_idx)
        start = time.time()
        curr_seq = seqs_all[[seq_idx],:,:,:]
        tensor_seq = torch.Tensor(curr_seq)

        if mh_or_sh=='mh':
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
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=8)
            pred = trainer.predict(model,loader)
            pred = torch.cat(pred).detach().numpy()[:,2] # get ratio 

            # put in correct order in res 
            for alt_seq_idx in range(alt_seqs.shape[0]): 
                ism_res[seq_idx,pos_idxs[alt_seq_idx], nuc_idxs[alt_seq_idx], genome_idxs[alt_seq_idx]] = pred[alt_seq_idx] 
        
        elif mh_or_sh=='sh':
            for genome_idx in [0,1]:
                curr_tensor_seq=tensor_seq[:,:,:,genome_idx]
                
                 # collect alt seqs 
                zero_idxs = torch.where(curr_tensor_seq[0,:,:]== 0) #tensor_seq[0,:,:,:] is [lenx4x2]
                pos_idxs = zero_idxs[0]
                nuc_idxs = zero_idxs[1]
                
                # init alt seqs 
                alt_seqs = torch.clone(curr_tensor_seq[0,:,:].expand([len(pos_idxs)] + list(curr_tensor_seq[0,:,:].size()))) # alt seqs init as current seq with shape [num_altsxlenx4]
    
                # fill in alt seqs based on zero idxs 
                for alt_seq_idx in range(alt_seqs.shape[0]): 
                    alt_seqs[alt_seq_idx,pos_idxs[alt_seq_idx],:] = 0 # set all nucs to 0 
                    alt_seqs[alt_seq_idx,pos_idxs[alt_seq_idx],nuc_idxs[alt_seq_idx]] = 1 # set curr nuc to 1 

                # predict for alt_seqs 
                placeholder_y = np.zeros((len(alt_seqs),1))
                dataset = TensorDataset(alt_seqs, torch.from_numpy(placeholder_y))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=8)
                pred = trainer.predict(model,loader)
                pred = torch.cat(pred).detach().numpy()[:,0,0] # [:,1,0] is obs

                # put in correct order in res 
                for alt_seq_idx in range(alt_seqs.shape[0]): 
                    ism_res[seq_idx,pos_idxs[alt_seq_idx], nuc_idxs[alt_seq_idx], genome_idx] = pred[alt_seq_idx] 

        end = time.time()
        print('single seq time')
        print(end - start)

        if seq_idx%1000 == 0: # don't save every time to improve speed 
            np.save(f'{save_dir}ism_res', ism_res) 
            
    if subtract_means: 
        print('subtracting mean from each position')
        mean = np.mean(ism_res, axis=2)
        ism_res = ism_res - mean[:, :, np.newaxis, :] 

    # save again at the end 
    np.save(f'{save_dir}ism_res', ism_res) 


if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--seqs_path")
    parser.add_argument("--mh_or_sh",default='mh')
    parser.add_argument("--num_shuffles",default=10,type=int)
    parser.add_argument("--device",default=0,type=int)
    parser.add_argument("--baseline_type",default='uniform')
    parser.add_argument("--attrib_type",default='grad')
    parser.add_argument("--batch_size",default=200,type=int)
    parser.add_argument("--subtract_means",default=1,type=bool)

    parser.add_argument("--which_fn")

    args = parser.parse_args()
    
    if args.which_fn=='save_gradient_based_attributions':
        save_gradient_based_attributions(save_dir=args.save_dir, ckpt_path=args.ckpt_path, seqs_path=args.seqs_path, mh_or_sh=args.mh_or_sh, num_shuffles=args.num_shuffles, device=args.device, baseline_type=args.baseline_type, attrib_type=args.attrib_type,subtract_means=args.subtract_means)
    
    if args.which_fn=='save_ism_attributions':
        save_ism_attributions(save_dir=args.save_dir, ckpt_path=args.ckpt_path, seqs_path=args.seqs_path, mh_or_sh=args.mh_or_sh, device=args.device, batch_size=args.batch_size,subtract_means=args.subtract_means)

    
