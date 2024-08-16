import numpy as np 
import pandas as pd
import os
import time
import h5py
import argparse
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import IntegratedGradients
import shap 
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl

#sys.path.insert(0, '/homes/gws/aspiro17/DeepAllele/')  # adding the path
from DeepAllele import model, data, tools,  surrogate_model 
import DeepAllele
#sys.path.pop(0)  

BASE_LIGHTNING_PATH = '/data/aspiro17/deepallele/leave_out_16_17_18/'
CHIP_SEQ_LEN = 551
ATAC_SEQ_LEN = 330

# dinuc shuffle functions from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)

def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")

def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens

def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


# TODO 
# Replace with argument pointing to file for parameters.

def load_saved_model(mh_or_sh = 'mh', sc_or_sum='sum', atac_or_chip='atac',pwk_or_spret='PWK',random_seed=0): 
    if atac_or_chip=='atac': 
        model_hyps = '4_1_256_15_256_5_4_0.0001_2_256_True'
        dir_name = 'ATAC_lightning_res'
        dataset_descriptor = sc_or_sum
    elif atac_or_chip=='chip': 
        model_hyps = '6_1_256_15_256_5_2_0.0001_2_256_True'
        dir_name = 'ChIP_lightning_res'
        dataset_descriptor = pwk_or_spret

    if mh_or_sh == 'mh': 
        path = os.path.join(BASE_LIGHTNING_PATH, dir_name, dataset_descriptor, mh_or_sh, model_hyps, 'random_seed_'+str(random_seed)) + '/'

    elif mh_or_sh == 'sh': 
        path = os.path.join(BASE_LIGHTNING_PATH, dir_name, dataset_descriptor, mh_or_sh, model_hyps,'B6/' 'random_seed_'+str(random_seed)) + '/'

    # get model name
    model_name = ''
    for item in os.listdir(path):
        if 'epoch' in item: 
            model_name=item 
    checkpoint_path = path + model_name

    if mh_or_sh == 'mh': 
        curr_model = model.SeparateMultiHeadResidualCNN.load_from_checkpoint(checkpoint_path)
        
    elif mh_or_sh == 'sh': 
        curr_model = model.SingleHeadResidualCNN.load_from_checkpoint(checkpoint_path)
    curr_model.eval()
    return curr_model

# We should have a file to define this
def get_surrogate_model(mh_model, atac_or_chip='atac'):
    if atac_or_chip == 'atac': 
        surrogate = model.SeparateMultiHeadResidualCNN_DeepliftSurrogate(
        kernel_number=256,
        kernel_length=15,
        filter_number=256,
        kernel_size=5,
        pooling_size=4,
        input_length=ATAC_SEQ_LEN*2)
    
    elif atac_or_chip == 'chip': 
        surrogate = model.SeparateMultiHeadResidualCNN_DeepliftSurrogate(
        conv_layers=6,
        kernel_number=256,
        kernel_length=15,
        filter_number=256,
        kernel_size=5,
        pooling_size=2,
        input_length=CHIP_SEQ_LEN*2)

    # transfer weights
    surrogate.conv0_b6.load_state_dict(mh_model.conv0.state_dict())
    for i in range(len(surrogate.convlayers_b6)):
        surrogate.convlayers_b6[i].load_state_dict(mh_model.convlayers[i].state_dict())

    surrogate.conv0_cast.load_state_dict(mh_model.conv0.state_dict())
    for i in range(len(surrogate.convlayers_cast)):
        surrogate.convlayers_cast[i].load_state_dict(mh_model.convlayers[i].state_dict())

    surrogate.fc0_b6.load_state_dict(mh_model.fc0.state_dict())
    for i in range(len(surrogate.fclayers_b6)):
        surrogate.fclayers_b6[i].load_state_dict(mh_model.fclayers[i].state_dict())

    surrogate.fc0_cast.load_state_dict(mh_model.fc0.state_dict())
    for i in range(len(surrogate.fclayers_b6)):
        surrogate.fclayers_cast[i].load_state_dict(mh_model.fclayers[i].state_dict())

    surrogate.counts_out_b6.load_state_dict(mh_model.counts_out.state_dict())
    surrogate.counts_out_cast.load_state_dict(mh_model.counts_out.state_dict())

    surrogate.ratio_out.load_state_dict(mh_model.ratio_out.state_dict())
    for i in range(len(surrogate.ratio_fclayers)):
        surrogate.ratio_fclayers[i].load_state_dict(mh_model.ratio_fclayers[i].state_dict())
    surrogate.eval()
    return surrogate



def get_predictions(save_dir, atac_or_chip='atac',random_seed=0, mh_or_sh='mh', sc_or_sum='sum',device=0):    
    os.makedirs(save_dir,exist_ok=True)
    
    # load seqs
    if atac_or_chip=='atac':
        data_path= '/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5'
        trainloader, valloader, train_feature, val_feature = data.load_h5(data_path, 0.9, 32, batch_id='sum',split_by_chrom=True, shuffle=False)
  
    elif atac_or_chip=='chip':
        data_path= '/data/tuxm/project/F1-ASCA/data/input/Chip-seq/processed_data/sequence_datasets_chip_PWK_B6_20230126.hdf5'
        trainloader, valloader, train_feature, val_feature = data.load_h5(data_path, 0.9, 32, split_by_chrom=True, shuffle=False)
  
    seqs_all = []
    for batch_id, (seqs, labels) in enumerate(valloader):
        seqs_all.append(seqs.cpu().numpy())
    seqs_all=np.concatenate(seqs_all)
    
    print(seqs_all.shape)
    
    model = load_saved_model(mh_or_sh = mh_or_sh, sc_or_sum=sc_or_sum, atac_or_chip=atac_or_chip,random_seed=random_seed)
    
    if mh_or_sh == 'mh': 
         model = get_surrogate_model(model, atac_or_chip=atac_or_chip)
            
    trainer = pl.Trainer(gpus=[device])
    res = trainer.predict(model, valloader)
    res=torch.cat(res).numpy()
    save_dir = os.path.join(save_dir, atac_or_chip) + '/'
    np.save(save_dir + 'mh_rand_seed='+str(random_seed) + '_predictions',res)

    
def get_attributions(x, model, baseline,target_idx=2,eps=1e-6,multiply_by_inputs=False,attrib_type='deeplift',num_baseline_repeats=3,n_steps=100,internal_batch_size=200): 
    
    # x and baseline should be passed in as the same dimens: for ex, [12,551,4]
    # for Captum DeepLiftShap, cannot pass in single baseline     
    
    if attrib_type=='deeplift': 
        attrib_funct = DeepLift(model.eval(),eps=eps,multiply_by_inputs=multiply_by_inputs)
        attributions, delta = attrib_funct.attribute(x, baseline, target=target_idx, return_convergence_delta=True)
        attributions = attributions.detach().cpu().numpy()
        
    if attrib_type=='deepliftshap': 
        if baseline.shape[0] == 1:
            baseline = torch.cat([baseline] * num_baseline_repeats, dim=0) # cannot use a single baseline with deepliftshap, repeating baseline
        attrib_funct = DeepLiftShap(model.eval(),multiply_by_inputs=multiply_by_inputs)
        attributions, delta = attrib_funct.attribute(x, baseline, target=target_idx, return_convergence_delta=True)
        attributions = attributions.detach().cpu().numpy()
        
    if attrib_type=='jacobdeepliftshap': 
        unsqueezed_baseline = baseline.unsqueeze(0) # add extra 1 dimen in front 
        attrib_funct = JacobDeepLiftShap(model.eval(),eps=eps)
        attributions, delta = attrib_funct.attribute(x, unsqueezed_baseline)
        attributions = attributions.detach().cpu().numpy()

    if attrib_type=='shap_deepexplainer': 
        e = shap.DeepExplainer(model, baseline)
        attributions = e.shap_values(x)
        if target_idx!=0: 
            attributions=attributions[target_idx] # for mh 
            
    if attrib_type=='ig': 
        ig = IntegratedGradients(model)
        attributions = ig.attribute(x,baselines=baseline,return_convergence_delta=False,target=target_idx,n_steps=n_steps,internal_batch_size=internal_batch_size)
        attributions = attributions.detach().cpu().numpy()
    
    if attrib_type=='grad': 
        input_seq = x.clone().requires_grad_(True)
        model_output = model(input_seq)[0]
        model_output[target_idx].backward(retain_graph=True)
        attributions = input_seq.grad.clone().cpu().numpy()
        
    model_diff = model(x).detach().cpu().numpy()[0][target_idx] - model(baseline).detach().cpu().numpy()[0][target_idx]
    delta = np.abs(model_diff - attributions.sum())
    
    return attributions, delta 


def get_deeplift_res(save_dir, atac_or_chip,random_seed, mh_or_sh, sc_or_sum,num_shuffles,device,baseline_type,eps,attrib_type,n_steps):

    os.makedirs(save_dir,exist_ok=True)
    
    # load seqs
    if atac_or_chip=='atac':
        data_path= '/data/tuxm/project/F1-ASCA/data/input/bulk_seq_ATAC_preprocessed_new_20230126.hdf5'
        trainloader, valloader, train_feature, val_feature = data.load_h5(data_path, 0.9, 32, batch_id='sum',split_by_chrom=True, shuffle=False)
  
    elif atac_or_chip=='chip':
        data_path= '/data/tuxm/project/F1-ASCA/data/input/Chip-seq/processed_data/sequence_datasets_chip_PWK_B6_20230126.hdf5'
        trainloader, valloader, train_feature, val_feature = data.load_h5(data_path, 0.9, 32, split_by_chrom=True, shuffle=False)
  
    seqs_all = []
    for batch_id, (seqs, labels) in enumerate(valloader):
        seqs_all.append(seqs.cpu().numpy())
    seqs_all=np.concatenate(seqs_all)
    
    print(seqs_all.shape)
    
    model = load_saved_model(mh_or_sh = mh_or_sh, sc_or_sum=sc_or_sum, atac_or_chip=atac_or_chip,random_seed=random_seed)
    
    if mh_or_sh == 'mh': 
         model = get_surrogate_model(model, atac_or_chip=atac_or_chip)
            
    model.to(device)
    deltas_res = []
    
    if baseline_type =='b6-b6': 
        all_seqs_res = np.ones((seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq).to(device)
            b6_seq = seqs_all[[seq_idx],:,:,0]
            baseline = torch.Tensor(np.stack((b6_seq, b6_seq), axis=-1)).to(device)
            if mh_or_sh=='mh':
                deeplift_res, delta =  get_attributions(tensor_seq,model,baseline, attrib_type=attrib_type,n_steps=n_steps)
                all_seqs_res[seq_idx,:,:,:] = deeplift_res
                deltas_res.append(delta)
            elif mh_or_sh=='sh':
                deeplift_res_0, delta_0 = get_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0, attrib_type=attrib_type,n_steps=n_steps)
                deeplift_res_1, delta_1 = get_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type,n_steps=n_steps)
                all_seqs_res[seq_idx,:,:,0] = deeplift_res_0
                all_seqs_res[seq_idx,:,:,1] = deeplift_res_1
                deltas_res.append(delta_0)
                deltas_res.append(delta_1)
  
    if baseline_type =='uniform': 
        all_seqs_res = np.ones((seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            start = time.time()
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq).to(device)
            baseline = (torch.ones_like(tensor_seq)*0.25).to(device)
            if mh_or_sh=='mh':
                deeplift_res, delta = get_attributions(tensor_seq,model,baseline,attrib_type=attrib_type,n_steps=n_steps)
                all_seqs_res[seq_idx,:,:,:] = deeplift_res
                deltas_res.append(delta)
            elif mh_or_sh=='sh':
                deeplift_res_0, delta_0 = get_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0,attrib_type=attrib_type,n_steps=n_steps)
                deeplift_res_1, delta_1 = get_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type,n_steps=n_steps)
                all_seqs_res[seq_idx,:,:,0] = deeplift_res_0
                all_seqs_res[seq_idx,:,:,1] = deeplift_res_1
                deltas_res.append(delta_0)
                deltas_res.append(delta_1)

    elif baseline_type=='dinuc_shuffled': 
        all_seqs_res = np.ones((num_shuffles, seqs_all.shape[0], seqs_all.shape[1], seqs_all.shape[2], seqs_all.shape[3]))*-1
        for seq_idx in range(seqs_all.shape[0]):
            start = time.time()
            print(seq_idx)
            curr_seq = seqs_all[[seq_idx],:,:,:]
            tensor_seq = torch.Tensor(curr_seq).to(device)
            for shuffle_idx in range(num_shuffles):
                # baseline for both genomes  
                shuffled_0 = dinuc_shuffle(curr_seq[0,:,:,0])
                shuffled_1 = dinuc_shuffle(curr_seq[0,:,:,1])
                shuffled_0 = np.expand_dims(shuffled_0, axis=-1)
                shuffled_1 = np.expand_dims(shuffled_1, axis=-1)
                baseline = np.concatenate((shuffled_0, shuffled_1), axis=-1)
                baseline = torch.Tensor(np.expand_dims(baseline, axis=0)).to(device)
                if mh_or_sh=='mh':
                    deeplift_res,delta = get_attributions(tensor_seq,model,baseline,attrib_type=attrib_type,n_steps=n_steps)
                    all_seqs_res[shuffle_idx,seq_idx,:,:,:] = deeplift_res
                    deltas_res.append(delta)
                elif mh_or_sh=='sh':
                    deeplift_res_0,delta_0 = get_attributions(tensor_seq[:,:,:,0],model,baseline[:,:,:,0],target_idx=0,attrib_type=attrib_type,n_steps=n_steps)
                    deeplift_res_1,delta_1 = get_attributions(tensor_seq[:,:,:,1],model,baseline[:,:,:,1],target_idx=0,attrib_type=attrib_type,n_steps=n_steps)
                    all_seqs_res[shuffle_idx,seq_idx,:,:,0] = deeplift_res_0
                    all_seqs_res[shuffle_idx,seq_idx,:,:,1] = deeplift_res_1
                    deltas_res.append(delta_0)
                    deltas_res.append(delta_1)
            end = time.time()
            print('one seq time')
            print(end-start)
    np.save(save_dir + 'deeplift_attribs', all_seqs_res) 
    np.save(save_dir + 'deeplift_deltas', deltas_res)


    
        
class JacobDeepLiftShap():
	"""A vectorized version of the DeepLIFT/SHAP algorithm from Captum.

	This approach is based on the Captum approach of assigning hooks to
	layers that modify the gradients to implement the rescale rule. This
	implementation is vectorized in a manner that can accept unique references
	for each example to be explained as well as multiple references for each
	example.

	The implementation is minimal and currently only supports the operations
	used in bpnet-lite. This is not meant to be a general-purpose implementation
	of the algorithm and may not work with custom architectures.
	

	Parameters
	----------
	model: bpnetlite.BPNet or bpnetlite.ChromBPNet
		A BPNet or ChromBPNet module as implemented in this repo.

	attribution_func: function or None, optional
		This function is used to aggregate the gradients after calculation.
		Useful when trying to handle the implications of one-hot encodings. If
		None, return the gradients as calculated. Default is None.

	eps: float, optional
		An epsilon with which to threshold gradients to ensure that there
		isn't an explosion. Default is 1e-10.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the attribution_func being applied to them. Default 
		is 0.001. 

	verbose: bool, optional
		Whether to print the convergence delta for each example that is
		explained, regardless of whether it surpasses the warning threshold.
		Note that convergence deltas are calculated on the gradients prior to 
		the attribution_func being applied to them. Default is False.
	"""

	def __init__(self, model, attribution_func=None, eps=1e-6, 
		warning_threshold=0.001, verbose=False):
		for module in model.named_modules():
			if isinstance(module[1], torch.nn.modules.pooling._MaxPoolNd):
				raise ValueError("Cannot use this implementation of " + 
					"DeepLiftShap with max pooling layers. Please use the " +
					"implementation in Captum.")

		self.model = model
		self.attribution_func = attribution_func
		self.eps = eps
		self.warning_threshold = warning_threshold
		self.verbose = verbose

		self.forward_handles = []
		self.backward_handles = []

	def attribute(self, inputs, baselines, args=None):
		assert inputs.shape[1:] == baselines.shape[2:]
		n_inputs, n_baselines = baselines.shape[:2]

		inputs = inputs.repeat_interleave(n_baselines, dim=0).requires_grad_()
		baselines = baselines.reshape(-1, *baselines.shape[2:]).requires_grad_()

		if args is not None:
			args = (arg.repeat_interleave(n_baselines, dim=0) for arg in args)
		else:
			args = None

		###

		try:
			self.model.apply(self._register_hooks)
			inputs_ = torch.cat([inputs, baselines])

			# Calculate the gradients using the rescale rule
			with torch.autograd.set_grad_enabled(True):
				if args is not None:
					args = (torch.cat([arg, arg]) for arg in 
						args)
					outputs = self.model(inputs_, *args)
				else:
					outputs = self.model(inputs_)

				outputs_ = torch.chunk(outputs, 2)[0].sum()
				gradients = torch.autograd.grad(outputs_, inputs)[0]

			output_diff = torch.sub(*torch.chunk(outputs[:,0], 2))
			input_diff = torch.sum((inputs - baselines) * gradients, dim=(1, 2)) 
			convergence_deltas = abs(output_diff - input_diff)
            #convergence_deltas = output_diff - input_diff
			#print(convergence_deltas)

#			if any(convergence_deltas > self.warning_threshold):
#				warnings.warn("Convergence deltas too high: " +   
#					str(convergence_deltas))

			if self.verbose:
				print(convergence_deltas)

			# Process the gradients to get attributions
			if self.attribution_func is None:
				attributions = gradients
			else:
				attributions = self.attribution_func((gradients,), (inputs,), 
					(baselines,))[0]

		finally:
			for forward_handle in self.forward_handles:
				forward_handle.remove()
			for backward_handle in self.backward_handles:
				backward_handle.remove()

		###

		attr_shape = (n_inputs, n_baselines) + attributions.shape[1:]
		attributions = torch.mean(attributions.view(attr_shape), dim=1, 
			keepdim=False)
		return attributions, convergence_deltas

	def _forward_pre_hook(self, module, inputs):
		module.input = inputs[0].clone().detach()

	def _forward_hook(self, module, inputs, outputs):
		module.output = outputs.clone().detach()

	def _backward_hook(self, module, grad_input, grad_output):
		delta_in_ = torch.sub(*module.input.chunk(2))
		delta_out_ = torch.sub(*module.output.chunk(2))

		delta_in = torch.cat([delta_in_, delta_in_])
		delta_out = torch.cat([delta_out_, delta_out_])

		delta = delta_out / delta_in

		grad_input = (torch.where(
			abs(delta_in) < self.eps, grad_input[0], grad_output[0] * delta),
		)
		return grad_input

	def _can_register_hook(self, module):
		if len(module._backward_hooks) > 0:
			return False
		if not isinstance(module, (torch.nn.ReLU, _ProfileLogitScaling)):
			return False
		return True

	def _register_hooks(self, module, attribute_to_layer_input=True):
		if not self._can_register_hook(module) or (
			not attribute_to_layer_input and module is self.layer
		):
			return

		# adds forward hook to leaf nodes that are non-linear
		forward_handle = module.register_forward_hook(self._forward_hook)
		pre_forward_handle = module.register_forward_pre_hook(self._forward_pre_hook)
		backward_handle = module.register_full_backward_hook(self._backward_hook)

		self.forward_handles.append(forward_handle)
		self.forward_handles.append(pre_forward_handle)
		self.backward_handles.append(backward_handle)
        
        
class _ProfileLogitScaling(torch.nn.Module):
	"""This ugly class is necessary because of Captum.

	Captum internally registers classes as linear or non-linear. Because the
	profile wrapper performs some non-linear operations, those operations must
	be registered as such. However, the inputs to the wrapper are not the
	logits that are being modified in a non-linear manner but rather the
	original sequence that is subsequently run through the model. Hence, this
	object will contain all of the operations performed on the logits and
	can be registered.


	Parameters
	----------
	logits: torch.Tensor, shape=(-1, -1)
		The logits as they come out of a Chrom/BPNet model.
	"""

	def __init__(self):
		super(_ProfileLogitScaling, self).__init__()

	def forward(self, logits):
		y = torch.nn.functional.log_softmax(logits, dim=-1)
		y = logits * torch.exp(y).detach()
		return y
        
        
        
    
if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()

    # setup
    parser.add_argument("--save_dir")
    parser.add_argument("--device",default=0,type=int)
    parser.add_argument("--which_funct",default='get_deeplift_res') 
    parser.add_argument("--atac_or_chip",type=str,default='chip') 
    parser.add_argument("--random_seed", default=0,type=int) 
    parser.add_argument("--mh_or_sh",default='mh') 
    parser.add_argument("--sc_or_sum",default='sum') 
    parser.add_argument("--num_shuffles",default=20,type=int) 
    parser.add_argument("--baseline_type",default='uniform') 
    parser.add_argument("--eps",default=1e-6,type=float) 
    parser.add_argument("--attrib_type",default='deeplift') 
    parser.add_argument("--n_steps",type=int, default=50) 
  
    args = parser.parse_args()
    
    if args.which_funct=='get_deeplift_res':
        if args.attrib_type == 'ig': 
            args.save_dir = os.path.join(args.save_dir, args.atac_or_chip, 'eps='+str(args.eps), args.baseline_type+'_baseline', args.mh_or_sh, str(args.random_seed), args.attrib_type, 'n_steps='+str(args.n_steps)) + '/'
        else: 
            args.save_dir = os.path.join(args.save_dir, args.atac_or_chip, 'eps='+str(args.eps), args.baseline_type+'_baseline', args.mh_or_sh, str(args.random_seed), args.attrib_type) + '/'
                
        get_deeplift_res(save_dir=args.save_dir, atac_or_chip=args.atac_or_chip,random_seed=args.random_seed, mh_or_sh=args.mh_or_sh, sc_or_sum=args.sc_or_sum,num_shuffles=args.num_shuffles,device=args.device,baseline_type=args.baseline_type,eps=args.eps, attrib_type=args.attrib_type,n_steps=args.n_steps)
        
    if args.which_funct=='get_preds': 
        
        get_predictions(save_dir=args.save_dir, atac_or_chip=args.atac_or_chip,random_seed=args.random_seed, mh_or_sh=args.mh_or_sh, sc_or_sum=args.sc_or_sum,device=args.device)

        

