
import functools

import torch


"""
    ToDo
        * from numpy to Tensor
        * chainer version ?
"""
def _apply_transform(tensors, **kwargs):
    device = kwargs.get('device', None)
    retain_comp_graph = kwargs.get('retain_comp_graph', True)
    if isinstance(tensors, torch.Tensor):
        if retain_comp_graph:
            return tensors.to(device)
        else:
            return tensors.detach().to(device)
    elif isinstance(tensors, dict):
        return { k:_apply_transform(v, **kwargs) for k,v in tensors.items() }
    elif isinstance(tensors, list):
        return [ _apply_transform(t, **kwargs) for t in tensors ]



"""
ToDo
    * rename this wrapper fucntion
    * multi-gpu ( maybe nothing to do ? )
    * allow two input types : dict or single Tensor
    * ? attach loss_fn to model (to consider)
    * automatic wrapping for nn.model
"""
def forward_wrap(func):
    @functools.wraps(func)
    def _forward_wrap(self, inputs, target_device=None, retain_comp_graph=True):
        # this is only valid for the case where model is put on a single device
        device = next(self.parameters()).device
        outputs = func(self, _apply_transform(inputs, device=device))
        if (target_device is None or target_device == device) and retain_comp_graph:# assume the single device for the model too
            return outputs
        else:
            # exception case needed if cannot be passed onto the device
            return _apply_transform(outputs, device=target_device, retain_comp_graph=retain_comp_graph)
    return _forward_wrap



# necessary ?
def input_default_wrapper(batch):
    return {"x":batch[0], "y":batch[1]}





def _compute_start_indices(batch_size, num_partition):
    min_par = int(batch_size / num_partition)
    assert min_par >= 1, "Too large partitions"
    num_more = batch_size - num_partition * min_par
    num_less = num_partition - num_more
    return [ (min_par+1)*i for i in range(num_more)] + [ min_par * i + num_more for i in range(num_more, num_more+num_less)] + [ batch_size ]


def _transpose_dict_to_list_applied_with_func(dict_, func):
    """
        inputs : { k->v, ... }
        outputs : [{ k->f(v)[0], ... }, { k->f(v)[1], ... }, ... ]
        Notice : func must return a list object
    """
    keys, values = zip(*(dict_.items()))
    return [ dict(zip(keys, v)) for v in zip(*map(func, values)) ]

def _transpose_list_to_list_applied_with_func(list_, func):
    return list(map(list, zip(*map(func, list_))))

def _transpose_tuple_to_list_applied_with_func(tuple_, func):
    return list(map(tuple, zip(*map(func, tuple_))))


import numpy as np
### working
def _partition_batch(batch, start_indices):
    """ rewritten idea
        try:
            import torch
            TENSORS.append(torch.Tensor)
        try:
            import cupy as cp
            TENSORS.append(cp.ndarray)
        if len(TENSORS) == 0:
            import numpy as np
            TENSORS.append(np.ndarray)
    """
    if isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        pair_indices = zip(start_indices[:-1], start_indices[1:])
        return [ batch[s_idx:e_idx] for s_idx,e_idx in pair_indices ]
    elif isinstance(batch, dict):
        return _transpose_dict_to_list_applied_with_func(batch, lambda x:_partition_batch(x, start_indices))
    elif isinstance(batch, tuple):
        return _transpose_tuple_to_list_applied_with_func(batch, lambda x:_partition_batch(x, start_indices))
    elif isinstance(batch, list):
        return _transpose_list_to_list_applied_with_func(batch, lambda x:_partition_batch(x, start_indices))


def _concat_results(results: list):
    r = results[0]
    if isinstance(r, torch.Tensor):
        return torch.cat(results, dim=0)
    elif isinstance(r, np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(r, dict):
        return { k : _concat_results([r[k] for r in results ]) for k in r.keys() }
    elif isinstance(r, tuple):
        return tuple(map(_concat_results, zip(*results)))
    elif isinstance(r, list):
        return list(map(_concat_results, zip(*results)))



