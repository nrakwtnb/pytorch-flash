


import functools
import torch


DEBUG = True


### utils


"""

Idea:
    * update_info -> turn on as context ?
"""
"""
def forward_wrap_imp(func):
    @functools.wraps(func)
    def _forward_wrap(self, inputs, target_device=None, retain_comp_graph=True, update_info=None, **kwargs):
        # this is only valid for the case where model is put on a single device
        device = next(self.parameters()).device# self.device ...
        if update_info is not None:
            Is_updated = self.name in update_info.get('updated_models', [self.name])
            Is_computed = self.name in update_info.get('computed_models', [self.name])
            if not Is_computed:
                return {}
            if not Is_updated:
                retain_comp_graph = False
        # > outputs = func(self, { k:v.to(device) for k,v in inputs.items() })
        outputs = func(self, _apply_transform(inputs, device=device), **kwargs)
        # > if retain_comp_graph:
        # >    outputs = outputs.detach()
        if (target_device is None or target_device == device) and retain_comp_graph:# assume the single device for the model too
            return outputs
        else:
            # exception case needed if cannot be passed onto the device
            # > return { k:v.to(target_device) for k,v in outputs.items() }
            return _apply_transform(outputs, device=target_device, retain_comp_graph=retain_comp_graph)
    return _forward_wrap
"""




import torch

"""
    * attach this func to the config class as its method
"""
def get_model(config, model_generator, model_args={}):
    model = model_generator(**model_args)
    
    if config["device"]["num_gpu"] > 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model.cuda(config["device"]["gpu"])
        config["device"]["name"] = device
    else:
        config["device"]["name"] = 'cpu'
    
    config["object"]["model"] = model



def wrap_metrics(func, get_y_values):
    def _wrap_metrics(results, *args, **kwargs):
        return func(*get_y_values(results, *args, **kwargs))
    return _wrap_metrics



# to rename the function later
def check(config):
    """
    if "grad_accumulation_steps" in config["other"].keys():
        grad_accumulation_steps = config["other"]["grad_accumulation_steps"]
        batch_size = config["train"]["batch_size"]
        train_batch_size = batch_size // grad_accumulation_steps
        assert train_batch_size * grad_accumulation_steps == batch_size
        config["train"]["train_batch_size"] = train_batch_size
    else:
        config["train"]["train_batch_size"] = config["train"]["batch_size"]
    """

    config["train"]["train_batch_size"] = config["train"]["batch_size"]

    if "object" not in config.keys():
        config["object"] = {}
        
    from collections import defaultdict
    config["object"].update({"metrics_log" : defaultdict(lambda :[])})



# default
def get_y_values(results):
    y_pred = results['outputs']['y']
    y_true = results['inputs']['y']
    return y_pred, y_true.to(y_pred.device)



def setup_metrics(config, output_transform, loss_fn):
    from ignite.metrics import Loss, Accuracy
    
    loss = wrap_metrics(loss_fn, output_transform)
    config["object"]["loss"] = loss
    
    metrics={
        'accuracy': Accuracy(output_transform=get_y_values),
        'loss': Loss(loss_fn, output_transform=get_y_values)
    }
    config["object"]["metrics"] = metrics



