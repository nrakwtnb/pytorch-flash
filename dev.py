


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






from ignite.engine.engine import Engine



"""
    working...
    * add ignore_final_batch if final_batch_size != batch_size
        + In this case, assert dataloader is not None
"""
from utils import _compute_start_indices, _partition_batch, _concat_results
from utils import input_default_wrapper
from utils import _apply_transform
import numpy as np
def create_trainer_imp(update_info_list,  data_loader, device=None, input_transform=input_default_wrapper, retain_comp_graph=False, Add_update_name_in_outputs=False, Add_update_name_in_loss=False, **kwargs):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            engine.to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    #if device:
    #    model.to(device)

    grad_accumulation_steps = kwargs.get('grad_accumulation_steps', 1)
    num_batch_division = grad_accumulation_steps
    if DEBUG:
        print("num_batch_division = ", num_batch_division)###
    # multi gpu ?

    dataset_size = len(data_loader.dataset)
    num_train_batches = len(data_loader)
    batch_size = data_loader.batch_size
    assert (dataset_size - 1) // batch_size + 1 == num_train_batches, "Invalid data_loader"
    #batch_size = train_batch_size * grad_accumulation_steps
    final_batch_iters = (dataset_size - 1) // batch_size * batch_size + 1
    final_batch_size = dataset_size - final_batch_iters + 1

    if num_batch_division > 1:
        start_indices_default = _compute_start_indices(batch_size, num_batch_division)
        batch_sizes_default = np.diff(start_indices_default)
        
        start_indices_final = [ idx for idx in start_indices_default if idx < final_batch_size ] + [final_batch_size]
        batch_sizes_final = np.diff(start_indices_final)

    def _update(engine, batch):

        inputs = input_transform(batch)
        outputs = {}
        if Add_update_name_in_loss:
            loss = {}
        else:
            loss = []
        for N, update_info in enumerate(update_info_list, 1):
            if 'skip_condition' in update_info:
                skip_condition = update_info['skip_condition']
                if skip_condition(engine.state):
                    continue

            model = update_info['model']
            optimizer = update_info['optimizer']
            loss_fn = update_info['loss_fn']
            
            model.train()# needed every time ? After calling evaluator run, back to train mode for example...
            
            if num_batch_division == 1:
                outputs_stage = model(inputs)
                loss_stage = loss_fn({"inputs":inputs, "outputs":outputs_stage})
                loss_stage.backward()
                if not retain_comp_graph:
                    loss_stage = loss_stage.detach()
            else:
                if engine.state.iteration % num_train_batches != 0:
                    start_indices = start_indices_default
                    batch_sizes = batch_sizes_default
                else:
                    start_indices = start_indices_final
                    batch_sizes = batch_sizes_final

                print(batch_sizes)###
                outputs_stage = []
                loss_stage = []
                for inputs_, bs in zip(_partition_batch(inputs, start_indices), batch_sizes):
                    outputs_stage_ = model(inputs_)
                    loss_stage_ = loss_fn({"inputs":inputs_, "outputs":outputs_stage_}) * bs / start_indices[-1]
                    loss_stage_.backward()
                    if not retain_comp_graph:
                        outputs_stage_ = _apply_transform(outputs_stage_, retain_comp_graph=False)
                        loss_stage_ = loss_stage_.detach()
                    outputs_stage.append(outputs_stage_)
                    loss_stage.append(loss_stage_)
                    print(loss_stage_)###
                outputs_stage = _concat_results(outputs_stage)
                loss_stage = sum(loss_stage)

            optimizer.step()
            optimizer.zero_grad()

            update_stage_name = update_info.get('name', str(N))
            if Add_update_name_in_outputs:
                outputs.update({ update_stage_name : outputs_stage })
            else:
                outputs.update(outputs_stage)
            if Add_update_name_in_loss:
                loss.update({ update_stage_name : loss_stage })
            else:
                loss.append(loss_stage)

        return {"inputs":inputs, "outputs":outputs, "loss":loss}

    return Engine(_update)




def create_evaluator_imp(evaluate_info_list, metrics={}, device=None, input_transform=input_default_wrapper, **kwargs):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    #if device:
    #    model.to(device)

    def _inference(engine, batch):
        for N, evaluate_info in enumerate(evaluate_info_list, 1):
            if 'skip_condition' in evaluate_info:
                skip_condition = evaluate_info['skip_condition']
                if skip_condition(engine.state):
                    continue

            model = evaluate_info['model']
            loss_fn = evaluate_info['loss_fn']

            model.eval()
            with torch.no_grad():
                inputs = input_transform(batch)
                outputs = model(inputs)
            return {"inputs":inputs, "outputs":outputs}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
