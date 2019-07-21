
import numpy as np

import torch
from ignite.engine.engine import Engine

from utils import _compute_start_indices, _partition_batch, _concat_results
from utils import input_default_wrapper
from utils import _apply_transform

DEBUG = False

"""
    working...
    * add ignore_final_batch if final_batch_size != batch_size
        + In this case, assert dataloader is not None
    * ? package args as config
"""
"""
Factory function for creating a trainer for supervised models.

Args:
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
def create_trainer(update_info_list,  data_loader, input_transform=input_default_wrapper, retain_comp_graph=False, Add_update_name_in_outputs=False, Add_update_name_in_loss=False, **kwargs):

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

                if DEBUG:
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
                    if DEBUG:
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
def create_evaluator(evaluate_info_list, metrics={}, input_transform=input_default_wrapper, Add_eval_name_in_outputs=False, **kwargs):

    def _inference(engine, batch):
        outputs = {}
        for N, evaluate_info in enumerate(evaluate_info_list, 1):
            if 'skip_condition' in evaluate_info:
                skip_condition = evaluate_info['skip_condition']
                if skip_condition(engine.state):
                    continue

            model = evaluate_info['model']
            #loss_fn = evaluate_info['loss_fn']

            model.eval()
            with torch.no_grad():
                inputs = input_transform(batch)
                outputs_stage = model(inputs)

            eval_stage_name = evaluate_info.get('name', str(N))
            if Add_eval_name_in_outputs:
                outputs.update({ evaluate_stage_name : outputs_stage })
            else:
                outputs.update(outputs_stage)
        return {"inputs":inputs, "outputs":outputs}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine



"""
    ToDo
        * ? add metrics evaluations
"""
def evaluation(model, dataloader, input_transform=input_default_wrapper, **kwargs):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs = input_transform(batch)
            outputs = model(inputs)
        if hasattr(engine.state, 'full_outputs'):
            engine.state.full_outputs.append(outputs)
        else:
            engine.state.__setattr__('full_outputs', [outputs])
        return {}
    engine = Engine(_inference)
    engine.run(dataloader)
    return _concat_results(engine.state.full_outputs)

