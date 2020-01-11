
import numpy as np

import torch
from ignite.engine.engine import Engine

from flash.utils import _compute_start_indices, _partition_batch, _concat_results
from flash.utils import input_default_wrapper
from flash.utils import _apply_transform

from develop.flash.module import DataServer

DEBUG = False

"""
    working...
    * add ignore_final_batch if final_batch_size != batch_size
        + In this case, assert dataloader is not None
    * ? package args as config
"""
def create_trainer(process_list, batch_size=None, input_transform=None, **kwargs):
    grad_accumulation_steps = kwargs.get('grad_accumulation_steps', 1)
    num_batch_division = grad_accumulation_steps
    
    data_server = kwargs.get('data_server', None)# temporal
    
    if data_server is None:# temporal
        # multi gpu ?
        data_server = DataServer(num_batch_division=num_batch_division, default_input_size=batch_size, input_transform=input_transform)
    for process in process_list:
        process.set_data_server(data_server)
    
    def _update(engine, batch):
        data_server.set_up(batch)
        for N, process in enumerate(process_list, 1):
            process.run(state=engine.state)
            
        #return {"inputs":inputs, "outputs":outputs, "loss":loss}
        return data_server.get_results()

    engine = Engine(_update)
    engine.data_server = data_server
    return engine



def create_evaluator(process_list, metrics={}, input_transform=None, **kwargs):
    
    data_server = kwargs.get('data_server', None)# temporal
    
    if data_server is None:# temporal
        data_server = DataServer(input_transform=input_transform)
    for process in process_list:
        process.set_data_server(data_server)
        
    def _inference(engine, batch):
        data_server.set_up(batch)
        for N, process in enumerate(process_list, 1):
            process.run(state=engine.state)
            
        return data_server.get_results()

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    engine.data_server = data_server
    return engine


"""
    ToDo
        * ? add metrics evaluations
"""
def evaluation(model, dataloader, input_transform=None, **kwargs):
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

