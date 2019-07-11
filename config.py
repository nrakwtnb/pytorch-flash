
from collections import defaultdict

import torch
from ignite.metrics import Accuracy

from utils import wrap_metrics, get_y_values
from metrics import Loss


"""
    * __repr__
    * __str__
    * __dict__
    * __getter__
    * __setter__
"""
class Config():
    def __init__(self):
        pass

    def load_config(self):
        pass

    def set_config(self, config):
        self.config = config
        self.check()

    def add_model_from_generator(self, model_name, model_generator, model_args={}, device=None):
        assert isinstance(model_name ,str)
        config = self.config
        model = model_generator(**model_args)
        device = config['device']['name'] if device is None else device
        if device == 'cuda':
            model.cuda(config['device']['gpu'])
        config['objects']['models'].update({model_name : model})
        
        """
        if config['device']['num_gpu'] > 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model.cuda(config['device']['gpu'])
            config['device']['name'] = device
        else:
            config['device']['name'] = 'cpu'
        
        config['objects']['model'] = model
        """

    def add_model(self, model_name, model, device=None):
        assert isinstance(model_name ,str)
        config = self.config
        config['objects']['models'].update({model_name : model})

    def add_optimizer(self, optimizer_name, optimizer):
        assert isinstance(optimizer_name ,str)
        config = self.config
        config['objects']['optimizers'].update({optimizer_name : optimizer})

    def add_loss_fn(self, loss_fn_name, loss_fn, output_transform=get_y_values):
        assert isinstance(loss_fn_name ,str)
        config = self.config
        loss_fn_ = wrap_metrics(loss_fn, output_transform)
        #config['objects']['loss_fns_info'].update({loss_fn_name : (loss_fn, output_transform)})
        config['objects']['loss_fns'].update({loss_fn_name : loss_fn_})

    def add_update_info(self, **update_info):
        assert 'model' in update_info
        assert 'loss_fn' in update_info
        assert 'optimizer' in update_info
        assert all(map(lambda x:isinstance(x, str), update_info.values()))
        config = self.config
        update_info_list = config['trainer']['update_info_list']
        update_info_list.append(update_info)
        
    def add_evaluate_info(self, **evaluate_info):
        assert 'model' in evaluate_info
        assert 'loss_fn' in evaluate_info
        assert all(map(lambda x:isinstance(x, str), evaluate_info.values()))
        config = self.config
        evaluate_info_list = config['trainer']['evaluate_info_list']
        evaluate_info_list.append(evaluate_info)

    #def setup_metrics(self, loss_fn, output_transform=get_y_values):
    def setup_metrics(self, output_transform=get_y_values, target_loss_fn_names=None):## working
        """
            ToDo
                * Add other metrics ...
        """
        config = self.config
        loss_fns = self.config['objects']['loss_fns']

        metrics = {}
        if target_loss_fn_names is None:
            if len(loss_fns) == 1:
                loss_fn = list(loss_fns.values())[0]
                metrics.update({ 'loss' : Loss(loss_fn) })###
            else:
                target_loss_fn_names = list(loss_fns.keys())
        if target_loss_fn_names is not None:
            for name, loss_fn in loss_fns.items():
                metrcis.update({ f"{name}-loss" : Loss(loss_fn) })###

        """
        if target_eval_stage_names is None:
            if len(evaluate_info_list) == 1:
                loss_fn = evaluate_info_list[0]['loss_fn']
                metrics.update({ 'loss' : Loss(loss_fn, output_transform=output_transform) })
            else:
                target_eval_stage_names = map(str, range(1, len()+1))
        if target_eval_stage_names is not None:
            for N, evaluate_info in enumerate(evaluate_info_list, 1):
                eval_stage_name = evaluate_info.get('name', str(N))
                if eval_stage_name in target_eval_stage_names:
                    loss_fn = evaluate_info['metrics_loss_fn']
                    metrcis.update({ f"{eval_stage_name}-loss" : Loss(loss_fn, output_transform=output_transform) })
        """

        metrics.update({ 'accuracy' : Accuracy(output_transform=output_transform) })
        config['objects']['metrics'] = metrics

    # to rename the function later
    def check(self):
        config = self.config
        """
        if "grad_accumulation_steps" in config['others'].keys():
            grad_accumulation_steps = config['others']['grad_accumulation_steps']
            batch_size = config['train']['batch_size']
            train_batch_size = batch_size // grad_accumulation_steps
            assert train_batch_size * grad_accumulation_steps == batch_size
            config['train']['train_batch_size'] = train_batch_size
        else:
            config['train']['train_batch_size'] = config['train']['batch_size']
        """

        config['train']['train_batch_size'] = config['train']['batch_size']

        ### change ?
        if config['device']['num_gpu'] > 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model.cuda(config['device']['gpu'])
            config['device']['name'] = device
        else:
            config['device']['name'] = 'cpu'

        if 'objects' not in config.keys():
            config['objects'] = {}
        if 'trainer' not in config.keys():
            config['trainer'] = {}
            
        config['objects'].update({'models' : {}})
        config['objects'].update({'loss_fns' : {}})
        #config['objects'].update({'metrics_loss_fns' : {}})
        config['objects'].update({'optimizers' : {}})
        config['objects'].update({'metrics_log' : defaultdict(lambda :[])})
        config['trainer'].update({'update_info_list' : []})
        config['trainer'].update({'evaluate_info_list' : []})

    def set_objects(self):
        import copy
        config = self.config
        trainer = config['trainer']
        objects = config['objects']
        key2obj = {
            'model' : objects['models'],
            'loss_fn' : objects['loss_fns'],
            'optimizer' : objects['optimizers']
        }

        update_info_list = []
        for update_info in trainer['update_info_list']:
            update_info_ = copy.copy(update_info)
            for key, obj in key2obj.items():
                update_info_[key] = obj[update_info[key]]
            update_info_list.append(update_info_)
        objects.update({'update_info_list' : update_info_list})

        key2obj = {
            'model' : objects['models'],
            'loss_fn' : objects['loss_fns']
        }

        evaluate_info_list = []
        for evaluate_info in trainer['evaluate_info_list']:
            evaluate_info_ = copy.copy(evaluate_info)
            for key, obj in key2obj.items():
                evaluate_info_[key] = obj[evaluate_info[key]]
            evaluate_info_list.append(evaluate_info_)
        objects.update({'evaluate_info_list' : evaluate_info_list})

    # to rename the function later
    ###def setup_updater(self, update_info_list, evaluate_info_list):###
    def setup_engines(self):
        from engine import create_trainer, create_evaluator
        config = self.config
        objects = config['objects']
        #model = objects['model']###
        #optimizer = objects['optimizer']###
        #loss = objects['loss']###
        device = config['device']['name']
        grad_accumulation_steps = config['others'].get('grad_accumulation_steps', 1)
        metrics = objects['metrics']
        train_loader = objects['data']['train_loader']
        #metrics_log = objects['metrics_log']
        update_info_list = objects['update_info_list']
        evaluate_info_list = objects['evaluate_info_list']
        
        trainer = create_trainer(update_info_list, data_loader=train_loader, grad_accumulation_steps=grad_accumulation_steps)
        train_evaluator = create_evaluator(evaluate_info_list, metrics=metrics)
        val_evaluator = create_evaluator(evaluate_info_list, metrics=metrics)
        
        objects['engine'] = {
            'trainer' : trainer,
            'train_evaluator' : train_evaluator,
            'val_evaluator' : val_evaluator
        }

    def create_default_events(self):
        from event import create_default_events
        create_default_events(self.config)

    def run(self):
        config = self.config
        objects = config['objects']
        trainer = objects['engine']['trainer']
        train_loader = objects['data']['train_loader']
        epochs = config['train']['epochs']

        trainer.run(data=train_loader, max_epochs=epochs)

