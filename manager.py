
import os
from collections import defaultdict

import torch
from ignite.metrics import Accuracy

from utils import wrap_metrics, get_y_values
from metrics import Loss


"""
    ToDo
    * assign a device to each model model
    * add the following properties
        * __repr__
        * __str__
        * __dict__
        * __getter__
        * __setter__
        * quick access to the field in config
            + manager.config.objects.engine.trainer, for example
"""
class TrainManager():
    def __init__(self):
        pass

    def load_config(self):
        pass

    def set_config(self, config: dict):
        self.config = config
        self.check()

    def add_model_from_generator(self, model_name:str, model_generator, model_args={}, device=None):
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
            #if device == 'cuda':
            #    model.cuda(config['device']['gpu'])
            config['device']['name'] = device
        else:
            config['device']['name'] = 'cpu'
        
        config['objects']['model'] = model
        """

    def add_model(self, model_name:str, model, device=None):
        assert isinstance(model_name ,str)
        config = self.config
        config['objects']['models'].update({model_name : model})

    def add_optimizer(self, optimizer_name:str, optimizer):
        assert isinstance(optimizer_name ,str)
        config = self.config
        config['objects']['optimizers'].update({optimizer_name : optimizer})

    # keep
    def add_loss_fn_old(self, loss_fn_name, loss_fn, output_transform=get_y_values):
        assert isinstance(loss_fn_name ,str)
        config = self.config
        loss_fn_ = wrap_metrics(loss_fn, output_transform)
        config['objects']['loss_fns'].update({loss_fn_name : loss_fn_})

    def add_loss_fn(self, loss_fn_name, loss_fn):
        assert isinstance(loss_fn_name ,str)
        config = self.config
        config['objects']['loss_fns'].update({loss_fn_name : loss_fn})

    def add_skip_condition(self, skip_condition_name, skip_condition):
        assert isinstance(skip_condition_name ,str)
        config = self.config
        config['objects']['skip_condition'].update({skip_condition_name : skip_condition})

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
        assert all(map(lambda x:isinstance(x, str), evaluate_info.values()))
        config = self.config
        evaluate_info_list = config['trainer']['evaluate_info_list']
        evaluate_info_list.append(evaluate_info)

    def set_dataloader(self, train_loader, val_loader, eval_train_loader=None):
        config = self.config
        data = {
            'train_loader' : train_loader,
            'val_loader' : val_loader
        }
        if eval_train_loader is not None:
            data.update({'eval_train_loader' : eval_train_loader})
        config["objects"].update({ 'data' : data })

    # temporal ?
    def setup_metrics(self, target_metrics, output_transform=get_y_values, target_loss_fn_names=None):
        """
            ToDo
                * Add other metrics ...
                * Refactoring and make/unify the logic clear
        """
        from metrics import get_precision, get_recall, get_F1score
        config = self.config
        loss_fns = self.config['objects']['loss_fns']

        metrics = {}
        if 'loss' in target_metrics:
            if target_loss_fn_names is None:
                if len(loss_fns) == 1:
                    loss_fn = list(loss_fns.values())[0]
                    metrics.update({ 'loss' : Loss(loss_fn) })###
                else:
                    target_loss_fn_names = list(loss_fns.keys())
            if target_loss_fn_names is not None:
                for name, loss_fn in loss_fns.items():
                    metrics.update({ f"{name}-loss" : Loss(loss_fn) })###
        if 'accuracy' in target_metrics:
            metrics.update({ 'accuracy' : Accuracy(output_transform=output_transform) })
        Is_average = 'precision' in target_metrics
        Is_classwise = 'precision_class' in target_metrics
        if Is_average or Is_classwise:
            metrics.update(get_precision(Is_average=Is_average, Is_classwise=Is_classwise, output_transform=output_transform))
        Is_average = 'recall' in target_metrics
        Is_classwise = 'recall_class' in target_metrics
        if Is_average or Is_classwise:
            metrics.update(get_recall(Is_average=Is_average, Is_classwise=Is_classwise, output_transform=output_transform))
        Is_average = 'F1' in target_metrics
        Is_classwise = 'F1_class' in target_metrics
        if Is_average or Is_classwise:
            metrics.update(get_F1score(Is_average=Is_average, Is_classwise=Is_classwise, output_transform=output_transform))

        print(f"set {tuple(list(metrics.keys()))}")

        config['objects']['metrics'] = metrics

    # to rename the function later
    def check(self):
        config = self.config

        config['train']['train_batch_size'] = config['train']['batch_size']###

        ### change ? (temporal)
        if config['device']['num_gpu'] > 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #if device == 'cuda':
            #    model.cuda(config['device']['gpu'])
            config['device']['name'] = device
        else:
            config['device']['name'] = 'cpu'

        if 'objects' not in config.keys():
            config['objects'] = {}
        if 'trainer' not in config.keys():
            config['trainer'] = {}
            
        config['objects'].update({'models' : {}})
        config['objects'].update({'loss_fns' : {}})
        config['objects'].update({'optimizers' : {}})
        config['objects'].update({'skip_condition' : {}})
        config['objects'].update({'metrics_log' : defaultdict(lambda :[])})
        config['trainer'].update({'update_info_list' : []})
        config['trainer'].update({'evaluate_info_list' : []})

        self._save_dir()

    def _save_dir(self):
        """
            ToDo
                * refactoring
        """
        config = self.config
        save_dir = config.get('handlers', {}).get('checkpoint', {}).get('save_dir', None)
        if save_dir is not None:
            if os.path.exists(save_dir):
                assert False, 'already exists !'
            else:
                os.makedirs(save_dir)
        save_dir = config.get('handlers', {}).get('output', {}).get('save_dir', None)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        save_dir = config.get('others', {}).get('save_dir', None)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def set_objects(self):
        import copy
        config = self.config
        trainer = config['trainer']
        objects = config['objects']
        key2obj = {
            'model' : objects['models'],
            'loss_fn' : objects['loss_fns'],
            'optimizer' : objects['optimizers'],
            'skip_condition' : objects['skip_condition']
        }

        update_info_list = []
        for update_info in trainer['update_info_list']:
            update_info_ = copy.copy(update_info)
            for key, obj in key2obj.items():
                if key in update_info:
                    update_info_[key] = obj[update_info[key]]
            update_info_list.append(update_info_)
        objects.update({'update_info_list' : update_info_list})

        key2obj = {
            'model' : objects['models'],
        }

        evaluate_info_list = []
        for evaluate_info in trainer['evaluate_info_list']:
            evaluate_info_ = copy.copy(evaluate_info)
            for key, obj in key2obj.items():
                evaluate_info_[key] = obj[evaluate_info[key]]
            evaluate_info_list.append(evaluate_info_)
        objects.update({'evaluate_info_list' : evaluate_info_list})

    # to rename the function later if necessary
    def setup_engines(self, trainer_args={}, evaluator_args={}):
        from engine import create_trainer, create_evaluator
        config = self.config
        objects = config['objects']
        device = config['device']['name']
        grad_accumulation_steps = config['others'].get('grad_accumulation_steps', 1)
        metrics = objects['metrics']
        train_loader = objects['data']['train_loader']
        #metrics_log = objects['metrics_log']
        update_info_list = objects['update_info_list']
        evaluate_info_list = objects['evaluate_info_list']
        
        trainer = create_trainer(update_info_list, data_loader=train_loader, grad_accumulation_steps=grad_accumulation_steps, **trainer_args)
        train_evaluator = create_evaluator(evaluate_info_list, metrics=metrics, **evaluator_args)
        val_evaluator = create_evaluator(evaluate_info_list, metrics=metrics, **evaluator_args)
        
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

    def close(self):
        from event import close_logger
        import json
        import os
        config = self.config
        others = config['others']
        objects = config['objects']
        if 'save_dir' in others:
            metrics_log = objects['metrics_log']
            save_path = os.path.join(others['save_dir'], 'metrics_log.json')
            with open(save_path, 'w') as f:
                json.dump(metrics_log, f)
        
        close_logger(objects['vis_tool'], others['vis_tool'])
