
from collections import defaultdict

import torch
from ignite.metrics import Loss, Accuracy

from utils import wrap_metrics, get_y_values


class Config():
    def __init__(self):
        pass

    def load_config(self):
        pass

    def set_config(self, config):
        self.config = config
        self.check()

    def get_model(self, model_generator, model_args={}):
        config = self.config
        model = model_generator(**model_args)
        
        if config["device"]["num_gpu"] > 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model.cuda(config["device"]["gpu"])
            config["device"]["name"] = device
        else:
            config["device"]["name"] = 'cpu'
        
        config["object"]["model"] = model

    def setup_metrics(self, loss_fn, output_transform=get_y_values):
        """
            ToDo
                * Add other metrics ...
        """
        config = self.config
        
        loss = wrap_metrics(loss_fn, output_transform)
        config["object"]["loss"] = loss
        
        metrics={
            'accuracy': Accuracy(output_transform=get_y_values),
            'loss': Loss(loss_fn, output_transform=get_y_values)
        }
        config["object"]["metrics"] = metrics

    # to rename the function later
    def check(self):
        config = self.config
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
            
        config["object"].update({"metrics_log" : defaultdict(lambda :[])})


    # to rename the function later
    def setup_updater(self, update_info_list, evaluate_info_list):
        from updater import create_trainer, create_evaluator
        config = self.config
        objects = config["object"]
        model = objects["model"]
        optimizer = objects["optimizer"]
        loss = objects["loss"]
        device = config["device"]["name"]
        grad_accumulation_steps = config["other"]["grad_accumulation_steps"]
        metrics = objects["metrics"]
        train_loader = objects["train_loader"]
        #metrics_log = objects["metrics_log"]
        
        trainer = create_trainer(update_info_list, device=device, data_loader=train_loader, grad_accumulation_steps=grad_accumulation_steps)
        train_evaluator = create_evaluator(evaluate_info_list, metrics=metrics, device=device)#, metrics_log=metrics_log)
        val_evaluator = create_evaluator(evaluate_info_list, metrics=metrics, device=device)#, metrics_log=metrics_log)
        
        config["object"]["engine"] = {
            "trainer" : trainer,
            "train_evaluator" : train_evaluator,
            "val_evaluator" : val_evaluator
        }
        
        return trainer

    def create_default_events(self):
        from event import create_default_events
        create_default_events(self.config)
