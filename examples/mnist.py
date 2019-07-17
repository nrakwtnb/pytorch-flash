
import sys
sys.path.append('../')

config = {
    'device' : {
        'num_gpu' : 0,
        'gpu' : 0,
    },
    'train' : {
        'epochs' : 3,
        'batch_size' : 32,
    },
    'handlers' : {
        'early_stopping' : {
            'patience': 1,
        },
        'checkpoint' : {
            'prefix' : 'mnist',
            'save_dir' : 'debug'
        }
    },
    'others' : {
        'grad_accumulation_steps' : 1,
        'val_batch_size' : 50,
        'log_interval' : 30,
        'vis_tool' : 'None',
    }
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mnist-path")
args = parser.parse_args()

from manager import TrainManager
manager = TrainManager()
manager.set_config(config)

import numpy as np
np.random.seed(10)
from debug import get_data_loaders
mnist_path = args.mnist_path
train_loader, val_loader = get_data_loaders(train_batch_size=manager.config["train"]["train_batch_size"],
                                            val_batch_size=manager.config["others"]["val_batch_size"], mnist_path=mnist_path,
                                            train_dataset_size=np.random.randint(0,60000,20000),
                                            val_dataset_size=np.random.randint(0,10000,2000), download=True)

from dataloader import get_sampled_loader
eval_train_loader = get_sampled_loader(train_loader, num_samples=2000)


manager.set_dataloader(train_loader=train_loader, val_loader=val_loader, eval_train_loader=eval_train_loader)

from debug import TestNet
model = TestNet()

from optimizers import get_optimzier

optimizer_info = {
    "name" : "SGD",
    "args" : {
        "lr" : 0.01,
        "momentum" : 0.5    
    },
}

optimizer = get_optimzier(optimizer_info, model)

manager.add_model('test', model)

manager.add_optimizer('test', optimizer)
from utils import wrap_metrics, get_y_values
import torch.nn as nn
manager.add_loss_fn('test', wrap_metrics(nn.NLLLoss(), get_y_values))


update1 = {'model' : 'test', 'loss_fn' : 'test', 'optimizer' : 'test'}
manager.add_update_info(**update1)

evaluate1 = {'model' : 'test',  'loss_fn' : 'test'}
manager.add_evaluate_info(**evaluate1)

manager.set_objects()

target_metrics = {'loss', 'accuracy', 'precision', 'recall', 'precision_class', 'recall_class', 'F1', 'F1_class'}
manager.setup_metrics(target_metrics=target_metrics, target_loss_fn_names=None)

manager.setup_engines()
manager.create_default_events()
manager.run()

manager.close()

