
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

from config import Config
cfg = Config()
cfg.set_config(config)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mnist-path")
args = parser.parse_args()

import numpy as np
from debug import get_data_loaders
mnist_path = args.mnist_path
train_loader, val_loader = get_data_loaders(train_batch_size=cfg.config["train"]["train_batch_size"],
                                            val_batch_size=cfg.config["others"]["val_batch_size"], mnist_path=mnist_path,
                                            train_dataset_size=np.random.randint(0,60000,20000),
                                            val_dataset_size=np.random.randint(0,10000,2000))

cfg.config['objects'].update({ 'data' : { 'train_loader' : train_loader, 'val_loader' : val_loader }})

from debug import TestNet
model = TestNet()

def get_optimzier(optimizer_info, model):
    opt_name = optimizer_info['name']
    opt_info = optimizer_info['info']
    if opt_name in ['SGD']:
        from torch.optim import SGD
        return SGD(model.parameters(), **opt_info)
    elif opt_name in ['Adam']:
        from torch.optim import Adam
        return Adam(model.parameters(), **opt_info)
    else:
        assert False, "Invalid optimizer name"

optimizer_info = {
    "name" : "SGD",
    "info" : {
        "lr" : 0.01,
        "momentum" : 0.5    
    },
}

optimizer = get_optimzier(optimizer_info, model)

cfg.add_model('test', model)

import torch.nn.functional as F
cfg.add_optimizer('test', optimizer)
cfg.add_loss_fn('test', F.nll_loss)


update1 = {'model' : 'test', 'loss_fn' : 'test', 'optimizer' : 'test'}
cfg.add_update_info(**update1)

evaluate1 = {'model' : 'test',  'loss_fn' : 'test'}
cfg.add_evaluate_info(**evaluate1)

cfg.set_objects()
cfg.setup_metrics()
cfg.setup_engines()
cfg.create_default_events()
cfg.run()


