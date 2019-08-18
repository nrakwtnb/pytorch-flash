
import sys
sys.path.append('../')

import os

NUM_MNIST_TRAIN_SAMPLES = 60000
NUM_MNIST_VALIDATION_SAMPLES = 10000

save_dir = './test_dcgan'
output_save_dir = os.path.join(save_dir, 'outputs')
model_save_dir = os.path.join(save_dir, 'models')
num_train_samples = 320
num_train_eval_samples = 200
num_validation_samples = 200

n_critic = 4
weight_clip_param = 0.05

config = {
    "device" : {
        "num_gpu" : 0,
        "gpu" : 0,
    },
    "train" : {
        "epochs" : 4,
        "batch_size" : 32,
    },
    "handlers" : {
        "early_stopping" : {
            "patience" : 2,
            'score_function' : lambda engine:-engine.state.metrics['G-loss']-engine.state.metrics['D-loss']
        },
        "checkpoint" : {
            "prefix" : 'gan',
            'save_dir' : model_save_dir,
            'target_models' : ['G', 'D'],
        },
        'output' : {
            'save_dir' : output_save_dir,
            'fileformat' : "test_ep{1}_{0}.jpg"
        }
    },
    "others" : {
        "save_dir" : save_dir,
        "grad_accumulation_steps" : 1,
        "eval_batch_size" : 20,
        "log_interval" : 25,
        "vis_tool" : "None",
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
                                            val_batch_size=manager.config["others"]["eval_batch_size"], mnist_path=mnist_path,
                                            train_dataset_size=np.random.randint(0,NUM_MNIST_TRAIN_SAMPLES,num_train_samples),
                                            val_dataset_size=np.random.randint(0,NUM_MNIST_VALIDATION_SAMPLES,num_validation_samples), download=True)

from dataloader import get_sampled_loader
eval_train_loader = get_sampled_loader(train_loader, num_samples=num_train_eval_samples)


manager.set_dataloader(train_loader=train_loader, val_loader=val_loader, eval_train_loader=eval_train_loader)



from architectures import Generator, Discriminator
from examples import gan_config

gen = Generator(gan_config.generator_info)
dis = Discriminator(gan_config.discriminator_info)


from gan.utils import GANGame

gan = GANGame(gen, dis, gan_config.latent_dim)

manager.add_model('G', gen)# for model checkpoint
manager.add_model('D', dis)# for model checkpoint
manager.add_model('GAN', gan)


import torch
from torch.nn.init import normal_
torch.manual_seed(1)
for k,p in gan.named_parameters():
    normal_(p, 0.0, 0.01)



from optimizers import get_optimzier

optimizer_info = {
    "name" : "Adam",
    "args" : {
        "lr" : 0.0002,
        "betas" : (0.5, 0.999)
    },
}

optimizerD = get_optimzier(optimizer_info, dis)
optimizerG = get_optimzier(optimizer_info, gen)

manager.add_optimizer('D', optimizerD)
manager.add_optimizer('G', optimizerG)


import torch
def get_fake_labels(y):
    return torch.full((y.size(0), ), 0, device=y.device)
def get_real_labels(y):
    return torch.full((y.size(0), ), 1, device=y.device)

import torch.nn as nn
bce = nn.BCELoss()
def loss_bce(case):
    if case == 'fake':
        get_labels = get_fake_labels
    elif case == 'real':
        get_labels = get_real_labels
    else:
        assert False, "Invalid"
    def loss_bce_(y):
        y = y.view(-1)
        return bce(y, get_labels(y))
    return loss_bce_

loss_fake = loss_bce('fake')
loss_real = loss_bce('real')

def G_update_loss(results):
    return loss_real(results['outputs']['dis_fake'])

def D_update_loss(results):
    return loss_real(results['outputs']['dis_real']) + loss_fake(results['outputs']['dis_fake'])

manager.add_loss_fn('D', D_update_loss)
manager.add_loss_fn('G', G_update_loss)

def skip_G(state):
    return state.iteration % n_critic == 1

manager.add_skip_condition('G', skip_G)

def D_pre_operator(model, iter_=1, **kwargs):
    if iter_ == 1:
        model.set_turn_D(True)
def G_pre_operator(model, **kwargs):
    model.set_turn_D(False)
manager.add_pre_operator('G', G_pre_operator)
manager.add_pre_operator('D', D_pre_operator)


update1 = {'name' : 'D', 'model' : 'GAN', 'loss_fn' : 'D', 'optimizer' : 'D', 'pre_operator' : 'D'}
manager.add_update_info(**update1)
update2 = {'name' : 'G', 'model' : 'GAN', 'loss_fn' : 'G', 'optimizer' : 'G', 'pre_operator' : 'G', 'skip_condition':'G'}
manager.add_update_info(**update2)

evaluate1 = {'model' : 'GAN', 'pre_operator' : 'D'}
manager.add_evaluate_info(**evaluate1)

manager.set_objects()


target_metrics = {'loss'}
manager.setup_metrics(target_metrics=target_metrics, target_loss_fn_names=None)

trainer_args = {"Add_update_name_in_outputs":True, "Add_update_name_in_loss":True}
manager.setup_engines(trainer_args)

manager.create_default_events()

from ignite.engine import Events
trainer = manager.config['objects']['engine']['trainer']
val_evaluator = manager.config['objects']['engine']['val_evaluator']

mean = 0.3081
std = 0.1307
def transform_images_from_normalized_tensors(x):
    x = 255 * (x * mean + std)
    _, c, h, w = x.shape
    x[x > 255] = 255
    x[x < 0] = 0
    img = x.astype(np.uint8).transpose(0,2,3,1)
    if c == 1:
        img = img[...,0]
    return img

from PIL import Image
def save_images(images, path_format, format_args):
    for n, img in enumerate(images):
        img_path = path_format.format(n, *format_args)
        Image.fromarray(img).save(img_path)

import os
@trainer.on(Events.EPOCH_COMPLETED)
def save_generated_images(engine):
    gen_output = val_evaluator.state.output['outputs']['gen'].detach().numpy()
    images = transform_images_from_normalized_tensors(gen_output)
    epoch = engine.state.epoch
    save_dir = config['handlers']['output']['save_dir']
    fileformat = config['handlers']['output']['fileformat']
    path_format = os.path.join(save_dir, fileformat)
    save_images(images, path_format, (epoch,))


manager.run()
manager.close()

