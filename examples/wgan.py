
import sys
sys.path.append('../')

import os

NUM_MNIST_TRAIN_SAMPLES = 60000
NUM_MNIST_VALIDATION_SAMPLES = 10000

save_dir = './test_wgan'
output_save_dir = os.path.join(save_dir, 'outputs')
model_save_dir = os.path.join(save_dir, 'models')
num_train_samples = 320
num_train_eval_samples = 200
num_validation_samples = 200

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
import torch.nn as nn

latent_dim = 24
gch = 32
num_ch = 1

deconv_init = {
    'kernel_size' : 4,
    'stride' : 1,
    'padding' : 0
}
deconv_double = {
    'kernel_size' : 4,
    'stride' : 2,
    'padding' : 1
}

block_info = []
for i in range(2):
    block_info.append({
        'deconv_info' : {'channels' : gch*2**(2-i), **deconv_init },
        'bn' : True,
        'activation' : nn.ReLU()
    })
block_info.append({
    'deconv_info' : {'channels' : gch, **deconv_double },
    'bn' : True,
    'activation' : nn.ReLU()
})
block_info.append({
    'deconv_info' : {'channels' : num_ch, **deconv_double },
    'bn' : False,
    'activation' : nn.Tanh()
})

generator_info = {
    'latent_dim' : latent_dim,
    'block_info': block_info
}

gen = Generator(generator_info)

dch = 32

conv_half = {
    'kernel_size' : 4,
    'stride' : 2,
    'padding' : 1
}
conv_final = {
    'kernel_size' : 4,
    'stride' : 1,
    'padding' : 0
}


block_info = []
for i in range(2):
    block_info.append({
        'conv_info' : {'channels' : dch*2**i, **conv_half },
        'bn' : True,
        'activation' : nn.LeakyReLU(0.2)
    })
block_info.append({
    'conv_info' : {'channels' : dch*2**2, **conv_final },
    'bn' : True,
    'activation' : nn.Sigmoid()
})
block_info.append({
    'conv_info' : {'channels' : 1, **conv_final },
    'bn' : False,
    'activation' : nn.Sigmoid()
})

discriminator_info = {
    'num_input_ch' : num_ch,
    'block_info': block_info
}

dis = Discriminator(discriminator_info)


import torch
# how to treat latent_dim ?
class GANGame(nn.Module):
    def __init__(self, generator, discriminator, latent_dim):
        super(GANGame, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.turn_D = True
    
    def set_turn_D(self, turn_D):
        self.turn_D = turn_D
    
    def forward(self, inputs):
        x_real = inputs['x']
        batch_size = x_real.shape[0]
        if 'seed' in inputs:
            torch.manual_seed(inputs['seed'])
        z = torch.randn(batch_size,self.latent_dim,1,1)# Ex. latent_dim = 12, (12,1,1) ?
        
        retain_G_comp_graph = not self.turn_D
        gen_out = self.generator(z, retain_comp_graph=retain_G_comp_graph)
        
        dis_fake_out = self.discriminator(gen_out)
        if self.turn_D:
            dis_real_out = self.discriminator(x_real)
            return {'gen' : gen_out, 'dis_real' : dis_real_out, 'dis_fake' : dis_fake_out, 'z':z}
        else:
            return {'gen' : gen_out, 'dis_fake' : dis_fake_out, 'z':z}

gan = GANGame(gen, dis, latent_dim)

manager.add_model('G', gen)# for model checkpoint
manager.add_model('D', dis)# for model checkpoint
manager.add_model('GAN', gan)


from torch.nn.init import normal_
torch.manual_seed(1)
for k,p in gan.named_parameters():
    normal_(p, 0.0, 0.01)



from optimizers import get_optimzier

optimizer_info = {
    "name" : "RMSprop",
    "args" : {
        "lr" : 0.0001,
        #"betas" : (0.5, 0.999)
    },
}

optimizerD = get_optimzier(optimizer_info, dis)
optimizerG = get_optimzier(optimizer_info, gen)

manager.add_optimizer('D', optimizerD)
manager.add_optimizer('G', optimizerG)


def create_GAN_loss_func(loss_func, auxiliary_loss_func=None, fake_label=0, real_label=1, fake_key='dis_fake', real_key='dis_real'):
    def get_fake_labels(y):
        return torch.full((y.size(0), ), fake_label, device=y.device)
    def get_real_labels(y):
        return torch.full((y.size(0), ), real_label, device=y.device)

    def generate_GAN_loss(loss_func, case):
        if case == 'fake':
            get_labels = get_fake_labels
        elif case == 'real':
            get_labels = get_real_labels
        else:
            assert False, "Invalid"
        def loss_func_for_fixed_labels(y):
            y = y.view(-1)
            return loss_func(y, get_labels(y))
        return loss_func_for_fixed_labels
    
    loss_fake = generate_GAN_loss(loss_func, 'fake')
    loss_real = generate_GAN_loss(loss_func, 'real')
    
    if auxiliary_loss_func is not None:
        if isinstance(auxiliary_loss_func, dict):
            auxiliary_loss_func_G = auxiliary_loss_func.get('G', None)
            auxiliary_loss_func_D = auxiliary_loss_func.get('D', None)
        else:
            auxiliary_loss_func_G = auxiliary_loss_func
            auxiliary_loss_func_D = auxiliary_loss_func
    else:
        auxiliary_loss_func_G = None
        auxiliary_loss_func_D = None
        
    if auxiliary_loss_func_G is None: 
        def G_update_loss(results):
            return loss_real(results['outputs'][fake_key])
    else:
        def G_update_loss(results):
            return loss_real(results['outputs'][fake_key]) + auxiliary_loss_func_G(results)
    if auxiliary_loss_func_D is None: 
        def D_update_loss(results):
            return loss_real(results['outputs'][real_key]) + loss_fake(results['outputs'][fake_key])
    else:
        def D_update_loss(results):
            return loss_real(results['outputs'][real_key]) + loss_fake(results['outputs'][fake_key]) + auxiliary_loss_func_D(results)
    return { 'G' : G_update_loss, 'D' : D_update_loss}

def WGAN_loss(y, label):
    return ((1 - 2 * label) * y).mean()

GAN_loss = create_GAN_loss_func(loss_func=WGAN_loss)
manager.add_loss_fn('D', GAN_loss['D'])
manager.add_loss_fn('G', GAN_loss['G'])


def D_pre_operator(model, iter_=1, **kwargs):
    if iter_ == 1:
        model.set_turn_D(True)
def G_pre_operator(model, **kwargs):
    model.set_turn_D(False)
manager.add_pre_operator('G', G_pre_operator)
manager.add_pre_operator('D', D_pre_operator)


from func import weight_clipping
manager.add_post_operator('clip', weight_clipping(0.05))



manager.reset_update_info()
manager.reset_evaluate_info()

update1 = {'name' : 'D', 'model' : 'GAN', 'loss_fn' : 'D', 'optimizer' : 'D', 'num_iter' : 4, 'pre_operator' : 'D', 'post_operator' : 'clip'}
manager.add_update_info(**update1)
update2 = {'name' : 'G', 'model' : 'GAN', 'loss_fn' : 'G', 'optimizer' : 'G', 'pre_operator' : 'G'}#, 'skip_condition':'G'}
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

