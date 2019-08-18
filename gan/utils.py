import torch
import torch.nn as nn

# how to treat latent_dim ?
class GANGame(nn.Module):
    def __init__(self, generator, discriminator, latent_dim):
        super(GANGame, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = (latent_dim,1,1)
        self.turn_D = True
    
    def set_turn_D(self, turn_D):
        self.turn_D = turn_D
    
    def forward(self, inputs):
        x_real = inputs['x']
        batch_size = x_real.shape[0]
        if 'seed' in inputs:
            torch.manual_seed(inputs['seed'])
        z = torch.randn(batch_size, *self.latent_dim)# Ex. latent_dim = 12, (12,1,1) ?
        
        retain_G_comp_graph = not self.turn_D
        gen_out = self.generator(z, retain_comp_graph=retain_G_comp_graph)
        
        dis_fake_out = self.discriminator(gen_out)
        if self.turn_D:
            dis_real_out = self.discriminator(x_real)
            return {'gen' : gen_out, 'dis_real' : dis_real_out, 'dis_fake' : dis_fake_out, 'z':z}
        else:
            return {'gen' : gen_out, 'dis_fake' : dis_fake_out, 'z':z}


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
