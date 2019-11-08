import torch.nn as nn

# generator

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


# discriminator

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
