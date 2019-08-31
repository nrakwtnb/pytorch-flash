
import torch
import torch.nn as nn

from flash.utils import forward_wrap
from flash.blocks import Block_Conv

"""
    The following are examples at this stage.
    To be fixed in the future...
"""


class Generator(nn.Module):
    def __init__(self, generator_info):
        super(Generator, self).__init__()
        self.generator_info = generator_info
        latent_dim = generator_info['latent_dim']
        num_blocks = len(generator_info['block_info'])
        blocks = []
        for i, block_info in enumerate(generator_info['block_info']):
            deconv_info = block_info['deconv_info']
            if i == 0:
                deconv_info['input_ch'] = latent_dim
            else:
                deconv_info['input_ch'] = num_ch
            num_ch = deconv_info['channels']
            deconv_info['output_ch'] = num_ch
            blocks.append(Block_Conv(block_info))
        self.blocks = nn.ModuleList(modules=blocks)
        self.nn = nn.Sequential(*blocks)

    @forward_wrap
    def forward(self, inputs=None):
        if isinstance(inputs, dict):
            z = inputs['z']
        elif isinstance(inputs, torch.Tensor):
            z = inputs
        elif inputs is None:
            assert 'not implemented'
            # generate itself
        else:
            assert False, "Invalid inputs"
        return self.nn(z)


class Discriminator(nn.Module):
    def __init__(self, discriminator_info):
        super(Discriminator, self).__init__()
        self.discriminator_info = discriminator_info
        num_input_ch = discriminator_info['num_input_ch']
        num_blocks = len(discriminator_info['block_info'])
        blocks = []
        for i, block_info in enumerate(discriminator_info['block_info']):
            if 'conv_info' in block_info:
                conv_info = block_info['conv_info']
                if i == 0:
                    conv_info['input_ch'] = num_input_ch
                else:
                    conv_info['input_ch'] = num_ch
                num_ch = conv_info['channels']
                conv_info['output_ch'] = num_ch
                blocks.append(Block_Conv(block_info))
            if 'linear_info' in block_info:
                # flatten just after conv layers
                assert 'not implemented'
        self.blocks = nn.ModuleList(modules=blocks)
        self.nn = nn.Sequential(*blocks)

    @forward_wrap
    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs['x']
        elif isinstance(inputs, torch.Tensor):
            x = inputs
        else:
            assert False, "Invalid inputs"
        return self.nn(x)


