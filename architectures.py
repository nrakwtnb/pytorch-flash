
import torch.nn as nn

from utils import forward_wrap

class Generator(nn.Module):
    def __init__(self, generator_info):
        super(Generator, self).__init__()
        #self.ngpu = ngpu
        latent_dim = generator_info['latent_dim']
        num_blocks = len(generator_info['block_info'])
        blocks = []
        for i, block_info in enumerate(generator_info['block_info']):#num_blocks):
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
    def forward(self, inputs):
        return self.nn(inputs)


class Discriminator(nn.Module):
    def __init__(self, discriminator_info):
        super(Discriminator, self).__init__()
        num_input_ch = discriminator_info['num_input_ch']
        num_blocks = len(discriminator_info['block_info'])
        blocks = []
        for i, block_info in enumerate(discriminator_info['block_info']):
            #for i in range(num_blocks):
            conv_info = block_info['conv_info']
            if i == 0:
                conv_info['input_ch'] = num_input_ch
            else:
                conv_info['input_ch'] = num_ch
            num_ch = conv_info['channels']
            conv_info['output_ch'] = num_ch
            blocks.append(Block_Conv(block_info))
        self.blocks = nn.ModuleList(modules=blocks)
        self.nn = nn.Sequential(*blocks)

    @forward_wrap
    def forward(self, inputs):
        return self.nn(inputs)


