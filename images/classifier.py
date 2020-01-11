
import torch
import torch.nn as nn

from develop.flash.utils import forward_wrap
from develop.flash.blocks import Block_Conv

class ConvFeatureExtractor(nn.Module):
    def __init__(self, block_info):
        super(ConvFeatureExtractor, self).__init__()
        self.block_info = block_info
        num_input_ch = block_info['num_input_ch']
        num_blocks = len(block_info['block_info'])
        blocks = []
        for i, block_info in enumerate(block_info['block_info']):
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
        #self.blocks = nn.ModuleList(modules=blocks)
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


class TraditionalTypeModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    @forward_wrap
    def forward(self, inputs):
        x = inputs['x']
        f = self.feature_extractor(x)
        f = f.view(len(f), -1).contiguous()
        y = self.classifier(f)
        return {'y':y}

