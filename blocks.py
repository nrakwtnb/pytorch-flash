
import torch.nn as nn


class Block_Conv(nn.Module):
    def __init__(self, info):
        super(Block_Conv, self).__init__()
        if 'conv_info' in info:
            conv_type = 'conv'
        elif 'deconv_info' in info:
            conv_type = 'deconv'
        else:
            assert False
        conv_info = info[f'{conv_type}_info']
        input_ch = conv_info['input_ch']
        output_ch = conv_info['output_ch']
        kernel_size = conv_info['kernel_size']
        stride = conv_info.get('stride', 1)
        padding = conv_info.get('padding',0)
        bias = conv_info.get(f'{conv_type}_bias', True)
        layers = []
        
        if conv_type == 'conv':
            layer = nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv = layer
        else:
            layer = nn.ConvTranspose2d(in_channels=input_ch, out_channels=output_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.deconv = layer
        layers.append(layer)
        
        if info['bn']:
            bn = nn.BatchNorm2d(num_features=output_ch)
            self.bn = bn
            layers.append(bn)
        
        act_info = info.get('activation', lambda x:x)
        if isinstance(act_info, str):
            if act_info.lower() == 'relu':
                act_fn = nn.ReLU()
        else:
            # check if the function
            act_fn = act_info
        layers.append(act_fn)
        self.activation = act_fn
        self.layers = layers
        
    def forward(self, x):# h = x.copy() beause inplace activation function may comes first...
        for l in self.layers:
            x = l(x)
        return x


