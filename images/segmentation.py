import torch
import torch.nn as nn


class UnetEncoder(nn.Module):
    def __init__(self, model, output_layer_list):
        super(UnetEncoder, self).__init__()
        if hasattr(model, 'children'):
            model_sequence = list(model.children())
        else:
            assert False
        output_layer_list = [0] + [ n if n >=0 else len(model_sequence)+n+1 for n in output_layer_list ]
        output_layer_list = list(set(output_layer_list))
        output_layer_list.sort()
        assert output_layer_list[0] >= 0 and output_layer_list[-1] <= len(model_sequence)
        layers = [ nn.Sequential(*model_sequence[output_layer_list[i]:output_layer_list[i+1]]) for i in range(len(output_layer_list)-1) ]
        #self.layers = nn.ModuleList(layers)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs

class UnetConnection(nn.Module):
    def __init__(self, connection_info_list):
        super(UnetConnection, self).__init__()
        self.connector = nn.ModuleList([info['module'](**info['args']) for info in connection_info_list])
    
    def forward(self, inputs):
        return [ layer(x) for layer, x in zip(self.connector, inputs) ]

class UnetDecoder(nn.Module):
    def __init__(self, decoder_info_list):
        super(UnetDecoder, self).__init__()
        self.layers = nn.ModuleList([info['module'](**info['args']) for info in decoder_info_list])
    
    def forward(self, inputs):
        x = None
        for layer,h in zip(self.layers, inputs[::-1]):
            x = h if x is None else torch.cat([x,h], dim=1)
            x = layer(x)
        return x

class UnetOutput(nn.Module):
    def __init__(self, output_info_list):
        super(UnetOutput, self).__init__()
        self.layers = nn.Sequential(*[info['module'](**info['args']) for info in output_info_list])
    
    def forward(self, x):
        return self.layers(x)

class Unet(nn.Module):
    def __init__(self, encoder_info, connection_info, decoder_info, output_info):
        super(Unet, self).__init__()
        if 'base_model' in encoder_info:
            self.encoder = UnetEncoder(encoder_info['base_model'], encoder_info['encoder_layer_list'])
            num_bridges = len(encoder_info['encoder_layer_list'])
        else:
            assert False, 'not implemented'
    
        if connection_info is not None:
            self.connector = UnetConnection(connection_info)
        else:
            self.connector = None
        self.decoder = UnetDecoder(decoder_info)
        self.output = UnetOutput(output_info)
    
    def forward(self, inputs):
        x = inputs['image']
        h = self.encoder(x)
        if self.connector is not None:
            h = self.connector(h)
        h = self.decoder(h)
        h = self.output(h)
        return {'y':h}


