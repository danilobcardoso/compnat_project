import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from nn.layers import ST_GCN


def build_layer(layer_params, c_in, partition_modes, activation_functions):
    c_out = layer_params[0]
    partition_mode = layer_params[1]
    model_type = layer_params[2]
    activation = layer_params[3]
    af = activation_functions[activation]
    ks = partition_modes[partition_mode]['ks']
    A = partition_modes[partition_mode]['A']
    layer = ST_GCN(c_in, c_out, ks)
    return layer, c_out, A, af

def load_previous_state(layer, previous_state, i):
    bias_key = 'internal_layers.{}.conv.bias'.format(i)
    weight_key = 'internal_layers.{}.conv.weight'.format(i)
    if previous_state:
        if bias_key in previous_state:
            ps_bias = previous_state[bias_key]
            layer.conv.bias.data.copy_(ps_bias)
            layer.conv.bias.requires_grad = False
        if weight_key in previous_state:
            ps_weight = previous_state[weight_key].detach().clone()
            layer.conv.weight.data.copy_(ps_weight)
            layer.conv.weight.requires_grad = False


class EvolutionaryNet(torch.nn.Module):
    def __init__(self, input_size, output_size, partition_modes, activation_functions, genome, previous_state=None):
        super(EvolutionaryNet, self).__init__()

        layers = []
        self.activation = []
        self.adj_matrix = []

        c_in = input_size
        i = 0
        for g in genome:
            layer, c_out, A, af = build_layer(g[0:4], c_in, partition_modes, activation_functions)
            load_previous_state(layer, previous_state, i)
            layers.append(layer)
            self.adj_matrix.append(A)
            self.activation.append(af)
            c_in = c_out
            i = i + 1
            if g[4]:
                layer, c_out, A, af = build_layer(g[5:9], c_in, partition_modes, activation_functions)
                load_previous_state(layer, previous_state, i)
                layers.append(layer)
                self.adj_matrix.append(A)
                self.activation.append(af)
                c_in = c_out
                i = i + 1


        self.internal_layers = torch.nn.ModuleList(layers)
        self.final_layer = torch.nn.Conv2d(c_in, output_size,
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              stride=(1, 1),
                              dilation=(1, 1),
                              bias=True)

    def forward(self, x):

        for i, l in enumerate(self.internal_layers):
            x, A = l(x, self.adj_matrix[i])
            x = self.activation[i](x)
            x = F.dropout(x, training=self.training)

        x = self.final_layer(x)
        return F.softmax(x, dim=1)