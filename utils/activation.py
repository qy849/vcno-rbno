import torch.nn as nn
import torch

class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


def get_activation(name): 

    if name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']: 
        return nn.ReLU()
    elif name in ['leakyrelu', 'LeakyReLU', 'leaky_relu']:
        return nn.LeakyReLU()
    elif name in ['prelu', 'PReLU']:
        return nn.PReLU()
    elif name in ['rrelu', 'RReLU']:
        return nn.RReLU()
    elif name in ['elu', 'ELU']:
        return nn.ELU()
    elif name in ['selu', 'SELU']:
        return nn.SELU()
    elif name in ['celu', 'CELU']:
        return nn.CELU()
    elif name in ['gelu', 'GELU']:
        return nn.GELU()
    elif name in ['silu', 'SiLU']:
        return nn.SiLU()
    elif name in ['mish', 'Mish']:
        return nn.Mish()
    elif name in ['glu', 'GLU']:
        return nn.GLU()
    elif name in ['sine', 'Sine']:
        return Sin()
    else:
        raise NotImplementedError