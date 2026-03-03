import torch
import torch.nn as nn

from utils import get_activation, get_initializer

class ResidualBlock1d(nn.Module):
    """
    input_shape = (batch_size, input_dim)
    output_shape = (batch_size, output_dim)
    """

    def __init__(self,
                 input_dim: int, 
                 output_dim: int, 
                 activation: str, 
                 init_func: str):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation1 = get_activation(self.activation_str)
        self.activation2 = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        # self.bn1 = nn.BatchNorm1d(self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        # self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.shortcut = nn.Sequential()
        if self.input_dim != self.output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
                # nn.BatchNorm1d(self.output_dim)
            )
        self.init_params()
        

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # y = self.activation1(self.bn1(self.linear1(x)))
        # y = self.bn2(self.linear2(y))
        # y = self.activation2(y + self.shortcut(x))

        y = self.activation1(self.linear1(x))
        y = self.linear2(y)
        y = self.activation2(y + self.shortcut(x))

        return y


class ResNet1d(nn.Module):
    """
    input_shape = (batch_size, input_dim)
    output_shape = (batch_size, output_dim)
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int,
                 hidden_depth: int,
                 activation: str,
                 init_func: str):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(activation)
        self.init_func = get_initializer(init_func)

        self.num_residual_blocks = (self.hidden_depth - 1)//2
        self.num_remaining_hidden_layers = (self.hidden_depth - 1)%2

        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_residual_blocks):
            self.residual_blocks.append(ResidualBlock1d(self.hidden_dim, 
                                                        self.hidden_dim, 
                                                        self.activation_str, 
                                                        self.init_func_str))

        if self.num_remaining_hidden_layers == 1:
            layers = [nn.Linear(self.hidden_dim, self.hidden_dim), 
                    #   nn.BatchNorm1d(self.hidden_dim), 
                      get_activation(activation)]
            self.remaining_hidden_layers = nn.ModuleList(layers)
        else:
            self.remaining_hidden_layers = nn.ModuleList()

        self.input_hidden_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.init_params()

    def init_params(self):

        for block in self.residual_blocks:
            block.init_params()
        for layer in self.remaining_hidden_layers:
            if isinstance(layer, nn.Linear):
                self.init_func(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.init_func(self.input_hidden_linear.weight)
        if self.input_hidden_linear.bias is not None:
            nn.init.zeros_(self.input_hidden_linear.bias)
        self.init_func(self.hidden_output_linear.weight)
        if self.hidden_output_linear.bias is not None:
            nn.init.zeros_(self.hidden_output_linear.bias)

    def forward(self, x):

        x = self.activation(self.input_hidden_linear(x))
        for block in self.residual_blocks:
            x = block(x)
        for layer in self.remaining_hidden_layers:
            x = layer(x)
        x = self.hidden_output_linear(x)

        return x