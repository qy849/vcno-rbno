import torch
import torch.nn as nn

from utils import get_activation, get_initializer


class ConvolutionalNN_65x65(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (65,65)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=3),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            self.activation,
            nn.Flatten(),
            nn.Linear(512*2*2, 512),
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        return x


class ConvolutionalNN_81x81(nn.Module):
    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):
        super().__init__()

        self.input_dim = (81, 81)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),   # → (64, 39, 39)
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2), # → (128, 18, 18)
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=3),# → (256, 5, 5)
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1),# → (512, 3, 3)
            self.activation,
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),  # Updated to match new output size
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


class ConvolutionalNN_65x129(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (65, 129)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),    # -> (64, 31, 63)
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),  # -> (128, 14, 30)
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=3), # -> (256, 4, 9)
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1), # -> (512, 2, 7)
            self.activation,
            nn.Flatten(),
            nn.Linear(512*2*7, 512),  # updated to 7168
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


class ConvolutionalNN_129x129(nn.Module):
    def __init__(self, 
                 output_dim: int,
                 activation: str,
                 init_func: str):
        super().__init__()

        self.input_dim = (129, 129)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        # Convolutional layers for 129x129 input
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),    # (64, 63, 63)
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),  # (128, 30, 30)
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=3), # (256, 9, 9)
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1), # (512, 7, 7)
            self.activation,
            nn.Flatten(),
            nn.Linear(512*7*7, 512),                      # flatten to dense
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


# class ConvolutionalNN_129x129(nn.Module):
#     def __init__(self, 
#                  output_dim: int,
#                  activation: str = "relu",
#                  init_func: str = "xavier_uniform"):
#         super().__init__()

#         self.input_dim = (129, 129)
#         self.output_dim = output_dim
#         self.activation_str = activation
#         self.init_func_str = init_func

#         self.activation = get_activation(self.activation_str)
#         self.init_func = get_initializer(self.init_func_str)

#         # Convolutional feature extractor
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=2),    # (64, 63, 63)
#             self.activation,
#             nn.Conv2d(64, 128, kernel_size=5, stride=2),  # (128, 30, 30)
#             self.activation,
#             nn.Conv2d(128, 256, kernel_size=5, stride=3), # (256, 9, 9)
#             self.activation,
#             nn.Conv2d(256, 512, kernel_size=3, stride=1), # (512, 7, 7)
#             self.activation,
#             nn.Flatten()
#         )

#         # Dense (fully connected) layers with skip connection
#         self.fc1 = nn.Linear(512 * 7 * 7, 1024)
#         self.fc2 = nn.Linear(1024, output_dim)
#         self.proj1 = nn.Linear(512 * 7 * 7, output_dim)

#         self.fc3 = nn.Linear(output_dim, output_dim)
#         self.fc4 = nn.Linear(output_dim, output_dim)

#         self.init_params() 
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv2d)):
#                 self.init_func(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         # Convolutional feature extraction
#         x = self.conv_layers(x)

#         # Dense layers with skip connection
#         h1 = self.activation(self.fc1(x))
#         h2 = self.activation(self.fc2(h1) + self.proj1(x))
#         h3 = self.activation(self.fc3(h2))
#         out = self.fc4(h3)
#         return out