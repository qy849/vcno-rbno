import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
                input_dim: int = 512, 
                output_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SELU(),
            nn.Linear(1024, 2048),
            nn.SELU(),
            nn.Linear(2048, 1024),
            nn.SELU(),
            nn.Linear(1024, output_dim)
            )

    def forward(self, x):
        return self.net(x)