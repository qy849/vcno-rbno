import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0)

class BatchIndicesIterator:
    def __init__(self, start: int, end: int, batch_size: int, shuffle: bool=True):
        self.start = start
        self.end = end
        if  self.start >= self.end:
            raise ValueError(f'The start index {self.start} must be less than the end index {self.end}.')
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = torch.arange(start, end)
        self.num_indices = len(self.indices)
        self.current_index = 0

        if self.shuffle:
            self.indices = self.indices[torch.randperm(self.num_indices)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.end - self.start:
            if self.shuffle:
                self.indices = self.indices[torch.randperm(self.num_indices)]
            self.current_index = 0
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch_indices



class AddSpatialCoordinates(nn.Module):
    def __init__(self, num_x: int, num_y: int): 
        super().__init__()
        self.num_x, self.num_y = (num_x, num_y)
        self.x = np.linspace(0, 1, num_x)
        self.y = np.linspace(0, 1, num_y)
        self.x_coor, self.y_coor = np.meshgrid(self.x, self.y)
        self.x_coor = torch.from_numpy(self.x_coor).to(dtype=torch.float32)
        self.y_coor = torch.from_numpy(self.y_coor).to(dtype=torch.float32)

    def forward(self, inputs):
        repeat_x_coor = self.x_coor.reshape(1,1,self.num_x,self.num_y).repeat(inputs.shape[0],1,1,1)
        repeat_y_coor = self.y_coor.reshape(1,1,self.num_x,self.num_y).repeat(inputs.shape[0],1,1,1)

        return torch.cat((inputs, repeat_x_coor, repeat_y_coor), dim=1)

    def to(self, device):
        self.x_coor = self.x_coor.to(device)
        self.y_coor = self.y_coor.to(device)
        return self


def pretty_print_loss(loss_dict):
    formatted_loss = {key: value.item() for key, value in loss_dict.items()}
    print(', '.join(f"'{key}': {value}" for key, value in formatted_loss.items()))


