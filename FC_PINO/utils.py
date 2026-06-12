import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import wandb
import h5py
import scipy.io
import torch
from os import path




###############################
# Functions for plotting and saving
###############################


def wandb_log(dict_, ep):
    wandb.log(dict_, step=ep)

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

def plot_embeddings(input_data, l):
        n_max = (input_data.shape[1] - 2) // 2
        batch_size = input_data.shape[0]
        num_to_plot = min(2, batch_size)
        plt.figure(figsize=(18, 6 * num_to_plot))
        for b in range(num_to_plot):
            # Plot lambda channel
            plt.subplot(num_to_plot, 3, 3 * b + 1)
            plt.plot(input_data[b, 0, :].detach().cpu().numpy(), label=f'λ (sample {b})')
            plt.title(f'Sample {b} - Lambda Channel (λ = {l[b].item() if hasattr(l, "__len__") else l:.3f})')
            plt.legend()
            # Plot cosine embeddings
            plt.subplot(num_to_plot, 3, 3 * b + 2)
            for i in range(n_max):
                plt.plot(input_data[b, 1+2*i, :].detach().cpu().numpy(), label=f'cos(2^{i+1}t)λ')
            plt.title(f'Sample {b} - Cosine Embeddings')
            plt.legend()
            # Plot sine embeddings
            plt.subplot(num_to_plot, 3, 3 * b + 3)
            for i in range(n_max):
                plt.plot(input_data[b, 1+2*i+1, :].detach().cpu().numpy(), label=f'sin(2^{i+1}t)λ')
            plt.title(f'Sample {b} - Sine Embeddings')
            plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('plots/embeddings.png')
        plt.close()