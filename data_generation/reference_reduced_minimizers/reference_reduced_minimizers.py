# %%
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
import basix.ufl

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)

from utils import load_yaml, format_elapsed_time, load_and_scatter, gather_and_save, convert_petsc_mat_to_torch_sparse_coo_tensor
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the reduced weights.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--output_reduced_basis_config_path', type=str, help='Path to the output reduced basis configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the train dataset')


    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    output_reduced_basis_args = load_yaml(args.output_reduced_basis_config_path)
    dataset_path = args.dataset_path

    quadratic_weight = np.load(dataset_path+'/hdiv_h1_quadratic_weight.npy')
    linear_weight = np.load(dataset_path+'/hdiv_h1_linear_weight.npy')
    bias = np.load(dataset_path+'/hdiv_h1_bias.npy')

    num_samples, reduced_dim, _ = quadratic_weight.shape
    print(f'num_samples: {num_samples} | reduced_dim: {reduced_dim}')

    reference_reduced_minimizers = np.zeros((num_samples, reduced_dim))
    for i in tqdm(range(num_samples)):
        reference_reduced_minimizers[i] = np.linalg.solve(quadratic_weight[i], -linear_weight[i])

    np.save(os.path.join(dataset_path, 'reference_reduced_minimizers.npy'), reference_reduced_minimizers)
