import os
import sys
import argparse

import numpy as np
from mpi4py import MPI


import dolfinx
from tqdm import tqdm


repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, load_npy, save_npy, format_elapsed_time, load_and_scatter, gather_and_save, timing, project
from data_generation.differential_equations import PoissonSetup2LeastSquares



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the permeability (setup2) for the Poisson problem.') 
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    dataset_path = args.dataset_path

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')


    dtype = 'float64'

    poisson_least_squares = PoissonSetup2LeastSquares(mesh_args, function_space_args)
    mesh = poisson_least_squares.mesh
    Vh = poisson_least_squares.Vh

    p_dim = dolfinx.fem.Function(poisson_least_squares.Vh['p']).x.array.shape[0]


    local_m_dof, split_num_functions = load_and_scatter(comm, dataset_path+'/m_dof.npy',dtype=dtype)
    local_p_dof = np.zeros((split_num_functions[rank], p_dim), dtype=dtype)


    for i in tqdm(range(split_num_functions[rank])):
        m = dolfinx.fem.Function(Vh['m'], dtype=dtype)
        m.x.array[:] = local_m_dof[i,:]
        p = dolfinx.fem.Function(Vh['p'], dtype=dtype)
        project(poisson_least_squares.permeability(m), p)
        local_p_dof[i,:] = p.x.array

    gather_and_save(comm, dataset_path+'/p_dof.npy', local_p_dof, split_num_functions, dtype=dtype)
    if rank == 0:
        print(f'Saved to {dataset_path}/p_dof.npy')
