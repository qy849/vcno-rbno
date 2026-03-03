import os
import sys
import argparse

import numpy as np

from mpi4py import MPI

import dolfinx
import scifem
from tqdm import tqdm


repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, set_seed, timing, format_elapsed_time, gather_and_save, timing, get_split, evaluate_expression, load_and_scatter, gather_and_save
from data_generation.differential_equations import PoissonSetup2LeastSquares


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate solutions on grid points.') 
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

    local_sigma_u_dof, split_num_functions = load_and_scatter(comm, dataset_path+'/sigma_u_dof_finer.npy',dtype=dtype)
    local_sigma_vertex_values = np.zeros((split_num_functions[rank], (mesh_args['num_x'] + 1) * (mesh_args['num_y'] + 1), 2), dtype=dtype)
    local_u_vertex_values = np.zeros((split_num_functions[rank], (mesh_args['num_x'] + 1) * (mesh_args['num_y'] + 1), 1), dtype=dtype)
    
    dolfinx_mesh_coords = mesh.geometry.x[:,:2]
    num_x, num_y = mesh_args['num_x'], mesh_args['num_y']
    x = np.linspace(0, mesh_args['upper_right_x'], num_x+1)
    y = np.linspace(0, mesh_args['upper_right_y'], num_y+1)
    image_mesh_coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    perm = [np.where((image_mesh_coords == row).all(axis=1))[0][0] for row in dolfinx_mesh_coords]

    for i in tqdm(range(split_num_functions[rank]), desc="Evaluating sigma_u on grid points"):
        sigma_u_fc = dolfinx.fem.Function(Vh['sigma_u'])
        sigma_u_fc.x.array[:] = local_sigma_u_dof[i]
        sigma_fc = sigma_u_fc.sub(0).collapse()
        u_fc = sigma_u_fc.sub(1).collapse()

        sigma_grid_evals = evaluate_expression(mesh, sigma_fc, mesh.geometry.x)[1]
        u_grid_evals = evaluate_expression(mesh, u_fc, mesh.geometry.x)[1]

        local_sigma_vertex_values[i, perm, 0] = sigma_grid_evals[:, 0]
        local_sigma_vertex_values[i, perm, 1] = sigma_grid_evals[:, 1]
        local_u_vertex_values[i, perm, 0] = u_grid_evals[:, 0]

    gather_and_save(comm, dataset_path+'/sigma_vertex_values_finer.npy', local_sigma_vertex_values, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/u_vertex_values_finer.npy', local_u_vertex_values, split_num_functions, dtype=dtype)

    if rank == 0:
        print(f'Saved to {dataset_path}/sigma_vertex_values_finer.npy and {dataset_path}/u_vertex_values_finer.npy')