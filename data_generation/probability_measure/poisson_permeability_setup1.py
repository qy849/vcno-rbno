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

from utils import load_yaml, set_seed, timing, format_elapsed_time, gather_and_save, timing, get_split
from data_generation.differential_equations import PoissonSetup1LeastSquares

set_seed(42)


def generate_random_p(Vh, num_samples, seed=None):
    """
    Generate a piecewise-constant DG0 coefficient field with random values
    inside 16 small squares on the unit square domain.

    Parameters
    ----------
    Vh : dolfinx.fem.FunctionSpace
    num_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.    

    Returns
    -------
    batch_p  : list of dolfinx.fem.Function
        List of DG0 functions containing the generated fields.
    batch_mu : list of np.ndarray
        List of the random exponents used for each sample.
    """
    if seed is not None:
        np.random.seed(seed)

    batch_p = []
    batch_mu = []
    mesh = Vh.mesh
    for i in tqdm(range(num_samples)):
        p = dolfinx.fem.Function(Vh)

        # Random exponents μ_i ~ U(-1,1)
        mu = np.random.uniform(-1.0, 1.0, 16)

        # Define squares (centered on (m/8, n/8) with half-width 1/16)
        squares = []
        for m in [1, 3, 5, 7]:
            for n in [1, 3, 5, 7]:
                x_min, x_max = m / 8 - 1 / 16, m / 8 + 1 / 16
                y_min, y_max = n / 8 - 1 / 16, n / 8 + 1 / 16
                squares.append(((x_min, x_max), (y_min, y_max)))

        values = np.ones(len(p.x.array))  # default value = 1

        # Cell midpoints
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim, 0)
        x = mesh.geometry.x
        c_to_v = mesh.topology.connectivity(tdim, 0)

        for cell in range(mesh.topology.index_map(tdim).size_local):
            verts = c_to_v.links(cell)
            midpoint = x[verts].mean(axis=0)
            mx, my, _ = midpoint
            # check squares
            for j, ((x_min, x_max), (y_min, y_max)) in enumerate(squares):
                if (x_min <= mx <= x_max) and (y_min <= my <= y_max):
                    values[cell] = 10.0 ** mu[j]
                    break

        p.x.array[:] = values
        batch_p.append(p)
        batch_mu.append(mu)
    return batch_p, batch_mu



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the permeability (setup1) for the Poisson problem.') 
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--input_probability_measure_config_path', type=str, help='Path to the input probability measure configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the train dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')

    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    input_probability_measure_args = load_yaml(args.input_probability_measure_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    num_train = input_probability_measure_args['num_train']
    num_test = input_probability_measure_args['num_test']
    num_samples = num_train + num_test

    poisson_setup1_least_squares = PoissonSetup1LeastSquares(mesh_args, function_space_args)

    mesh = poisson_setup1_least_squares.mesh
    Vh = poisson_setup1_least_squares.Vh

    p_dof_dim = dolfinx.fem.Function(Vh['p']).x.array.shape[0]

    if rank == 0:
        split_num_functions = get_split(N=num_samples, size=size)
        counts = [num * p_dof_dim for num in split_num_functions]
        displacements = [sum(counts[:i]) for i in range(size)]
    else:
        split_num_functions = None
        counts = None
        displacements = None

    split_num_functions = comm.bcast(split_num_functions, root=0)
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    local_seed = rank # Different seed for each processor

    if rank == 0:
        start_time = MPI.Wtime()


    local_p_dof = np.zeros((split_num_functions[rank], p_dof_dim), dtype=np.float64)
    batch_p, batch_mu = generate_random_p(Vh['p'], num_samples=split_num_functions[rank], seed=local_seed)
    for i in range(split_num_functions[rank]):
        local_p_dof[i, :] = batch_p[i].x.array

    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Elapsed time (rank 0 | {split_num_functions[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    if rank == 0: 
        p_dof = np.zeros([num_samples, p_dof_dim], dtype='float64')
    else:
        p_dof = None

    comm.Gatherv(local_p_dof, [p_dof, counts, displacements, MPI.DOUBLE], root=0)
    if rank == 0:
        assert p_dof.shape == (num_samples, p_dof_dim)

        # Use np.unique along axis=0
        unique_data, indices, counts = np.unique(p_dof, axis=0, return_index=True, return_counts=True)

        # Check duplicates
        has_duplicates = np.any(counts > 1)
        print("Has duplicates:", has_duplicates)

        np.save(train_dataset_path + '/p_dof.npy', p_dof[:num_train, :])
        np.save(test_dataset_path + '/p_dof.npy', p_dof[num_train:, :])
        print(f'Saved to {train_dataset_path}/p_dof.npy and {test_dataset_path}/p_dof.npy')