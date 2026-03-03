import os
import sys
import argparse

import numpy as np

import dolfinx
import dolfinx.fem.petsc

from mpi4py import MPI


import torch
from petsc4py import PETSc


repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import PoissonSetup1LeastSquares
from utils import load_yaml, format_elapsed_time, load_and_scatter, gather_and_save, convert_petsc_mat_to_torch_sparse_coo_tensor
from tqdm import tqdm


def convert_weight_to_tensor(weight, dtype):
    weight_tensor = {'A00': None, 'A01': None, 'A11': None}
    weight_tensor['A00'] = convert_petsc_mat_to_torch_sparse_coo_tensor(weight['A00'], dtype=dtype)
    weight_tensor['A01'] = torch.tensor(weight['A01'].getArray(),dtype=dtype)
    weight_tensor['A11'] = torch.tensor(weight['A11'], dtype=dtype)
    return weight_tensor

def compute_reduced_weight(weight: dict[torch.Tensor], pod_basis_dof: torch.Tensor)-> torch.Tensor:

    quadratic_weight = pod_basis_dof.T @ weight['A00'] @ pod_basis_dof
    linear_weight = pod_basis_dof.T @ weight['A01']
    bias = weight['A11']

    reduced_weight = {'quadratic': quadratic_weight, 'linear': linear_weight, 'bias': bias}

    return reduced_weight


def compute_reduced_weight_v2(weight,  pod_basis_dof_petsc, pod_basis_dof_petsc_T) -> dict[np.ndarray]:
    """
    Compute the reduced weight using the POD basis in PETSc format.
    """
    temp = weight['A00'].matMult(pod_basis_dof_petsc)
    quadratic_weight = pod_basis_dof_petsc_T.matMult(temp)
    linear_weight = pod_basis_dof_petsc_T.createVecLeft()
    pod_basis_dof_petsc_T.mult(weight['A01'], linear_weight)
    bias = weight['A11']

    def mat_to_numpy(mat: PETSc.Mat):
        m, n = mat.getSize()
        arr = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                arr[i, j] = mat.getValue(i, j)
        return arr

    quadratic_weight = mat_to_numpy(quadratic_weight)
    linear_weight = linear_weight.getArray(readonly=True)
    reduced_weight = {'quadratic': quadratic_weight, 'linear': linear_weight, 'bias': bias}

    return reduced_weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the reduced weights.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--output_reduced_basis_config_path', type=str, help='Path to the output reduced basis configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')


    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    output_reduced_basis_args = load_yaml(args.output_reduced_basis_config_path)
    dataset_path = args.dataset_path

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    poisson_least_squares = PoissonSetup1LeastSquares(mesh_args, function_space_args)

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    dtype = 'float64'
    tensor_dtype = torch.float64

    local_p_dof, split_num_functions = load_and_scatter(comm, dataset_path+'/p_dof.npy',dtype=dtype)


    if rank == 0:
        pod_basis_dof = np.load(dataset_path + '/hdiv_h1_pod_basis_dof.npy')
        shape = pod_basis_dof.shape
        dtype = pod_basis_dof.dtype
    else:
        pod_basis_dof = None
        shape = None
        dtype = None

    # Broadcast shape and dtype
    shape = comm.bcast(shape, root=0)
    dtype = comm.bcast(dtype, root=0)

    # Allocate buffer on non-root ranks
    if rank != 0:
        pod_basis_dof = np.empty(shape, dtype=dtype)

    # Broadcast the array data
    comm.Bcast(pod_basis_dof, root=0)

    pod_basis_dof_petsc = PETSc.Mat().createDense(size=pod_basis_dof.shape, array=pod_basis_dof, comm=PETSc.COMM_SELF)
    pod_basis_dof_petsc_T = PETSc.Mat().createDense(size=(pod_basis_dof.shape[1], pod_basis_dof.shape[0]), array=pod_basis_dof.T, comm=PETSc.COMM_SELF)
    pod_basis_dof_petsc.assemblyBegin()
    pod_basis_dof_petsc.assemblyEnd()
    pod_basis_dof_petsc_T.assemblyBegin()
    pod_basis_dof_petsc_T.assemblyEnd()


    local_quadratic_weight = np.zeros((split_num_functions[rank], output_reduced_basis_args['num_basis'], output_reduced_basis_args['num_basis']), dtype=dtype)
    local_linear_weight = np.zeros((split_num_functions[rank], output_reduced_basis_args['num_basis']), dtype=dtype)
    local_bias = np.zeros((split_num_functions[rank], 1), dtype=dtype)
    

    pod_basis_dof = torch.tensor(pod_basis_dof, dtype=tensor_dtype)
    if rank == 0:
        print(f'pod_basis_dof shape: {pod_basis_dof.shape}')

    if rank == 0:
        start_time = MPI.Wtime()

    for i in tqdm(range(split_num_functions[rank])):
        p = dolfinx.fem.Function(poisson_least_squares.Vh['p'])
        p.x.array[:] = local_p_dof[i]
        weight = poisson_least_squares.compute_weight(p)
        # weight = convert_weight_to_tensor(weight, dtype=tensor_dtype)
        # reduced_weight = compute_reduced_weight(weight, pod_basis_dof)
        reduced_weight = compute_reduced_weight_v2(weight, pod_basis_dof_petsc, pod_basis_dof_petsc_T)
        local_quadratic_weight[i] = reduced_weight['quadratic']
        local_linear_weight[i] = reduced_weight['linear']
        local_bias[i] = reduced_weight['bias']
 
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Elapsed time (rank 0 | {split_num_functions[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')


    gather_and_save(comm, dataset_path+'/hdiv_h1_quadratic_weight.npy', local_quadratic_weight, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/hdiv_h1_linear_weight.npy', local_linear_weight, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/hdiv_h1_bias.npy', local_bias, split_num_functions, dtype=dtype)

    if rank == 0:
        print(f'Reduced weights saved to {dataset_path}/hdiv_h1_quadratic_weight.npy, {dataset_path}/hdiv_h1_linear_weight.npy, {dataset_path}/hdiv_h1_bias.npy')