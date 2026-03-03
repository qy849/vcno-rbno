import os
import sys
import argparse

import numpy as np
import ufl
from mpi4py import MPI
import basix
import dolfinx
import dolfinx.fem.petsc
from scifem import create_real_functionspace

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, load_npy, save_npy, format_elapsed_time, load_and_scatter, gather_and_save, timing, evaluate_expression
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import time

from data_generation.differential_equations import PoissonSetup1LeastSquares


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the auxiliary PDE to the Poisson equation.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    poisson_least_squares = PoissonSetup1LeastSquares(mesh_args, function_space_args)
    mesh = poisson_least_squares.mesh
    Vh = poisson_least_squares.Vh

    print(f'Running: {sys.argv[0]}')

    start_time = time.time()
    q = poisson_least_squares.solve_q()
    w = poisson_least_squares.solve_w()
    print(f"Auxiliary PDE solve time taken: {time.time() - start_time:.3f} seconds")

    q.name = "q"
    w.name = "w"

    np.save(os.path.join(train_dataset_path, "aux_q_dof.npy"), q.x.array)
    np.save(os.path.join(train_dataset_path, "aux_w_dof.npy"), w.x.array)
    np.save(os.path.join(test_dataset_path, "aux_q_dof.npy"), q.x.array)
    np.save(os.path.join(test_dataset_path, "aux_w_dof.npy"), w.x.array)
    print(f'Saved to {train_dataset_path}/aux_q_dof.npy, {train_dataset_path}/aux_w_dof.npy, {test_dataset_path}/aux_q_dof.npy, {test_dataset_path}/aux_w_dof.npy')

    print('Plotting auxiliary variables...')
    x = mesh.geometry.x[:, 0]
    y = mesh.geometry.x[:, 1]

    # Common style parameters
    title_fontsize = 20
    axis_fontsize = 16
    cbar_fontsize = 16

    q_evals = evaluate_expression(mesh, q, mesh.geometry.x)[1]
    w_evals = evaluate_expression(mesh, w, mesh.geometry.x)[1]
    print(f'q_evals shape: {q_evals.shape}')
    print(f'w_evals shape: {w_evals.shape}')

    z = ufl.grad(q)
    z_evals = evaluate_expression(mesh, z, mesh.geometry.x)[1]
    print(f'z_evals shape: {z_evals.shape}')


    # q
    plt.tricontourf(x, y, q_evals[:, 0], cmap='turbo', levels=100)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    cbar.ax.tick_params(labelsize=cbar_fontsize)  # colorbar ticks size
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.title(r"$q$", fontsize=title_fontsize)
    plt.savefig(os.path.join(train_dataset_path, "q_poisson_setup1.png"), dpi=300, bbox_inches="tight")
    plt.close()


    # w
    plt.tricontourf(x, y, w_evals[:, 0], cmap='turbo', levels=100)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.title(r"$w$", fontsize=title_fontsize)
    plt.savefig(os.path.join(train_dataset_path, "w_poisson_setup1.png"), dpi=300, bbox_inches="tight")
    plt.close()  
    print(f'Plots saved to {train_dataset_path}/q_poisson_setup1.png and {train_dataset_path}/w_poisson_setup1.png')


    # z1
    plt.tricontourf(x, y, z_evals[:, 0], cmap='turbo', levels=100)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=cbar_fontsize)  # colorbar ticks size
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.title(r"$z_{1}$", fontsize=title_fontsize)
    plt.savefig(os.path.join(train_dataset_path, "z1_poisson_setup1.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # z2
    plt.tricontourf(x, y, z_evals[:, 1], cmap='turbo', levels=100)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=cbar_fontsize)  # colorbar ticks size
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.title(r"$z_{2}$", fontsize=title_fontsize)
    plt.savefig(os.path.join(train_dataset_path, "z2_poisson_setup1.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f'Plots saved to {train_dataset_path}/z1_poisson_setup1.png and {train_dataset_path}/z2_poisson_setup1.png')