import ufl
import sys

import numpy as np
import matplotlib.pyplot as plt

import dolfinx
import ufl
from mpi4py import MPI
import basix.ufl
import dolfinx.fem.petsc
from dolfinx.plot import vtk_mesh
import pyvista as pv
import vtk
from scipy.sparse import coo_matrix, bmat
from matplotlib.patches import Rectangle


# vtk_mathtext = vtk.vtkMathTextFreeTypeTextRenderer()
# vtk_mathtext.MathTextIsSupported()


def plot_mesh(mesh, show_edges=True): 

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)

    pv.start_xvfb()
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=show_edges)
    plotter.view_xy()
    if not pv.OFF_SCREEN:
        plotter.show()
    
    return plotter


def plot_real_valued_function(u, V):
    mesh = V.mesh
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    pv.start_xvfb()


    add_mesh_args = {
        "show_edges": False,
        "cmap": "turbo",
        "scalar_bar_args": {"vertical": True, 
                            "position_x": 0.88, 
                            "position_y": 0.08,
                            "height": 0.835,
                            "width": 0.05,
                            "title": "", 
                            "n_labels": 5,
                            "fmt": "%.2f"
                            },
    }

    add_text_args = {
        "position": "upper_edge",
        "font_size": 14,
        "color": "black",
        "font": "times",
    }
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[u.name] = u.x.array.real
    grid.set_active_scalars(u.name)

    plotter = pv.Plotter()
    plotter.add_mesh(grid, **add_mesh_args)
    plotter.add_text(u.name, **add_text_args)
    plotter.view_xy()
    plotter.window_size = [700, 600]  
    plotter.camera.zoom(1.2) 
    if not pv.OFF_SCREEN:
        plotter.show()

    return plotter

def plot_complex_valued_function(u, V):
    mesh = V.mesh
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    pv.start_xvfb()
    
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["real"] = u.x.array.real
    grid.point_data["imag"] = u.x.array.imag
    grid.point_data["abs"] = np.abs(u.x.array)

    add_mesh_args = {
        "show_edges": False,
        "cmap": "turbo",
        "scalar_bar_args": {"vertical": True, 
                            "position_x": 0.88, 
                            "position_y": 0.08,
                            "height": 0.835,
                            "width": 0.05,
                            "title": "", 
                            "n_labels": 5,
                            "fmt": "%.2f"
                            },
    }

    add_text_args = {
        "position": "upper_edge",
        "font_size": 14,
        "color": "black",
        "font": "times",
    }


    def plot_scalar(grid, scalar_name, title):
        grid.set_active_scalars(scalar_name)
        plotter = pv.Plotter()
        plotter.add_mesh(grid, **add_mesh_args)
        plotter.add_text(title, **add_text_args)
        plotter.view_xy()
        plotter.window_size = [700, 600]  # Adjust window size to reduce blank space
        plotter.camera.zoom(1.2)  # Zoom in slightly to fill space
        if not pv.OFF_SCREEN:
            plotter.show()
        return plotter

    plotters = []

    plotters.append(plot_scalar(grid, "real", f"$\\mathrm{{Re}}({u.name})$"))
    plotters.append(plot_scalar(grid, "imag", f"$\\mathrm{{Im}}({u.name})$"))
    plotters.append(plot_scalar(grid, "abs", f"$\\mathrm{{Abs}}({u.name})$"))
    
    return plotters


def plot_eigenvalues(eigenvalues: np.ndarray, title: str):
    indices = np.arange(1, len(eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.log10(indices), np.log10(eigenvalues), color='blue', marker='o', markersize=3)
    ax.set_xlabel(r'$\log_{10}(i)$', fontsize=15)
    ax.set_ylabel(r'$\log_{10}(\lambda_i)$', fontsize=15, rotation=0, labelpad=30)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=15) 
    plt.close()
    return fig



# def plot_block_sparsity(matrix_dict: dict, precision: float=1e-10) -> None:
#     """
#     Reconstructs and plots the sparsity pattern of a block matrix:
#         [ A00  A01 ]
#         [ A10  A11 ]
#     where A10 = A01.T

#     Parameters:
#         matrix_dict (dict): Should include:
#             - 'A00': torch sparse COO tensor (n x n)
#             - 'A01': torch dense tensor (n,)
#             - 'A11': scalar tensor (1,)
#     """
#     A00 = matrix_dict['A00'].coalesce()
#     A01 = matrix_dict['A01']
#     A11 = matrix_dict['A11']

#     # Convert A00 to SciPy COO
#     A00_scipy = coo_matrix(
#         (A00.values().cpu().numpy(), A00.indices().cpu().numpy()),
#         shape=A00.size()
#     )

#     # A01: shape (n,) → (n, 1)
#     A01_np = A01.cpu().numpy().reshape(-1, 1)
#     A01_scipy = coo_matrix(A01_np)

#     # A10 = A01.T
#     A10_scipy = coo_matrix(A01_np.T)

#     # A11: scalar to 1x1 sparse
#     A11_scipy = coo_matrix([[A11.item()]])

#     # Assemble full matrix
#     A_full = bmat([[A00_scipy, A01_scipy],
#                    [A10_scipy, A11_scipy]], format='coo')

#     # Plot sparsity pattern
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111)
#     ax.spy(A_full, markersize=0.002, precision=precision)
#     ax.set_title(r'Sparsity pattern of $W$', fontsize=15)
#     ax.set_xlabel('Column index', fontsize=15)
#     ax.set_ylabel('Row index', fontsize=15)

#     ax.tick_params(axis='both', which='major', labelsize=15)
#     fig.tight_layout()
#     return fig


# def plot_block_submatrix_sparsity(matrix_dict: dict, row_range: tuple, col_range: tuple, precision: float=1e-10) -> None:
#     """
#     Visualizes the sparsity pattern of a submatrix of the full block matrix:
#         [ A00  A01 ]
#         [ A10  A11 ]
#     where A10 = A01.T

#     Parameters:
#         matrix_dict (dict): Contains:
#             - 'A00': torch sparse COO tensor (n x n)
#             - 'A01': torch dense tensor (n,)
#             - 'A11': scalar tensor (1,)
#         row_range (tuple): (start_row, end_row) for the submatrix
#         col_range (tuple): (start_col, end_col) for the submatrix
#     """
#     A00 = matrix_dict['A00'].coalesce()
#     A01 = matrix_dict['A01']
#     A11 = matrix_dict['A11']

#     # Convert A00 to SciPy COO
#     A00_scipy = coo_matrix(
#         (A00.values().cpu().numpy(), A00.indices().cpu().numpy()),
#         shape=A00.size()
#     )

#     # Convert A01 to sparse column
#     A01_np = A01.cpu().numpy().reshape(-1, 1)
#     A01_scipy = coo_matrix(A01_np)
#     A10_scipy = coo_matrix(A01_np.T)
#     A11_scipy = coo_matrix([[A11.item()]])

#     # Assemble full block matrix and convert to CSR for slicing
#     A_full = bmat([[A00_scipy, A01_scipy],
#                    [A10_scipy, A11_scipy]], format='csr')

#     # Slice the desired submatrix
#     r0, r1 = row_range
#     c0, c1 = col_range
#     A_sub = A_full[r0:r1, c0:c1]

#     # Plot
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111)
#     ax.spy(A_sub, markersize=1.0, precision=precision)
#     ax.set_title(rf'Sparsity pattern of $W[{r0}:{r1}, {c0}:{c1}]$', fontsize=15)
#     ax.set_xlabel('Column index', fontsize=15)
#     ax.set_ylabel('Row index', fontsize=15)

#     ax.tick_params(axis='both', which='major', labelsize=15)
#     fig.tight_layout()
#     return fig


def plot_block_sparsity(matrix_dict: dict, 
                        precision: float = 1e-10,
                        highlight_box: tuple = None, 
                        box_color='red', 
                        box_linewidth=2, 
                        box_alpha=0.5) -> None:
    """
    Reconstructs and plots the sparsity pattern of a block matrix:
        [ A00  A01 ]
        [ A10  A11 ]
    where A10 = A01.T

    Parameters:
        matrix_dict (dict): Should include:
            - 'A00': torch sparse COO tensor (n x n)
            - 'A01': torch dense tensor (n,)
            - 'A11': scalar tensor (1,)
        highlight_box (tuple): (row_start, row_end, col_start, col_end) for annotation.
        box_color (str): Color of the box.
        box_linewidth (int): Width of the box edge.
    """
    # Extract components
    A00 = matrix_dict['A00'].coalesce()
    A01 = matrix_dict['A01']
    A11 = matrix_dict['A11']

    # Convert A00 to SciPy COO
    A00_scipy = coo_matrix(
        (A00.values().cpu().numpy(), A00.indices().cpu().numpy()),
        shape=A00.size()
    )

    # A01: shape (n,) → (n, 1)
    A01_np = A01.cpu().numpy().reshape(-1, 1)
    A01_scipy = coo_matrix(A01_np)

    # A10 = A01.T
    A10_scipy = coo_matrix(A01_np.T)

    # A11: scalar to 1x1 sparse
    A11_scipy = coo_matrix([[A11.item()]])

    # Assemble full matrix
    A_full = bmat([[A00_scipy, A01_scipy],
                   [A10_scipy, A11_scipy]], format='coo')

    # Plot sparsity pattern
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.spy(A_full, markersize=0.002, precision=precision)
    ax.set_title(r'Sparsity pattern of $W$', fontsize=14)
    ax.set_xlabel('Column index', fontsize=15)
    ax.set_ylabel('Row index', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Add rectangle if specified
    if highlight_box:
        r_start, r_end, c_start, c_end = highlight_box
        width = c_end - c_start
        height = r_end - r_start
        rect = Rectangle((c_start, r_start), width, height,
                         fill=True, edgecolor=box_color, facecolor=box_color, linewidth=box_linewidth, alpha=box_alpha)
        ax.add_patch(rect)

    ax.ticklabel_format(style='plain', axis='both')
    # ax.xaxis.offsetText.set_fontsize(14)  # for x-axis
    # ax.yaxis.offsetText.set_fontsize(14)  # for y-axis

    # # Move offset text away from ticks to prevent overlap
    # ax.xaxis.offsetText.set_position((0.15, 1.0))  # (x, y) in axes coordinates
    # ax.yaxis.offsetText.set_position((-0.2, 0.0))

    fig.tight_layout()

    return fig



def plot_block_submatrix_sparsity(
    matrix_dict: dict,
    row_range: tuple,
    col_range: tuple,
    precision: float = 1e-10,
    highlight_box: tuple = None,
    box_color: str = 'red',
    box_linewidth: float = 2,
    box_alpha: float = 0.5
) -> None:
    """
    Visualizes the sparsity pattern of a submatrix of the full block matrix:
        [ A00  A01 ]
        [ A10  A11 ]
    where A10 = A01.T

    Parameters:
        matrix_dict (dict): Contains:
            - 'A00': torch sparse COO tensor (n x n)
            - 'A01': torch dense tensor (n,)
            - 'A11': scalar tensor (1,)
        row_range (tuple): (start_row, end_row) for the submatrix
        col_range (tuple): (start_col, end_col) for the submatrix
        highlight_box (tuple): (row_start, row_end, col_start, col_end) relative to submatrix
        box_color (str): color of the highlight box
        box_linewidth (float): line width of the highlight box
        box_alpha (float): transparency of the highlight box
    """
    # Extract components
    A00 = matrix_dict['A00'].coalesce()
    A01 = matrix_dict['A01']
    A11 = matrix_dict['A11']

    # Convert A00 to SciPy COO
    A00_scipy = coo_matrix(
        (A00.values().cpu().numpy(), A00.indices().cpu().numpy()),
        shape=A00.size()
    )

    # Convert A01 to sparse column and A10
    A01_np = A01.cpu().numpy().reshape(-1, 1)
    A01_scipy = coo_matrix(A01_np)
    A10_scipy = coo_matrix(A01_np.T)
    A11_scipy = coo_matrix([[A11.item()]])

    # Assemble full matrix and convert to CSR for slicing
    A_full = bmat([[A00_scipy, A01_scipy],
                   [A10_scipy, A11_scipy]], format='csr')

    # Slice submatrix
    r0, r1 = row_range
    c0, c1 = col_range
    A_sub = A_full[r0:r1, c0:c1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spy(A_sub, markersize=1.0, precision=precision)
    ax.set_title(rf'Sparsity pattern of $W[{r0}:{r1},{c0}:{c1}]$', fontsize=14)
    ax.set_xlabel('Column index', fontsize=15)
    ax.set_ylabel('Row index', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.ticklabel_format(style='plain', axis='both')  # disable auto scientific notation

    # Add rectangle if specified
    if highlight_box:
        hr0, hr1, hc0, hc1 = highlight_box
        width = hc1 - hc0
        height = hr1 - hr0
        rect = Rectangle(
            (hc0, hr0), width, height,
            fill=True,
            edgecolor=box_color,
            facecolor=box_color,
            linewidth=box_linewidth,
            alpha=box_alpha
        )
        ax.add_patch(rect)

    fig.tight_layout()
    return fig
