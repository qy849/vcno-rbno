import dolfinx
import numpy as np
from mpi4py import MPI

def evaluate_expression(mesh, expr, points):
    # Determine what process owns a point and what cells it lies within
    # _, _, owning_points, cells = cpp.geometry.determine_point_ownership(
    #     mesh._cpp_object, points, 1e-6
    # )
    point_ownership_data = dolfinx.cpp.geometry.determine_point_ownership(mesh._cpp_object, points, 1e-6)
    owning_points = np.asarray(point_ownership_data.dest_points).reshape(-1, 3)
    cells = point_ownership_data.dest_cells
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmap
    ref_x = np.zeros((len(cells), mesh.geometry.dim), dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    data_size = int(np.prod(expr.ufl_shape))
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(expr, ref_x, comm=MPI.COMM_SELF)
        values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))
        # strip basis_values per cell
        basis_values = np.empty((len(cells), data_size), dtype=dolfinx.default_scalar_type)
        for i in range(len(cells)):
            basis_values[i] = values[i, i * data_size : (i + 1) * data_size]
    else:
        basis_values = np.zeros((0, data_size), dtype=dolfinx.default_scalar_type)
    return cells, basis_values