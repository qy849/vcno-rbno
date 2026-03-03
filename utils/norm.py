import dolfinx
import ufl
import numpy as np
from mpi4py import MPI

def norm_L2(uh):
    mesh = uh.function_space.mesh

    form = dolfinx.fem.form(ufl.inner(uh, uh) * ufl.dx)
    local_L2_square = dolfinx.fem.assemble_scalar(form)
    global_L2_square = mesh.comm.allreduce(local_L2_square, op=MPI.SUM)
    
    return np.sqrt(global_L2_square)