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

from utils import load_yaml, load_npy, save_npy, format_elapsed_time, load_and_scatter, gather_and_save, timing 
from tqdm import tqdm

class PoissonSetup1LeastSquares:
    def __init__(self, mesh_args, function_space_args):
        self.mesh_args = mesh_args
        self.function_space_args = function_space_args

        self._mesh = None
        self._Vh = None
        self._bcs_sigma_u = None
        self._bcs_q = None
        self._bcs_w = None
        self._mark_id = None
        self._ds = None
        self.q = None
        self.w = None
     
    @property
    def mesh(self):
        if self._mesh is None:
            comm = MPI.COMM_SELF
            mesh_args = self.mesh_args

            point_coords = [[mesh_args["lower_left_x"], mesh_args["lower_left_y"]], 
                           [mesh_args["upper_right_x"], mesh_args["upper_right_y"]]] 
            num_cells = [mesh_args["num_x"], mesh_args["num_y"]]
            if mesh_args["mesh_cell_type"] == "triangle":
                cell_type = dolfinx.cpp.mesh.CellType.triangle
            elif mesh_args["mesh_cell_type"] == "quadrilateral":
                cell_type = dolfinx.cpp.mesh.CellType.quadrilateral
            else:
                raise ValueError("Unknown cell type")

            self._mesh = dolfinx.mesh.create_rectangle(comm=comm, points=point_coords, n=num_cells, cell_type=cell_type)
        return self._mesh

    @property
    def Vh(self):
        if self._Vh is None:
            mesh = self.mesh
            mesh_args = self.mesh_args
            function_space_args = self.function_space_args

            p_element = basix.ufl.element(family=function_space_args["p"]["family"],
                                          cell=mesh_args["mesh_cell_type"],  
                                          degree=function_space_args["p"]["degree"])
            u_element = basix.ufl.element(family=function_space_args["u"]["family"],
                                          cell=mesh_args["mesh_cell_type"],  
                                          degree=function_space_args["u"]["degree"])
            sigma_element = basix.ufl.element(family=function_space_args["sigma"]["family"],
                                              cell=mesh_args["mesh_cell_type"],  
                                              degree=function_space_args["sigma"]["degree"])
            sigma_u_element = basix.ufl.mixed_element([sigma_element, u_element])

            # for auxiliary variables w and q
            w_element = basix.ufl.element(family=function_space_args["w"]["family"],
                                          cell=mesh_args["mesh_cell_type"],  
                                          degree=function_space_args["w"]["degree"])
            q_element = basix.ufl.element(family=function_space_args["q"]["family"],
                                          cell=mesh_args["mesh_cell_type"],  
                                          degree=function_space_args["q"]["degree"])

            CG1_element = basix.ufl.element(family="CG", cell=mesh_args["mesh_cell_type"], degree=1)
            CG1_vector_element = basix.ufl.element(family="CG", cell=mesh_args["mesh_cell_type"], degree=1, shape=(2,))
            CG1_vector_scalar_element = basix.ufl.mixed_element([CG1_vector_element, CG1_element])

            DG0_element = basix.ufl.element(family="DG", cell=mesh_args["mesh_cell_type"], degree=0)

            self._Vh = {
                'p': dolfinx.fem.functionspace(mesh, p_element),
                'sigma_u': dolfinx.fem.functionspace(mesh, sigma_u_element),
                'w': dolfinx.fem.functionspace(mesh, w_element),
                'q': dolfinx.fem.functionspace(mesh, q_element), 
                'CG1': dolfinx.fem.functionspace(mesh, CG1_element), 
                'CG1_vector': dolfinx.fem.functionspace(mesh, CG1_vector_element),
                'CG1_vector_scalar': dolfinx.fem.functionspace(mesh, CG1_vector_scalar_element),
                'DG0': dolfinx.fem.functionspace(mesh, DG0_element)
            }
            self._Vh['sigma'] = self._Vh['sigma_u'].sub(0).collapse()[0]
            self._Vh['u'] = self._Vh['sigma_u'].sub(1).collapse()[0]

        return self._Vh


    @property
    def bcs_sigma_u(self):
        if self._bcs_sigma_u is None:
            mesh = self.mesh
            Vh = self.Vh

            topology_dim = mesh.topology.dim
            facet_dim = topology_dim - 1

            # zero Dirichlet BC on top and bottom boundaries for sigma 
            facets_top = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[1], 1.0))
            dofs_top = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(0), Vh['sigma']), facet_dim, facets_top)
            f_top = dolfinx.fem.Function(Vh['sigma'])
            bc_top = dolfinx.fem.dirichletbc(f_top, dofs_top, Vh['sigma_u'].sub(0))

            facets_bottom = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[1], 0.0))
            dofs_bottom = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(0), Vh['sigma']), facet_dim, facets_bottom)
            f_bottom = dolfinx.fem.Function(Vh['sigma'])
            bc_bottom = dolfinx.fem.dirichletbc(f_bottom, dofs_bottom, Vh['sigma_u'].sub(0))

            # zero Dirichlet BC on left and right boundaries for u (value)
            facets_left = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 0.0))
            dofs_left = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(1), Vh['u']), facet_dim, facets_left)
            f_left = dolfinx.fem.Function(Vh['u'])
            bc_left = dolfinx.fem.dirichletbc(f_left, dofs_left, Vh['sigma_u'].sub(1))

            facets_right = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 1.0))
            dofs_right = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(1), Vh['u']), facet_dim, facets_right)
            f_right = dolfinx.fem.Function(Vh['u'])
            bc_right = dolfinx.fem.dirichletbc(f_right, dofs_right, Vh['sigma_u'].sub(1))

            self._bcs_sigma_u = [bc_top, bc_bottom, bc_left, bc_right]
        return self._bcs_sigma_u

    @property
    def bcs_q(self):
        if self._bcs_q is None:
            mesh = self.mesh
            Vh = self.Vh

            # zero Dirichlet BC on left and right boundaries
            dofs_left = dolfinx.fem.locate_dofs_geometrical(Vh['q'], lambda x: np.isclose(x[0], 0.0))
            f_left = dolfinx.fem.Function(Vh['q']) 
            bc_left = dolfinx.fem.dirichletbc(f_left, dofs_left)

            dofs_right = dolfinx.fem.locate_dofs_geometrical(Vh['q'], lambda x: np.isclose(x[0], 1.0))
            f_right = dolfinx.fem.Function(Vh['q']) 
            bc_right = dolfinx.fem.dirichletbc(f_right, dofs_right)

            self._bcs_q = [bc_left, bc_right]
        return self._bcs_q


    @property
    def bcs_w(self):
        if self._bcs_w is None:
            mesh = self.mesh
            Vh = self.Vh
            u0_np = self.u0_np
            
            dofs_left = dolfinx.fem.locate_dofs_geometrical(Vh['w'], lambda x: np.isclose(x[0], 0.0))
            f_left = dolfinx.fem.Function(Vh['w']) 
            f_left.interpolate(u0_np)
            bc_left = dolfinx.fem.dirichletbc(f_left, dofs_left)


            dofs_right = dolfinx.fem.locate_dofs_geometrical(Vh['w'], lambda x: np.isclose(x[0], 1.0))
            f_right = dolfinx.fem.Function(Vh['w']) 
            f_right.interpolate(u0_np)
            bc_right = dolfinx.fem.dirichletbc(f_right, dofs_right)

            self._bcs_w = [bc_left, bc_right]
        return self._bcs_w


    @property
    def mark_id(self):
        if self._mark_id is None:
            self._mark_id = {"left_right": 0, "bottom_top": 1}
        return self._mark_id

    @property
    def ds(self):  
        if self._ds is None:
            mesh = self.mesh   
            mark_id = self.mark_id    

            boundaries = [
                (mark_id["left_right"], lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1)),  # left and right
                (mark_id["bottom_top"], lambda x: np.isclose(x[1], 0) | np.isclose(x[1], 1))  # bottom and top
            ]
            facet_indices, facet_markers = [], []
            fdim = mesh.topology.dim - 1
            for (marker, locator) in boundaries:
                facets = dolfinx.mesh.locate_entities(mesh, fdim, locator)
                facet_indices.append(facets)
                facet_markers.append(np.full_like(facets, marker))
            facet_indices = np.hstack(facet_indices).astype(np.int32)
            facet_markers = np.hstack(facet_markers).astype(np.int32)
            sorted_facets = np.argsort(facet_indices)
            facet_tag = dolfinx.mesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

            self._ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

        return self._ds

    def f1(self):
       # --- indicator χ = 1 on ⋃ Omega_i, 0 elsewhere  (DG0 / P0 space)
        Vh = self.Vh
        chi = dolfinx.fem.Function(Vh['DG0'])

        h = 1.0/16.0
        centers = np.array([(m/8.0, n/8.0) for m in (1,3,5,7) for n in (1,3,5,7)], dtype=float)

        def chi_callable(x):
            # x shape: (2, npts)
            X, Y = x[0], x[1]
            inside = np.zeros(X.size, dtype=bool)
            for cx, cy in centers:
                inside |= (np.abs(X - cx) <= h) & (np.abs(Y - cy) <= h)
            return inside.astype(np.float64)

        chi.interpolate(chi_callable)

        return ufl.as_vector((0.5*chi, -0.5*chi))

    def f2(self):
        mesh = self.mesh
        x = ufl.SpatialCoordinate(mesh)
        return dolfinx.fem.Constant(self.mesh, dolfinx.default_scalar_type(1.0)) 

    def g(self):
        mesh = self.mesh
        x = ufl.SpatialCoordinate(mesh)
        return 0.1 * (1 - x[1]) * ufl.cos(2 * ufl.pi * x[0]) 


    def u0(self):
        mesh = self.mesh
        x = ufl.SpatialCoordinate(mesh)
        return 0.1 * (1 - x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    def u0_np(self, x):
        return 0.1 * (1 - x[0]) * np.sin(4 * np.pi * x[1])

    def solve_q(self) -> dolfinx.fem.Function: 
        mesh = self.mesh
        Vh = self.Vh
        bcs_q = self.bcs_q
        mark_id = self.mark_id
        ds = self.ds
        dx = ufl.Measure("dx", mesh)
        g = self.g()

        q = ufl.TrialFunction(Vh['q'])
        v = ufl.TestFunction(Vh['q']) 

        bilinear_form = ufl.inner(ufl.grad(q), ufl.grad(v)) * dx 
        linear_form = ufl.inner(g, v) * ds(mark_id["bottom_top"])

        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=bcs_q, petsc_options=petsc_options)
        q = problem.solve()
        q.name = "q"
        return q

    def solve_w(self) -> dolfinx.fem.Function:
        mesh = self.mesh
        Vh = self.Vh
        bcs_w = self.bcs_w
        dx = ufl.Measure("dx", mesh)

        w = ufl.TrialFunction(Vh['w'])
        v = ufl.TestFunction(Vh['w'])

        bilinear_form = ufl.inner(ufl.grad(w), ufl.grad(v)) * dx 
        linear_form = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), v) * dx
        
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu", 
            "pc_factor_mat_solver_type": "mumps",
        }

        problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=bcs_w, petsc_options=petsc_options)
        w = problem.solve()
        w.name = "w"
        return w


    def solve_sigma_u(self, p: dolfinx.fem.Function) -> dolfinx.fem.Function:
        mesh = self.mesh
        Vh = self.Vh
        bcs_sigma_u = self.bcs_sigma_u
        f1 = self.f1()
        f2 = self.f2()
        if self.q is None:
            self.q = self.solve_q()
        if self.w is None:
            self.w = self.solve_w()
        q = self.q
        w = self.w
        z = ufl.grad(q)

        (sigma, u) = ufl.TrialFunctions(Vh['sigma_u'])
        (tau, v) = ufl.TestFunctions(Vh['sigma_u'])


        dx = ufl.Measure("dx", mesh)
   
        bilinear_form = ufl.inner(sigma - p * ufl.grad(u), tau - p * ufl.grad(v)) * dx
        bilinear_form += ufl.inner(ufl.div(sigma), ufl.div(tau)) * dx
                    
        linear_form = ufl.inner(p * ufl.grad(w) - z + f1, tau - p * ufl.grad(v)) * dx
        linear_form += -ufl.inner(f2, ufl.div(tau)) * dx
      
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=bcs_sigma_u, petsc_options=petsc_options)
        sigma_u = problem.solve()
        sigma_u.name = "sigma_u"

        return sigma_u


    def compute_weight(self, p: dolfinx.fem.Function):
        mesh = self.mesh
        Vh = self.Vh
        f1 = self.f1()
        f2 = self.f2()
        if self.q is None:
            self.q = self.solve_q()
        if self.w is None:
            self.w = self.solve_w()
        q = self.q
        w = self.w
        z = ufl.grad(q)


        R = create_real_functionspace(mesh)
        Vh['sigma_u_r'] = ufl.MixedFunctionSpace(Vh['sigma_u'], R)

        sigma_u_trial, r_trial = ufl.TrialFunctions(Vh['sigma_u_r'])
        sigma_u_test, r_test = ufl.TestFunctions(Vh['sigma_u_r'])

        sigma_trial, u_trial = ufl.split(sigma_u_trial)
        sigma_test, u_test = ufl.split(sigma_u_test)


        dx = ufl.Measure("dx", domain=mesh)

        constant_term = - (p * ufl.grad(w) - z + f1)

        a00 = ufl.inner(sigma_trial - p * ufl.grad(u_trial), sigma_test - p * ufl.grad(u_test)) * dx
        a01 = ufl.inner(r_trial * constant_term, sigma_test - p * ufl.grad(u_test)) * dx
        # a10 = ufl.inner(sigma_trial - p * ufl.grad(u_trial), r_test * constant_term) * dx
        a11 = ufl.inner(r_trial * constant_term, r_test * constant_term) * ufl.dx

        a00 += ufl.inner(ufl.div(sigma_trial), ufl.div(sigma_test)) * dx
        a01 += ufl.inner(r_trial * f2, ufl.div(sigma_test)) * dx
        # a10 += ufl.inner(ufl.div(sigma_trial), r_test * f2) * dx
        a11 += ufl.inner(r_trial * f2, r_test * f2) * ufl.dx

        # Assemble individual blocks
        A00 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a00))
        A01 = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(a01))
        # A10 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a10))
        A11 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(a11))

        A00.assemble()
        A01.assemble()
        # A10.assemble()
        # A11.assemble()

        A = {
            'A00': A00,
            'A01': A01,
            # 'A10': A10,
            'A11': A11
        }

        return A


    def compute_physical_loss_1(self, sigma, u, p):
        if self.q is None:
            self.q = self.solve_q()
        if self.w is None:
            self.w = self.solve_w()
        q = self.q 
        w = self.w
        z = ufl.grad(q)
        f1 = self.f1()

        temp = sigma - (p * ufl.grad(u) + p * ufl.grad(w) - z + f1)
        loss = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(temp, temp) * ufl.dx))

        return loss

    def compute_physical_loss_2(self, sigma, u, p):
        f2 = self.f2()
        temp = ufl.div(sigma) + f2
        loss = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(temp, temp) * ufl.dx))
        return loss


    def compute_hdiv_h1_norm(self, sigma, u): 
        integral = ufl.inner(sigma, sigma) * ufl.dx + ufl.inner(u, u) * ufl.dx
        integral += ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        integral += ufl.inner(ufl.div(sigma), ufl.div(sigma)) * ufl.dx
        return np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)))

    def compute_squared_hdiv_h1_norm(self, sigma, u): 
        integral = ufl.inner(sigma, sigma) * ufl.dx + ufl.inner(u, u) * ufl.dx
        integral += ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        integral += ufl.inner(ufl.div(sigma), ufl.div(sigma)) * ufl.dx
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral))


    def compute_L2_norm(self, func): 
        integral = ufl.inner(func, func)*ufl.dx
        return np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)))


    def compute_squared_L2_norm(self, func): 
        integral = ufl.inner(func, func)*ufl.dx
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral))



    def project_Hdiv_H1(self, sigma_u_hat: dolfinx.fem.Function) -> dolfinx.fem.Function:
        Vh = self.Vh
        mesh = self.mesh
        bcs_sigma_u = self.bcs_sigma_u

        (sigma, u) = ufl.TrialFunctions(Vh['sigma_u'])
        (tau, v) = ufl.TestFunctions(Vh['sigma_u'])


        sigma_hat, u_hat = ufl.split(sigma_u_hat)

        bilinear_form = ufl.inner(sigma, tau) * ufl.dx + ufl.inner(ufl.div(sigma), ufl.div(tau)) * ufl.dx
        bilinear_form += ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        linear_form = ufl.inner(sigma_hat, tau) * ufl.dx + ufl.inner(ufl.div(sigma_hat), ufl.div(tau)) * ufl.dx
        linear_form += ufl.inner(u_hat, v) * ufl.dx + ufl.inner(ufl.grad(u_hat), ufl.grad(v)) * ufl.dx
        
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=bcs_sigma_u, petsc_options=petsc_options)
        sigma_u = problem.solve()
        sigma_u.name = "sigma_u"

        return sigma_u


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the Poisson problem (with spatially variant permeability) in the least squares form.')
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

    poisson_least_squares = PoissonSetup1LeastSquares(mesh_args=mesh_args, function_space_args=function_space_args)

    sigma_u_dim = dolfinx.fem.Function(poisson_least_squares.Vh['sigma_u']).x.array.shape[0]
    sigma_dim = dolfinx.fem.Function(poisson_least_squares.Vh['sigma']).x.array.shape[0]
    u_dim = dolfinx.fem.Function(poisson_least_squares.Vh['u']).x.array.shape[0]

    dtype = 'float64'

    local_p_dof, split_num_functions = load_and_scatter(comm, dataset_path+'/p_dof.npy',dtype=dtype)
    local_sigma_u_dof = np.zeros((split_num_functions[rank], sigma_u_dim), dtype=dtype)
    local_sigma_dof = np.zeros((split_num_functions[rank], sigma_dim), dtype=dtype)
    local_u_dof = np.zeros((split_num_functions[rank], u_dim), dtype=dtype)

    if rank == 0:
        start_time = MPI.Wtime()
    for i in tqdm(range(split_num_functions[rank])):
        p = dolfinx.fem.Function(poisson_least_squares.Vh['p'], dtype=dtype)
        p.x.array[:] = local_p_dof[i,:]
        sigma_u = poisson_least_squares.solve_sigma_u(p=p)
        sigma = sigma_u.sub(0).collapse()
        u = sigma_u.sub(1).collapse()
        local_sigma_u_dof[i,:] = sigma_u.x.array
        local_sigma_dof[i,:] = sigma.x.array
        local_u_dof[i,:] = u.x.array
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Elapsed time (rank 0 | {split_num_functions[rank]} solves): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    gather_and_save(comm, dataset_path+'/sigma_u_dof_finer.npy', local_sigma_u_dof, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/sigma_dof_finer.npy', local_sigma_dof, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/u_dof_finer.npy', local_u_dof, split_num_functions, dtype=dtype)

    if rank == 0:
        print(f'Saved to {dataset_path}/sigma_u_dof_finer.npy, {dataset_path}/sigma_dof_finer.npy, {dataset_path}/u_dof_finer.npy')