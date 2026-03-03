from sympy.core.mul import mul
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

from utils import load_yaml, load_npy, save_npy, format_elapsed_time, load_and_scatter, gather_and_save, timing 
from tqdm import tqdm


class TractionExact():
    def __init__(self, a=0.6, b=4.0, c=0.3, d=10.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x):
        t1 = self.a * np.exp(- (x[1] - 0.5)**2 / self.b)
        t2 = self.c * (1 + x[1] / self.d)
        return np.stack((t1,t2), axis=0)


class ElasticityLeastSquares:
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
        self._traction = None
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
            sigma_element_component = basix.ufl.element(family=function_space_args["sigma"]["family"], 
                                                        cell=mesh_args["mesh_cell_type"], 
                                                        degree=function_space_args["sigma"]["degree"])
            u_element_component = basix.ufl.element(family=function_space_args["u"]["family"], 
                                                    cell=mesh_args["mesh_cell_type"], 
                                                    degree=function_space_args["u"]["degree"])
            sigma_element = basix.ufl.mixed_element([sigma_element_component, sigma_element_component])
            u_element = basix.ufl.mixed_element([u_element_component, u_element_component])
            sigma_u_element = basix.ufl.mixed_element([sigma_element, u_element])

            # for auxiliary variables w and q
            w_element_component = basix.ufl.element(family=function_space_args["w"]["family"],
                                                    cell=mesh_args["mesh_cell_type"],  
                                                    degree=function_space_args["w"]["degree"])
            q_element_component = basix.ufl.element(family=function_space_args["q"]["family"],
                                                    cell=mesh_args["mesh_cell_type"],  
                                                    degree=function_space_args["q"]["degree"]) 
            w_element = basix.ufl.mixed_element([w_element_component, w_element_component])
            q_element = basix.ufl.mixed_element([q_element_component, q_element_component])
            # self._Vh['w'] = dolfinx.fem.functionspace(mesh,


            CG1_element = basix.ufl.element(family="CG", cell=mesh_args["mesh_cell_type"], degree=1)
            CG1_vector_element = basix.ufl.element(family="CG", cell=mesh_args["mesh_cell_type"], degree=1, shape=(2,))
            CG1_tensor_element = basix.ufl.mixed_element([CG1_vector_element, CG1_vector_element])
            # CG1_tensor_element = basix.ufl.element(family="CG", cell=mesh_args["mesh_cell_type"], degree=1, shape=(2,2))
            CG1_tensor_vector_element = basix.ufl.mixed_element([CG1_tensor_element, CG1_vector_element])

            self._Vh ={
                'p': dolfinx.fem.functionspace(mesh, p_element),
                'sigma_component': dolfinx.fem.functionspace(mesh, sigma_element_component),
                'u_component': dolfinx.fem.functionspace(mesh, u_element_component),
                'sigma': dolfinx.fem.functionspace(mesh, sigma_element),
                'u': dolfinx.fem.functionspace(mesh, u_element),
                'sigma_u': dolfinx.fem.functionspace(mesh, sigma_u_element),
                'w_component': dolfinx.fem.functionspace(mesh, w_element_component),
                'w': dolfinx.fem.functionspace(mesh, w_element),
                'q_component': dolfinx.fem.functionspace(mesh, q_element_component),
                'q': dolfinx.fem.functionspace(mesh, q_element),
                'CG1': dolfinx.fem.functionspace(mesh, CG1_element),
                'CG1_vector': dolfinx.fem.functionspace(mesh, CG1_vector_element),
                'CG1_tensor': dolfinx.fem.functionspace(mesh, CG1_tensor_element),
                'CG1_tensor_vector': dolfinx.fem.functionspace(mesh, CG1_tensor_vector_element),
            }
        return self._Vh

    @property
    def traction(self):
        if self._traction is None:
            mesh = self.mesh
            Vh = self.Vh

            traction_element = basix.ufl.element("CG", "triangle", 5, shape=(2,))
            Vh['traction'] = dolfinx.fem.functionspace(mesh, traction_element)

            traction_exact = TractionExact()
            traction_approx = dolfinx.fem.Function(Vh['traction'])
            traction_approx.interpolate(traction_exact)
            self._traction = traction_approx
        return self._traction


    @property
    def bcs_sigma_u(self):
        if self._bcs_sigma_u is None:
            mesh = self.mesh
            Vh = self.Vh
            mesh_args = self.mesh_args

            topology_dim = mesh.topology.dim
            facet_dim = topology_dim - 1

            facets_top = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[1], mesh_args["upper_right_y"]))
            dofs_top = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(0), Vh['sigma']), facet_dim, facets_top)
            f_top = dolfinx.fem.Function(Vh['sigma'])
            bc_top = dolfinx.fem.dirichletbc(f_top, dofs_top, Vh['sigma_u'].sub(0))


            facets_bottom = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[1], mesh_args["lower_left_y"]))
            dofs_bottom = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(0), Vh['sigma']), facet_dim, facets_bottom)
            f_bottom = dolfinx.fem.Function(Vh['sigma'])
            bc_bottom = dolfinx.fem.dirichletbc(f_bottom, dofs_bottom, Vh['sigma_u'].sub(0))


            facets_right = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], mesh_args["upper_right_x"]))
            dofs_right = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(0), Vh['sigma']), facet_dim, facets_right)
            f_right = dolfinx.fem.Function(Vh['sigma'])
            bc_right = dolfinx.fem.dirichletbc(f_right, dofs_right, Vh['sigma_u'].sub(0))


            facets_left = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], mesh_args["lower_left_x"]))
            dofs_left = dolfinx.fem.locate_dofs_topological((Vh['sigma_u'].sub(1), Vh['u']), facet_dim, facets_left)
            f_left = dolfinx.fem.Function(Vh['u'])
            bc_left = dolfinx.fem.dirichletbc(f_left, dofs_left, Vh['sigma_u'].sub(1))

            self._bcs_sigma_u = [bc_top, bc_bottom, bc_right, bc_left]
        return self._bcs_sigma_u


    @property
    def bcs_q(self):
        if self._bcs_q is None:
            mesh = self.mesh
            Vh = self.Vh
            mesh_args = self.mesh_args

            topology_dim = mesh.topology.dim
            facet_dim = topology_dim - 1

            # zero Dirichlet BC on left
            facets_left = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], mesh_args["lower_left_x"]))
            dofs_left = dolfinx.fem.locate_dofs_topological((Vh['q'], Vh['q']), facet_dim, facets_left)
            f_left = dolfinx.fem.Function(Vh['q']) 
            bc_left = dolfinx.fem.dirichletbc(f_left, dofs_left, Vh['q'])

            self._bcs_q = [bc_left]
        return self._bcs_q

    @property
    def bcs_w(self):
        if self._bcs_w is None:
            mesh = self.mesh
            Vh = self.Vh
            mesh_args = self.mesh_args
            
            u01_np = self.u01_np
            u02_np = self.u02_np

            topology_dim = mesh.topology.dim
            facet_dim = topology_dim - 1
            
            # zero Dirichlet BC on left
            facets_left = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], mesh_args["lower_left_x"]))
            dofs_left = dolfinx.fem.locate_dofs_topological((Vh['w'], Vh['w']), facet_dim, facets_left)
            f_left = dolfinx.fem.Function(Vh['w']) 
            f_left.sub(0).interpolate(u01_np)
            f_left.sub(1).interpolate(u02_np)
            bc_left = dolfinx.fem.dirichletbc(f_left, dofs_left, Vh['w'])

            self._bcs_w = [bc_left]
        return self._bcs_w



    @property
    def mark_id(self):
        if self._mark_id is None:
            self._mark_id =  {"left": 0, "right": 1, "bottom_top": 2}
        return self._mark_id


    @property
    def ds(self):
        if self._ds is None:
            mark_id = self.mark_id
            mesh = self.mesh
            mesh_args = self.mesh_args

            boundaries = [
                        (mark_id["left"], lambda x: np.isclose(x[0], mesh_args["lower_left_x"])),  
                        (mark_id["right"], lambda x: np.isclose(x[0], mesh_args["upper_right_x"])),  
                        (mark_id["bottom_top"], lambda x: np.isclose(x[1], mesh_args["lower_left_y"]) | np.isclose(x[1], mesh_args["upper_right_y"])) 
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


    def f(self):
        mesh = self.mesh
        return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0))) 

    def u0(self):
        mesh = self.mesh
        return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0 ,0.0)))

    def u01_np(self, x):
        return np.zeros((1, x.shape[1]), dtype=np.float64)

    def u02_np(self, x):
        return np.zeros((1, x.shape[1]), dtype=np.float64)

    def get_lame_parameters(self, p: dolfinx.fem.Function):
        mesh = self.mesh
        Vh = self.Vh

        E_0 = 1.0 
        nu = 0.4 # Poisson ratio
        E = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(E_0)) + ufl.exp(p) # Young's modulus
        lambda_ = (E*nu)/((1.0+nu)*(1.0-2.0*nu)) #  Lamé’s first parameter
        mu = E/(2.0*(1.0+nu)) #  Lamé’s second parameter

        return lambda_, mu


    def solve_q(self) -> dolfinx.fem.Function: 
        mesh = self.mesh
        Vh = self.Vh
        bcs_q = self.bcs_q
        mark_id = self.mark_id
        ds = self.ds
        dx = ufl.Measure("dx", mesh)
        t = self.traction

        q = ufl.TrialFunction(Vh['q'])
        v = ufl.TestFunction(Vh['q']) 

        bilinear_form = ufl.inner(ufl.grad(q), ufl.grad(v)) * dx 
        linear_form = ufl.inner(t, v) * ds(mark_id["right"])

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
        linear_form = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0))), v) * dx
        
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
        Vh = self.Vh
        mesh = self.mesh
        bcs_sigma_u = self.bcs_sigma_u
        f = self.f()
        if self.q is None:
            self.q = self.solve_q()
        if self.w is None:
            self.w = self.solve_w()
        q = self.q
        w = self.w
        z = ufl.grad(q)

        (sigma, u) = ufl.TrialFunctions(Vh['sigma_u'])
        (tau, v) = ufl.TestFunctions(Vh['sigma_u'])

        sigma1, sigma2 = ufl.split(sigma)
        sigma_ = ufl.as_vector([sigma1, sigma2])

        tau1, tau2 = ufl.split(tau)
        tau_ = ufl.as_vector([tau1, tau2])

        epsilon_u = ufl.sym(ufl.grad(u))
        epsilon_v = ufl.sym(ufl.grad(v))
        epsilon_w = ufl.sym(ufl.grad(w))

        lambda_, mu = self.get_lame_parameters(p=p)

        def C_inv_action(sigma):
            result = 1/(2*mu) * sigma
            result -=  lambda_/(4*mu*(lambda_+mu)) * ufl.tr(sigma) * ufl.Identity(2)
            return result

        def C_action(epsilon): 
            result = 2*mu*epsilon
            result += lambda_ * ufl.tr(epsilon) * ufl.Identity(2)
            return result

        def asymmetry(sigma):
            e1 = ufl.unit_vector(i=0, d=2)
            e2 = ufl.unit_vector(i=1, d=2)
            result = ufl.inner(sigma[0], -e2) + ufl.inner(sigma[1], e1)
            return result

        dx = ufl.Measure("dx", domain=mesh)

        bilinear_form = ufl.inner(ufl.div(sigma_), ufl.div(tau_)) * dx

        bilinear_form += ufl.inner(C_inv_action(sigma_), tau_) * dx
        bilinear_form -= ufl.inner(epsilon_u, tau_) * dx
        bilinear_form -= ufl.inner(sigma_, epsilon_v) * dx
        bilinear_form += ufl.inner(C_action(epsilon_u), epsilon_v) * dx

        bilinear_form += ufl.inner(asymmetry(sigma_), asymmetry(tau_)) * dx

        linear_form = -ufl.inner(f, ufl.div(tau_)) * dx
        linear_form -= ufl.inner(z, C_inv_action(tau_)) * dx
        linear_form += ufl.inner(epsilon_w, tau_) * dx
        linear_form += ufl.inner(z, epsilon_v) * dx
        linear_form -= ufl.inner(C_action(epsilon_w), epsilon_v) * dx

        q1, q2 = ufl.split(q)
        z_ = ufl.as_vector([ufl.grad(q1), ufl.grad(q2)])
        linear_form -= ufl.inner(asymmetry(z_), asymmetry(tau_)) * dx

        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=bcs_sigma_u, petsc_options=petsc_options)
        sigma_u = problem.solve()

        return sigma_u


    def compute_weight(self, p: dolfinx.fem.Function): 
        mesh = self.mesh
        Vh = self.Vh
        f = self.f()
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

        sigma1_trial, sigma2_trial = ufl.split(sigma_trial)
        sigma1_test, sigma2_test = ufl.split(sigma_test)

        sigma_trial_ = ufl.as_vector((sigma1_trial, sigma2_trial))
        sigma_test_ = ufl.as_vector((sigma1_test, sigma2_test))

        div_sigma_trial_ = ufl.as_vector((ufl.div(sigma1_trial),  ufl.div(sigma2_trial)))
        div_sigma_test_ = ufl.as_vector((ufl.div(sigma1_test),  ufl.div(sigma2_test)))

        epsilon_u_trial = ufl.sym(ufl.grad(u_trial))
        epsilon_u_test = ufl.sym(ufl.grad(u_test))
        epsilon_w = ufl.sym(ufl.grad(w))


        n = ufl.FacetNormal(mesh)
        dx = ufl.Measure("dx", domain=mesh)

        lambda_, mu = self.get_lame_parameters(p=p)

        def C_inv_action(sigma):
            result = 1/(2*mu) * sigma
            result -=  lambda_/(4*mu*(lambda_+mu)) * ufl.tr(sigma) * ufl.Identity(2)
            return result

        def C_action(epsilon): 
            result = 2*mu*epsilon
            result += lambda_ * ufl.tr(epsilon) * ufl.Identity(2)
            return result

        a00 = ufl.inner(div_sigma_trial_, div_sigma_test_) * dx
        a01 = ufl.inner(r_trial * f, div_sigma_test_) * dx
        a11 = ufl.inner(r_trial * f, r_test * f) * dx


        a00 += ufl.inner(sigma_trial_, C_inv_action(sigma_test_)) * dx
        a00 -= ufl.inner(epsilon_u_trial, sigma_test_) * dx
        a00 -= ufl.inner(sigma_trial_, epsilon_u_test) * dx
        a00 += ufl.inner(epsilon_u_trial, C_action(epsilon_u_test)) * dx

        a01 += ufl.inner(r_trial * z, C_inv_action(sigma_test_)) * dx
        a01 -= ufl.inner(r_trial * epsilon_w, sigma_test_) * dx
        a01 -= ufl.inner(r_trial * z, epsilon_u_test) * dx
        a01 += ufl.inner(r_trial * epsilon_w, C_action(epsilon_u_test)) * dx


        a11 += ufl.inner(r_trial * z, r_test * C_inv_action(z)) * dx
        a11 -= ufl.inner(r_trial * epsilon_w, r_test * z) * dx
        a11 -= ufl.inner(r_trial * z, r_test * epsilon_w) * dx
        a11 += ufl.inner(r_trial * epsilon_w, r_test * C_action(epsilon_w)) * dx

        A00 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a00))
        A01 = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(a01))
        A11 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(a11))

        A00.assemble()
        A01.assemble()

        
        A = {
            'A00': A00,
            'A01': A01,
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

        lambda_, mu = self.get_lame_parameters(p=p)

        def C_inv_action(sigma):
            result = 1/(2*mu) * sigma
            result -=  lambda_/(4*mu*(lambda_+mu)) * ufl.tr(sigma) * ufl.Identity(2)
            return result

        def C_action(epsilon): 
            result = 2*mu*epsilon
            result += lambda_ * ufl.tr(epsilon) * ufl.Identity(2)
            return result

        # epsilon_u = ufl.sym(ufl.grad(u))
        # epsilon_w = ufl.sym(ufl.grad(w))

        epsilon = ufl.sym(ufl.grad(u + w))

        integral = ufl.inner(sigma + z, C_inv_action(sigma + z)) * ufl.dx
        integral -= ufl.inner(epsilon, sigma + z) * ufl.dx
        integral -= ufl.inner(sigma + z, epsilon) * ufl.dx
        integral += ufl.inner(epsilon, C_action(epsilon)) * ufl.dx

        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral))

    def compute_physical_loss_2(self, sigma, u, p):
        f = self.f()
        temp = ufl.div(sigma) + f
        loss = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(temp, temp) * ufl.dx))
        return loss

    def compute_hdiv_h1_norm(self, sigma, u, p):
        lambda_, mu = self.get_lame_parameters(p=p)
        def C_inv_action(sigma):
            result = 1/(2*mu) * sigma
            result -=  lambda_/(4*mu*(lambda_+mu)) * ufl.tr(sigma) * ufl.Identity(2)
            return result

        def C_action(epsilon): 
            result = 2*mu*epsilon
            result += lambda_ * ufl.tr(epsilon) * ufl.Identity(2)
            return result

        epsilon_fc = ufl.sym(ufl.grad(u))

        integral = ufl.inner(epsilon_fc, C_action(epsilon_fc)) * ufl.dx
        integral += ufl.inner(sigma, C_inv_action(sigma)) * ufl.dx
        integral += ufl.inner(ufl.div(sigma), ufl.div(sigma)) * ufl.dx

        return np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)))

    
    def compute_squared_hdiv_h1_norm(self, sigma, u, p):
        lambda_, mu = self.get_lame_parameters(p=p)
        def C_inv_action(sigma):
            result = 1/(2*mu) * sigma
            result -=  lambda_/(4*mu*(lambda_+mu)) * ufl.tr(sigma) * ufl.Identity(2)
            return result

        def C_action(epsilon): 
            result = 2*mu*epsilon
            result += lambda_ * ufl.tr(epsilon) * ufl.Identity(2)
            return result

        epsilon_fc = ufl.sym(ufl.grad(u))

        integral = ufl.inner(epsilon_fc, C_action(epsilon_fc)) * ufl.dx
        integral += ufl.inner(sigma, C_inv_action(sigma)) * ufl.dx
        integral += ufl.inner(ufl.div(sigma), ufl.div(sigma)) * ufl.dx

        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral))


    def compute_L2_norm(self, func): 
        integral = ufl.inner(func, func)*ufl.dx
        return np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)))


    def compute_squared_L2_norm(self, func):
        integral = ufl.inner(func, func)*ufl.dx
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral))


    def project_Hdiv_H1(self, sigma_u_hat: dolfinx.fem.Function, p: dolfinx.fem.Function): 
        Vh = self.Vh
        mesh = self.mesh
        bcs_sigma_u = self.bcs_sigma_u

        (sigma, u) = ufl.TrialFunctions(Vh['sigma_u'])
        (tau, v) = ufl.TestFunctions(Vh['sigma_u'])

        sigma1, sigma2 = ufl.split(sigma)
        sigma_ = ufl.as_vector([sigma1, sigma2])

        tau1, tau2 = ufl.split(tau)
        tau_ = ufl.as_vector([tau1, tau2])

        epsilon_u = ufl.sym(ufl.grad(u))
        epsilon_v = ufl.sym(ufl.grad(v))


        sigma_hat, u_hat = ufl.split(sigma_u_hat)
        sigma1_hat, sigma2_hat = ufl.split(sigma_hat)
        sigma_hat_ = ufl.as_vector([sigma1_hat, sigma2_hat])
        epsilon_u_hat = ufl.sym(ufl.grad(u_hat))


        lambda_, mu = self.get_lame_parameters(p=p)
        def C_inv_action(sigma):
            result = 1/(2*mu) * sigma
            result -=  lambda_/(4*mu*(lambda_+mu)) * ufl.tr(sigma) * ufl.Identity(2)
            return result

        def C_action(epsilon): 
            result = 2*mu*epsilon
            result += lambda_ * ufl.tr(epsilon) * ufl.Identity(2)
            return result

        dx = ufl.Measure("dx", domain=mesh)

        bilinear_form = ufl.inner(ufl.div(sigma_), ufl.div(tau_)) * dx
        bilinear_form += ufl.inner(C_inv_action(sigma_), tau_) * dx
        bilinear_form += ufl.inner(C_action(epsilon_u), epsilon_v) * dx

        linear_form = ufl.inner(ufl.div(sigma_hat_), ufl.div(tau_)) * dx
        linear_form += ufl.inner(C_inv_action(sigma_hat_), tau_) * dx
        linear_form += ufl.inner(C_action(epsilon_u_hat), epsilon_v) * dx

        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=bcs_sigma_u, petsc_options=petsc_options)
        sigma_u = problem.solve()

        return sigma_u


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the linear elasticity problem (with spatially variant Youngs modulus) in the least squares form.')
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

    elasticity_least_squares = ElasticityLeastSquares(mesh_args=mesh_args, function_space_args=function_space_args)
    
    sigma_u_dim = dolfinx.fem.Function(elasticity_least_squares.Vh['sigma_u']).x.array.shape[0]
    sigma_dim = dolfinx.fem.Function(elasticity_least_squares.Vh['sigma']).x.array.shape[0]
    u_dim = dolfinx.fem.Function(elasticity_least_squares.Vh['u']).x.array.shape[0]

    dtype = 'float64'

    local_p_dof, split_num_functions = load_and_scatter(comm, dataset_path+'/p_dof.npy',start_index=50, end_index=100, dtype=dtype)
    local_sigma_u_dof = np.zeros((split_num_functions[rank], sigma_u_dim), dtype=dtype)
    local_sigma_dof = np.zeros((split_num_functions[rank], sigma_dim), dtype=dtype)
    local_u_dof = np.zeros((split_num_functions[rank], u_dim), dtype=dtype)

    if rank == 0:
        start_time = MPI.Wtime()
    for i in tqdm(range(split_num_functions[rank])):
        p = dolfinx.fem.Function(elasticity_least_squares.Vh['p'], dtype=dtype)
        p.x.array[:] = local_p_dof[i,:]
        sigma_u = elasticity_least_squares.solve_sigma_u(p=p)

        sigma = sigma_u.sub(0).collapse()
        u = sigma_u.sub(1).collapse()

        local_sigma_u_dof[i] = sigma_u.x.array
        local_sigma_dof[i] = sigma.x.array
        local_u_dof[i] = u.x.array
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Elapsed time (rank 0 | {split_num_functions[rank]} solves): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    gather_and_save(comm, dataset_path+'/sigma_u_dof_finer.npy', local_sigma_u_dof, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/sigma_dof_finer.npy', local_sigma_dof, split_num_functions, dtype=dtype)
    gather_and_save(comm, dataset_path+'/u_dof_finer.npy', local_u_dof, split_num_functions, dtype=dtype)

    if rank == 0:
        print(f'Saved to {dataset_path}/sigma_u_dof_finer.npy, {dataset_path}/sigma_dof_finer.npy, {dataset_path}/u_dof_finer.npy')