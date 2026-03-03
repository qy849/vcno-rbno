import dolfinx
import ufl
from petsc4py import PETSc

def project(v, target_func, bcs=[]):

    """ 
    Adapted from 
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-
    function-dolfinx/4142/6
    """
    
    V = target_func.function_space
    target_func_dim = target_func.x.array.shape[0]


    Pv = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    a = ufl.inner(Pv, w) * ufl.dx
    L = ufl.inner(v, w) * ufl.dx

    # a = dolfinx.fem.form(a)
    # A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
    # A.assemble()

    # L = dolfinx.fem.form(L)
    # b = dolfinx.fem.petsc.assemble_vector(L)

    # dolfinx.fem.apply_lifting(b, [a], [bcs])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # dolfinx.fem.set_bc(b, bcs)

    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
    temp_fc = problem.solve()

    # solver = PETSc.KSP().create(A.getComm())
    # solver.setOperators(A)

    # vec = PETSc.Vec().create()
    # vec.setSizes(target_func_dim)
    # vec.setUp()
    # solver.solve(b, vec)

    target_func.x.array[:] = temp_fc.x.array


def project_Hdiv(v, target_func, bcs=[]):
    
    V = target_func.function_space
    target_func_dim = target_func.x.array.shape[0]


    Pv = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    a = ufl.inner(Pv, w) * ufl.dx + ufl.inner(ufl.div(Pv), ufl.div(w)) * ufl.dx
    L = ufl.inner(v, w) * ufl.dx + ufl.inner(ufl.div(v), ufl.div(w)) * ufl.dx

    # a = dolfinx.fem.form(a)
    # A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
    # A.assemble()

    # L = dolfinx.fem.form(L)
    # b = dolfinx.fem.petsc.assemble_vector(L)

    # dolfinx.fem.apply_lifting(b, [a], [bcs])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # dolfinx.fem.set_bc(b, bcs)
    
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
    temp_fc = problem.solve()


    # solver = PETSc.KSP().create(A.getComm())
    # solver.setOperators(A)

    # vec = PETSc.Vec().create()
    # vec.setSizes(target_func_dim)
    # vec.setUp()
    # solver.solve(b, vec)

    target_func.x.array[:] = temp_fc.x.array



def project_H1(v, target_func, bcs=[]):

    
    V = target_func.function_space
    target_func_dim = target_func.x.array.shape[0]


    Pv = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    a = ufl.inner(Pv, w) * ufl.dx + ufl.inner(ufl.grad(Pv), ufl.grad(w)) * ufl.dx
    L = ufl.inner(v, w) * ufl.dx + ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx

    # a = dolfinx.fem.form(a)
    # A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
    # A.assemble()

    # L = dolfinx.fem.form(L)
    # b = dolfinx.fem.petsc.assemble_vector(L)

    # dolfinx.fem.apply_lifting(b, [a], [bcs])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # dolfinx.fem.set_bc(b, bcs)
    
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
    temp_fc = problem.solve()

    # solver = PETSc.KSP().create(A.getComm())
    # solver.setOperators(A)

    # vec = PETSc.Vec().create()
    # vec.setSizes(target_func_dim)
    # vec.setUp()
    # solver.solve(b, vec)

    target_func.x.array[:] = temp_fc.x.array
