import os
import sys
import argparse

import numpy as np
import torch    
import ufl
import dolfinx
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, load_npy, save_npy, convert_petsc_mat_to_torch_sparse_coo_tensor

from data_generation.differential_equations import ElasticityLeastSquares


class Hdiv_H1_Norm(torch.nn.Module):
    def __init__(self, Vh, is_weight_tensor: bool = False):
        super().__init__()
        self.Vh = Vh
        self.mesh = Vh['sigma_u'].mesh
        self.is_weight_tensor = is_weight_tensor


    def convert_sigma(self, sigma_formal):
        RT_fc, bubble_fc = ufl.split(sigma_formal)
        sigma_true = RT_fc + ufl.curl(bubble_fc)
        return sigma_true


    def get_lame_parameters(self, p: dolfinx.fem.Function):
        mesh = self.mesh
        Vh = self.Vh

        E_0 = 1.0 
        nu = 0.4 # Poisson ratio
        E = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(E_0)) + ufl.exp(p) # Young's modulus
        lambda_ = (E*nu)/((1.0+nu)*(1.0-2.0*nu)) #  Lamé’s first parameter
        mu = E/(2.0*(1.0+nu)) #  Lamé’s second parameter

        return lambda_, mu

    def set_weight(self, p: dolfinx.fem.Function):
        Vh = self.Vh
        mesh = self.mesh
    
        (sigma_trial, u_trial) = ufl.TrialFunctions(Vh['sigma_u'])
        (sigma_test, u_test) = ufl.TestFunctions(Vh['sigma_u'])
        sigma1_trial, sigma2_trial = ufl.split(sigma_trial)
        sigma1_test, sigma2_test = ufl.split(sigma_test)
 
        sigma_trial_ = ufl.as_vector((sigma1_trial, sigma2_trial))
        sigma_test_ = ufl.as_vector((sigma1_test, sigma2_test))


        div_sigma_trial_ = ufl.as_vector((ufl.div(sigma1_trial),  ufl.div(sigma2_trial)))
        div_sigma_test_ = ufl.as_vector((ufl.div(sigma1_test),  ufl.div(sigma2_test)))

        epsilon_trial = ufl.sym(ufl.grad(u_trial))
        epsilon_test = ufl.sym(ufl.grad(u_test))

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

        
        weight_1 = ufl.inner(epsilon_trial, C_action(epsilon_test)) * dx
        weight_1 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(weight_1))
        weight_1.assemble()
        self.weight_1 = weight_1

        weight_2 = ufl.inner(sigma_trial_, C_inv_action(sigma_test_)) * dx
        weight_2 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(weight_2))
        weight_2.assemble()
        self.weight_2 = weight_2


        weight_3 = ufl.inner(div_sigma_trial_, div_sigma_test_) * dx
        weight_3 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(weight_3))
        weight_3.assemble()
        self.weight_3 = weight_3


        if self.is_weight_tensor: 
            self.weight_1 = convert_petsc_mat_to_torch_sparse_coo_tensor(self.weight_1, dtype=torch.float64)
            self.weight_2 = convert_petsc_mat_to_torch_sparse_coo_tensor(self.weight_2, dtype=torch.float64)
            self.weight_3 = convert_petsc_mat_to_torch_sparse_coo_tensor(self.weight_3, dtype=torch.float64)
        
        self.weight = self.weight_1 + self.weight_2 + self.weight_3

    def forward(self, y: torch.Tensor, verbose: bool = False):
        """
        y: dof
        """
        term_1 = torch.dot(y, self.weight_1 @ y)
        term_2 = torch.dot(y, self.weight_2 @ y)
        term_3 = torch.dot(y, self.weight_3 @ y)
        if verbose:
            print(f'term_1: {term_1} | term_2: {term_2} | term_3: {term_3}')
            print(f'sqrt_term_1: {torch.sqrt(term_1)} | sqrt_term_2: {torch.sqrt(term_2)} | sqrt_term_3: {torch.sqrt(term_3)}')

        square_norm_value = term_1 + term_2 + term_3
        norm_value = torch.sqrt(square_norm_value)
        return norm_value
    

    def to(self, device):
        if not self.is_weight_tensor and device.type == 'cuda':
            raise ValueError("The weight matrices should be set as tensors.")
        self.weight_1 = self.weight_1.to(device)
        self.weight_2 = self.weight_2.to(device)
        self.weight_3 = self.weight_3.to(device)
        self.weight = self.weight.to(device)
        return self


def compute_pod_basis(snapshot: torch.Tensor, num_basis: int, weight: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute POD basis using NumPy.

    Parameters:
        snapshot: (num_samples, num_dof) complex or real ndarray
        num_basis: number of POD modes to extract
        weight: (num_dof, num_dof) ndarray, optional

    Returns:
        sqrt_eigvals: (num_basis,) ndarray of sqrt of top eigenvalues
        pod_basis: (num_dof, num_basis) ndarray of POD modes
    """

    num_samples, num_dof = snapshot.shape
    assert num_basis < num_dof

    if weight is None:
        C = snapshot @ snapshot.T
    else:
        C = snapshot @ weight @ snapshot.T    
        
    eigvals, eigvecs = torch.linalg.eigh(C) # ascending order
    eigvals = eigvals[-num_basis:]
    eigvecs = eigvecs[:,-num_basis:]    

    eigvals = eigvals.flip([0])  # now descending order
    eigvecs = eigvecs.flip([1]) 

    temp = snapshot.T @ eigvecs
    pod_basis = torch.zeros(num_dof, num_basis, dtype=snapshot.dtype)
    for i in range(num_basis):
        pod_basis[:, i] = temp[:, i] / torch.sqrt(eigvals[i])

    return torch.sqrt(eigvals), pod_basis



def plot_eigenvalues(eigenvalues: np.ndarray, title: str):
    indices = np.arange(1, len(eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.log10(indices), np.log10(eigenvalues), color='blue', marker='o', markersize=3)
    ax.set_xlabel(r'$\log_{10}(i)$', fontsize=25)
    ax.set_ylabel(r'$\log_{10}(\lambda_i)$', fontsize=25, rotation=0, labelpad=30)
    ax.set_title(title, fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    plt.close()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the POD basis of elasticity solutions.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--output_reduced_basis_config_path', type=str, help='Path to the output reduced basis configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    output_reduced_basis_args = load_yaml(args.output_reduced_basis_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    elasticity_least_squares = ElasticityLeastSquares(mesh_args, function_space_args)

    sigma_u_dof = load_npy(train_dataset_path+'/sigma_u_dof.npy')

    dtype = sigma_u_dof.dtype
    if dtype == np.float32:
        torch_dtype = torch.float32
    elif dtype == np.float64:
        torch_dtype = torch.float64
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')

    snapshot = sigma_u_dof[:output_reduced_basis_args['num_evals']]

    # Compute POD basis
    print('Start computing POD basis in the Hdiv-H1 norm...')
    snapshot = torch.from_numpy(snapshot).to(torch_dtype)

    hdiv_h1_norm = Hdiv_H1_Norm(elasticity_least_squares.Vh, is_weight_tensor=True)

    p_dof = load_npy(train_dataset_path+'/p_dof.npy')
    mean_p_dof = np.mean(p_dof, axis=0)
    save_npy(train_dataset_path + '/mean_p_dof.npy', mean_p_dof)
    save_npy(test_dataset_path + '/mean_p_dof.npy', mean_p_dof)
    mean_p_fc = dolfinx.fem.Function(elasticity_least_squares.Vh['p'])
    mean_p_fc.x.array[:] = mean_p_dof
    hdiv_h1_norm.set_weight(mean_p_fc)
    weight = hdiv_h1_norm.weight



    eigvals, pod_basis_dof = compute_pod_basis(snapshot, num_basis=output_reduced_basis_args['num_basis'], weight=weight)
    print('Finished computing POD basis.')
    print(f'POD basis shape: {pod_basis_dof.shape}')

    # Save results
    save_npy(train_dataset_path + '/hdiv_h1_pod_basis_dof.npy', pod_basis_dof.numpy())
    save_npy(train_dataset_path + '/hdiv_h1_pod_eigvals.npy', eigvals.numpy())
    print(f'POD basis and eigenvalues saved to {train_dataset_path}/hdiv_h1_pod_basis_dof.npy and {train_dataset_path}/hdiv_h1_pod_eigvals.npy')


    save_npy(test_dataset_path + '/hdiv_h1_pod_basis_dof.npy', pod_basis_dof.numpy())
    save_npy(test_dataset_path + '/hdiv_h1_pod_eigvals.npy', eigvals.numpy())
    print(f'POD basis and eigenvalues saved to {test_dataset_path}/hdiv_h1_pod_basis_dof.npy and {test_dataset_path}/hdiv_h1_pod_eigvals.npy')

    # Plot eigenvalues
    fig = plot_eigenvalues(eigvals, 'Eigenvalues (POD basis)')
    fig.savefig(train_dataset_path + '/hdiv_h1_pod_eigenvalues.png', dpi=300, bbox_inches='tight')
    print(f'Eigenvalue plot saved to {train_dataset_path}/hdiv_h1_pod_eigenvalues.png')

    # Compute projection coefficients
    pod_coeff_labels = torch.from_numpy(sigma_u_dof).to(torch_dtype) @ weight @ pod_basis_dof
    save_npy(train_dataset_path + '/hdiv_h1_pod_coeff_labels.npy', pod_coeff_labels.numpy())
    print(f'train POD coefficients saved to {train_dataset_path}/hdiv_h1_pod_coeff_labels.npy')


    test_sigma_u_dof = load_npy(test_dataset_path+'/sigma_u_dof.npy')
    test_pod_coeff_labels = torch.from_numpy(test_sigma_u_dof).to(torch_dtype) @ hdiv_h1_norm.weight @ pod_basis_dof
    save_npy(test_dataset_path + '/hdiv_h1_pod_coeff_labels.npy', test_pod_coeff_labels.numpy())
    print(f'test POD coefficients saved to {test_dataset_path}/hdiv_h1_pod_coeff_labels.npy')
