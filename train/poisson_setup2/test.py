# %%
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
import basix.ufl

repo_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(repo_path)

from data_generation.differential_equations import PoissonSetup2LeastSquares
from utils import load_yaml, load_pkl, load_npy, format_elapsed_time, timing
from utils import plot_real_valued_function, plot_block_sparsity, plot_block_submatrix_sparsity

from scifem import create_real_functionspace

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from petsc4py import PETSc
import scifem


from utils import project, norm_L2, convert_petsc_mat_to_torch_sparse_coo_tensor, convert_weight_to_tensor
from utils import evaluate_expression, print_model_size
from typing import Optional
import pickle
import csv

from typing import Optional
from train.train_loss import SurrogateLoss
from models import ConvolutionalNN_129x129
from train.train_utils import BatchIndicesIterator, pretty_print_loss
from train.soap import SOAP
from utils.set_seed import set_seed
import time
from tqdm import tqdm

# %%
set_seed(2025)

# %%
mesh_config_path= repo_path + "/configs/poisson_setup2/config_data/config_mesh.yaml"
function_space_config_path= repo_path + "/configs/poisson_setup2/config_data/config_function_space.yaml"
output_reduced_basis_config_path= repo_path + "/configs/poisson_setup2/config_data/config_output_reduced_basis.yaml"
train_dataset_path = repo_path + "/results/poisson_setup2/train_dataset"
test_dataset_path = repo_path + "/results/poisson_setup2/test_dataset"
model_train_outputs_path = repo_path + "/results/poisson_setup2/model_train_outputs/rbno_physics_loss/test"
model_test_outputs_path = repo_path + "/results/poisson_setup2/model_test_outputs/rbno_physics_loss/test"

mesh_args = load_yaml(mesh_config_path)
function_space_args = load_yaml(function_space_config_path)
output_reduced_basis_args = load_yaml(output_reduced_basis_config_path)
poisson_least_squares = PoissonSetup2LeastSquares(mesh_args, function_space_args)

# %%
p_dof = load_npy(train_dataset_path+'/p_dof.npy')
p_vertex_values = load_npy(train_dataset_path+'/p_vertex_values.npy')
sigma_u_dof = load_npy(train_dataset_path+'/sigma_u_dof.npy')

# %%
print("p_dof shape: ", p_dof.shape)
print("p_vertex_values shape: ", p_vertex_values.shape)
print("sigma_u_dof shape: ", sigma_u_dof.shape)

# %%
mesh = poisson_least_squares.mesh
Vh = poisson_least_squares.Vh

# %%
torch_dtype = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}

# %%
pod_basis_dof = np.load(train_dataset_path+'/hdiv_h1_pod_basis_dof.npy')[:,:output_reduced_basis_args['num_basis']]
reference_reduced_minimizers = np.load(train_dataset_path+'/reference_reduced_minimizers.npy')[:,:output_reduced_basis_args['num_basis']]


# %%
pod_basis_dof = torch.tensor(pod_basis_dof, dtype=torch_dtype['float64'])
reference_reduced_minimizers = torch.tensor(reference_reduced_minimizers, dtype=torch_dtype['float64'])

quadratic_weight = torch.tensor(np.load(train_dataset_path+'/hdiv_h1_quadratic_weight.npy'), dtype=torch_dtype['float64'])
linear_weight = torch.tensor(np.load(train_dataset_path+'/hdiv_h1_linear_weight.npy'), dtype=torch_dtype['float64'])
bias = torch.tensor(np.load(train_dataset_path+'/hdiv_h1_bias.npy'), dtype=torch_dtype['float64'])

# %%
reduced_weight_list = []
for i in range(len(quadratic_weight)):
    reduced_weight = {}
    reduced_weight['quadratic'] = quadratic_weight[i]
    reduced_weight['linear'] = linear_weight[i]
    reduced_weight['bias'] = bias[i]
    reduced_weight_list.append(reduced_weight)

# %%
surrogate_loss = SurrogateLoss(reduced_weight_list)

# %%
dolfinx_mesh_coords = mesh.geometry.x[:,:2]

num_x, num_y = mesh_args['num_x'], mesh_args['num_y']
x = np.linspace(0, mesh_args['upper_right_x'], num_x+1)
y = np.linspace(0, mesh_args['upper_right_y'], num_y+1)
image_mesh_coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
perm = [np.where((image_mesh_coords == row).all(axis=1))[0][0] for row in dolfinx_mesh_coords]

CG_element = basix.ufl.element('CG', 'triangle', 1)
Vh['CG1'] = dolfinx.fem.functionspace(mesh, CG_element)

num_p_fc = len(p_dof)
image_p_vertex_values = np.zeros((num_p_fc, 1, mesh_args['num_y']+1, mesh_args['num_x']+1))
for i in tqdm(range(num_p_fc)):
    image_p_vertex_values[i,0,:,:] = p_vertex_values[i].reshape(num_x+1, num_y+1).T
# %%
device = torch.device('cuda:1')
p_vertex_values_tensor = torch.tensor(image_p_vertex_values, dtype=torch_dtype['float32']).to(device)

# %%
surrogate_loss = SurrogateLoss(reduced_weight_list).astype(torch_dtype['float32']).to(device)
num_basis = output_reduced_basis_args['num_basis']


num_train_list = [16, 64, 256, 1024, 4096]
for num_train in num_train_list:
    print(f'\nStarting training with num_train = {num_train}\n')
    model = ConvolutionalNN_129x129(output_dim=num_basis, activation='leakyrelu', init_func='xavier_uniform')
    model = model.to(device)

    # %%
    _, _ = print_model_size(model)

    # %%
    iterations = 3000
    num_valid = 500

    batch_size = {
        'train': 1000,
        'valid': num_valid
    }

    # %%
    loss_history = {
        'train': [],
        'valid': []
    }
    valid_start_index = 4500


    train_batch_indices_iterator = BatchIndicesIterator(start=0, end=num_train, batch_size=batch_size['train'], shuffle=True)
    valid_batch_indices_iterator = BatchIndicesIterator(start=valid_start_index, end=valid_start_index + num_valid, batch_size=batch_size['valid'], shuffle=False)

    print(f'num_training_samples: {num_train}')
    optimizer = SOAP(params=model.parameters(), lr=5e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=5)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_train_outputs_path, f"best_model_params_{num_train}.pth")
    best_model_iterations = []  

    start_time = time.time()
    counter = 0
    while counter < iterations:
        for batch_indices in train_batch_indices_iterator:
            model.train()
            loss = {'physical': torch.tensor(0.0).to(device),
                    'data': torch.tensor(0.0).to(device),
                    'total': torch.tensor(0.0).to(device)}

            input_ = p_vertex_values_tensor[batch_indices]
            output = model(input_)

            for j, indice in enumerate(batch_indices):
                loss['physical'] += (surrogate_loss(output[j], indice))[0]
                # loss['data'] += data_loss(output[j], reference_reduced_minimizers[indice])

            loss['physical'] /= len(batch_indices)
            loss['data'] /= len(batch_indices)
            loss['total'] = loss['physical'] + loss['data']

            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()
            counter += 1

            print(f'Iteration: {counter}')
            print(f'Train loss: {loss["total"].item()}\n')

            loss_history['train'].append(loss['total'].item())

            with torch.no_grad():
                model.eval()
                valid_loss = {'physical': torch.tensor(0.0).to(device),
                            'data': torch.tensor(0.0).to(device),
                            'total': torch.tensor(0.0).to(device)}
                for valid_batch_indices in valid_batch_indices_iterator:
                    valid_input = p_vertex_values_tensor[valid_batch_indices]
                    valid_output = model(valid_input)
                    for j, indice in enumerate(valid_batch_indices):
                        valid_loss['physical'] += (surrogate_loss(valid_output[j], indice))[0]
                        # valid_loss['data'] += data_loss(valid_output[j], pod_reduced_outputs[indice])
                    valid_loss['physical'] /= len(valid_batch_indices)
                    valid_loss['data'] /= len(valid_batch_indices)
                    valid_loss['total'] = valid_loss['physical'] + valid_loss['data']

                val_loss_val = valid_loss['total'].item()
                print(f'Validation loss: {val_loss_val}\n')

                loss_history['valid'].append(val_loss_val)

                # Save best model
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    torch.save(model.state_dict(), best_model_path)
                    best_model_iterations.append(counter) 
                    print(f"New best model saved at iteration {counter} with validation loss {best_val_loss:.6f}")

        lr_scheduler.step()


    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    # Save training history and final model
    loss_history = {key: np.array(value) for key, value in loss_history.items()}
    np.save(os.path.join(model_train_outputs_path, f'loss_history_{num_train}.npy'), loss_history)
    torch.save(model.state_dict(), os.path.join(model_train_outputs_path, f"latest_model_params_{num_train}.pth"))

    # %%
    plt.figure(figsize=(8, 6))  # Optional: make figure larger for readability

    # Plot lines
    counter_indices = np.arange(1, len(loss_history['train'])+1)
    plt.loglog(counter_indices, loss_history['train'], label='Training Loss', color='blue')
    plt.loglog(counter_indices, loss_history['valid'], label='Validation Loss', color='orange')

    # Markers
    plt.plot(counter_indices[-1], loss_history['train'][-1], 'bo', markersize=5, label='Final training loss')
    plt.plot(best_model_iterations[-1], loss_history['valid'][best_model_iterations[-1]-1], 'go', markersize=5, label='Lowest validation loss')
    plt.plot(best_model_iterations, [loss_history['valid'][i-1] for i in best_model_iterations], 'gx', markersize=2, label='Best model saved')

    # Annotations (increase fontsize)
    plt.annotate(f'{loss_history["train"][-1]:.2e}',
                xy=(counter_indices[-1], loss_history['train'][-1]),
                xytext=(-50, 18),
                textcoords='offset points',
                color='red',
                fontsize=16, 
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.annotate(f'{loss_history["valid"][best_model_iterations[-1]-1]:.2e}',
                xy=(best_model_iterations[-1], loss_history['valid'][best_model_iterations[-1]-1]),
                xytext=(-20, 18),
                textcoords='offset points',
                color='red',
                fontsize=16,  
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Axis labels and title
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(f'Loss history (num_train: {num_train} | num_valid: {num_valid})', fontsize=20)

    # Tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Legend font size
    plt.legend(fontsize=14)

    # Save and show
    plt.savefig(os.path.join(model_train_outputs_path, f'loss_history_{num_train}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # %%
    model = ConvolutionalNN_129x129(output_dim=num_basis, activation='leakyrelu', init_func='xavier_uniform')
    model.load_state_dict(torch.load(os.path.join(model_train_outputs_path, f"best_model_params_{num_train}.pth")))
    model = model.to(device)

    # %%
    num_test = 500

    test_p_dof = load_npy(test_dataset_path+'/p_dof.npy')[:num_test]
    test_sigma_u_dof = load_npy(test_dataset_path+'/sigma_u_dof.npy')[:num_test]
    test_p_vertex_values = load_npy(test_dataset_path+'/p_vertex_values.npy')[:num_test]

    # %%
    test_image_p_vertex_values = np.zeros((num_test, 1, mesh_args['num_y']+1, mesh_args['num_x']+1))
    for i in tqdm(range(num_test)):
        test_image_p_vertex_values[i,0,:,:] = test_p_vertex_values[i].reshape(num_x+1, num_y+1).T

    # %%
    test_p_vertex_values_tensor = torch.tensor(test_image_p_vertex_values, dtype=torch_dtype['float32']).to(device)

    # %%
    test_pod_coeff_pred  = model(test_p_vertex_values_tensor)
    test_pod_coeff_pred = test_pod_coeff_pred.cpu().detach()
    test_pred = test_pod_coeff_pred.to(torch.float64) @ pod_basis_dof.T
    test_pred = test_pred.numpy()

    # %%
    np.save(os.path.join(model_test_outputs_path, f"test_pred_sigma_u_dof_{num_train}.npy"), test_pred)

    # %%
    compute_squared_L2_norm = poisson_least_squares.compute_squared_L2_norm
    compute_squared_hdiv_h1_norm = poisson_least_squares.compute_squared_hdiv_h1_norm

    # %%
    sigma_u_norm_dict = {
        'squared_L2': np.zeros(num_test),
        'avg_squared_L2': 0.0,
        'squared_hdiv_h1': np.zeros(num_test),
        'avg_squared_hdiv_h1': 0.0
    }

    # %%
    for i in tqdm(range(num_test)):
        sigma_u_label_fc = dolfinx.fem.Function(Vh['sigma_u'])
        sigma_u_label_fc.x.array[:] = test_sigma_u_dof[i] 
        sigma_u_norm_dict['squared_L2'][i] = compute_squared_L2_norm(sigma_u_label_fc)
    sigma_u_norm_dict['avg_squared_L2'] = np.mean(sigma_u_norm_dict['squared_L2'])

    # %%
    for i in tqdm(range(num_test)):
        sigma_u_label_fc = dolfinx.fem.Function(Vh['sigma_u'])
        sigma_u_label_fc.x.array[:] = test_sigma_u_dof[i]
        sigma_label_fc = sigma_u_label_fc.sub(0).collapse()
        u_label_fc = sigma_u_label_fc.sub(1).collapse()
        sigma_u_norm_dict['squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(sigma_label_fc, u_label_fc)
    sigma_u_norm_dict['avg_squared_hdiv_h1'] = np.mean(sigma_u_norm_dict['squared_hdiv_h1'])

    # %%
    print(f"Average squared L2 norm of sigma_u: {sigma_u_norm_dict['avg_squared_L2']:.2e}")
    print(f"Average squared H(div)xH1 norm of sigma_u: {sigma_u_norm_dict['avg_squared_hdiv_h1']:.2e}")

    # %%
    sigma_u_error_dict = {
        'squared_L2': np.zeros(num_test),
        'relative_squared_L2': np.zeros(num_test),
        'squared_hdiv_h1': np.zeros(num_test),
        'relative_squared_hdiv_h1': np.zeros(num_test)
    }

    # %%
    for i in tqdm(range(num_test)):
        sigma_u_label_fc = dolfinx.fem.Function(Vh['sigma_u'])
        sigma_u_label_fc.x.array[:] = test_sigma_u_dof[i]
        sigma_label_fc = sigma_u_label_fc.sub(0).collapse()
        u_label_fc = sigma_u_label_fc.sub(1).collapse()

        sigma_u_pred_fc = dolfinx.fem.Function(Vh['sigma_u'])
        sigma_u_pred_fc.x.array[:] = test_pred[i]
        sigma_pred_fc = sigma_u_pred_fc.sub(0).collapse()
        u_pred_fc = sigma_u_pred_fc.sub(1).collapse()

        sigma_u_error_dict['squared_L2'][i] = compute_squared_L2_norm(sigma_u_label_fc - sigma_u_pred_fc)
        sigma_u_error_dict['relative_squared_L2'][i] = sigma_u_error_dict['squared_L2'][i] / sigma_u_norm_dict['avg_squared_L2']

        sigma_u_error_dict['squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(sigma_label_fc - sigma_pred_fc, u_label_fc - u_pred_fc)
        sigma_u_error_dict['relative_squared_hdiv_h1'][i] = sigma_u_error_dict['squared_hdiv_h1'][i] / sigma_u_norm_dict['avg_squared_hdiv_h1']

    # %%
    sigma_u_error_dict['bochner_L2'] = np.sqrt(np.mean(sigma_u_error_dict['squared_L2']))
    sigma_u_error_dict['relative_bochner_L2'] = np.sqrt(np.mean(sigma_u_error_dict['relative_squared_L2']))
    sigma_u_error_dict['bochner_hdiv_h1'] = np.sqrt(np.mean(sigma_u_error_dict['squared_hdiv_h1']))
    sigma_u_error_dict['relative_bochner_hdiv_h1'] = np.sqrt(np.mean(sigma_u_error_dict['relative_squared_hdiv_h1']))

    sigma_u_error_dict['std_L2'] = np.std(np.sqrt(sigma_u_error_dict['squared_L2']))
    sigma_u_error_dict['std_relative_L2'] = np.std(np.sqrt(sigma_u_error_dict['relative_squared_L2']))
    sigma_u_error_dict['std_hdiv_h1'] = np.std(np.sqrt(sigma_u_error_dict['squared_hdiv_h1']))
    sigma_u_error_dict['std_relative_hdiv_h1'] = np.std(np.sqrt(sigma_u_error_dict['relative_squared_hdiv_h1']))

    # %%
    print(f'sigma_u relative Bochner L2 error (std): {sigma_u_error_dict["relative_bochner_L2"]:.2e} ({sigma_u_error_dict["std_relative_L2"]:.2e})')
    print(f'sigma_u relative Bochner H(div) x H1 error (std): {sigma_u_error_dict["relative_bochner_hdiv_h1"]:.2e} ({sigma_u_error_dict["std_relative_hdiv_h1"]:.2e})')

    # %%
    np.save(os.path.join(model_test_outputs_path, f"sigma_u_error_dict_{num_train}.npy"), sigma_u_error_dict)

    # %%
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker

    # %%
    def plot_ref_pred_diff(x, y, ref_f_grid_evals, pred_f_grid_evals, diff_f_grid_evals, variable_name, 
                            levels=100, 
                            ref_pred_format='%.3f',
                            ref_pred_colorbar_pad=0.02, 
                            diff_colorbar_pad=0.01,
                            tick_labelsize=15,  # NEW: control tick font size
                            colorbar_labelsize=15):  # NEW: control colorbar tick font size

        vmin = min(ref_f_grid_evals.min(), pred_f_grid_evals.min())
        vmax = max(ref_f_grid_evals.max(), pred_f_grid_evals.max())

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # Reference
        cf0 = axs[0].tricontourf(x, y, ref_f_grid_evals, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax)
        axs[0].set_title(fr'Reference {variable_name}', fontsize=18)
        axs[0].tick_params(axis='both', which='major', labelsize=tick_labelsize)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])


        # Prediction
        cf1 = axs[1].tricontourf(x, y, pred_f_grid_evals, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax)
        axs[1].set_title(fr'Prediction {variable_name}', fontsize=18)
        axs[1].tick_params(axis='both', which='major', labelsize=tick_labelsize)
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])

        # Shared colorbar for first two subplots
        cbar_shared = fig.colorbar(cf1, ax=[axs[0], axs[1]], format=ref_pred_format, pad=ref_pred_colorbar_pad)
        cbar_shared.ax.tick_params(labelsize=colorbar_labelsize)  # Increase colorbar ticks
        cbar_shared.locator = ticker.MaxNLocator(nbins=5)
        cbar_shared.update_ticks()

        # Difference plot (independent colorbar)
        cf2 = axs[2].tricontourf(x, y, diff_f_grid_evals, levels=levels, cmap='turbo')
        axs[2].set_title(fr'Difference {variable_name}', fontsize=18)
        axs[2].tick_params(axis='both', which='major', labelsize=tick_labelsize)
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])

        cbar_diff = fig.colorbar(cf2, ax=axs[2], pad=diff_colorbar_pad)
        cbar_diff.ax.tick_params(labelsize=colorbar_labelsize)  # Increase colorbar ticks
        cbar_diff.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar_diff.formatter.set_scientific(True)
        cbar_diff.formatter.set_powerlimits((-2, 2))
        cbar_diff.locator = ticker.MaxNLocator(nbins=5)
        cbar_diff.update_ticks()

        cbar_diff.ax.yaxis.get_offset_text().set_fontsize(colorbar_labelsize)

        # Remove ticks on x and y axes if needed
        for ax in axs:
            ax.tick_params(left=False, bottom=False)

        return fig

    # %%
    for i in range(3):
        # plt.figure(figsize=(6, 5))  # Optional: larger figure size
        img = plt.imshow(test_p_vertex_values_tensor[i,0,:,:].cpu().numpy(),
                        extent=[0, 1, 0, 1], origin='lower', cmap='turbo')

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)  # Increase colorbar tick labels

        plt.title('parameter (setup 1)', fontsize=20)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)

        plt.savefig(os.path.join(model_test_outputs_path, f'test_p_{i}_{num_train}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # %%
    for test_sample_index in range(3):
        pred_sigma_u_fc = dolfinx.fem.Function(Vh['sigma_u'])
        pred_sigma_u_fc.x.array[:] = test_pred[test_sample_index]
        pred_sigma_fc = pred_sigma_u_fc.sub(0).collapse()
        pred_u_fc = pred_sigma_u_fc.sub(1).collapse()

        ref_sigma_u_fc = dolfinx.fem.Function(Vh['sigma_u'])
        ref_sigma_u_fc.x.array[:] = test_sigma_u_dof[test_sample_index]
        ref_sigma_fc = ref_sigma_u_fc.sub(0).collapse()
        ref_u_fc = ref_sigma_u_fc.sub(1).collapse()


        pred_sigma_grid_evals = evaluate_expression(mesh, pred_sigma_fc, mesh.geometry.x)[1]
        pred_u_grid_evals = evaluate_expression(mesh, pred_u_fc, mesh.geometry.x)[1]


        ref_sigma_grid_evals = evaluate_expression(mesh, ref_sigma_fc, mesh.geometry.x)[1]
        ref_u_grid_evals = evaluate_expression(mesh, ref_u_fc, mesh.geometry.x)[1]


        diff_sigma_grid_evals = pred_sigma_grid_evals - ref_sigma_grid_evals
        diff_u_grid_evals = pred_u_grid_evals - ref_u_grid_evals

        x = mesh.geometry.x[:, 0]
        y = mesh.geometry.x[:, 1]


        fig = plot_ref_pred_diff(x, y, ref_u_grid_evals[:,0], pred_u_grid_evals[:,0], diff_u_grid_evals[:,0], r'$u^{\circ}$',
                                levels=100,
                                ref_pred_format='%.3f',
                                ref_pred_colorbar_pad=0.01,
                                diff_colorbar_pad=-0.01)
        fig.savefig(os.path.join(model_test_outputs_path, f'u_ref_pred_diff_{test_sample_index}_{num_train}.png'), dpi=300, bbox_inches='tight')


        fig = plot_ref_pred_diff(x, y, ref_sigma_grid_evals[:,0], pred_sigma_grid_evals[:, 0], diff_sigma_grid_evals[:, 0], r'$\sigma_1^{\circ}$', 
                                levels=100,
                                ref_pred_format='%.1f',
                                ref_pred_colorbar_pad=0.01, 
                                diff_colorbar_pad=-0.01)
        fig.savefig(os.path.join(model_test_outputs_path, f'sigma_1_ref_pred_diff_{test_sample_index}_{num_train}.png'), dpi=300, bbox_inches='tight')


        fig = plot_ref_pred_diff(x, y, ref_sigma_grid_evals[:,1], pred_sigma_grid_evals[:, 1], diff_sigma_grid_evals[:, 1], r'$\sigma_2^{\circ}$', 
                                levels=100,
                                ref_pred_format='%.1f',
                                ref_pred_colorbar_pad=0.01, 
                                diff_colorbar_pad=-0.01)
        fig.savefig(os.path.join(model_test_outputs_path, f'sigma_2_ref_pred_diff_{test_sample_index}_{num_train}.png'), dpi=300, bbox_inches='tight')

    # %%
    compute_physical_loss_1 = poisson_least_squares.compute_physical_loss_1
    compute_physical_loss_2 = poisson_least_squares.compute_physical_loss_2

    # %%
    residual_loss_dict = {
        'loss_1': np.zeros(num_test),
        'loss_2': np.zeros(num_test),
        'total_loss': np.zeros(num_test),
        'sqrt_total_loss': np.zeros(num_test)
    }
    for test_index in range(num_test): 
        pred_fc = dolfinx.fem.Function(Vh['sigma_u'])
        pred_fc.x.array[:] = test_pred[test_index]
        pred_sigma_fc = pred_fc.sub(0).collapse()
        pred_u_fc = pred_fc.sub(1).collapse()

        p_fc = dolfinx.fem.Function(Vh['p'])
        p_fc.x.array[:] = test_p_dof[test_index]

        residual_loss_1 = compute_physical_loss_1(pred_sigma_fc, pred_u_fc, p_fc)
        residual_loss_2 = compute_physical_loss_2(pred_sigma_fc, pred_u_fc, p_fc)

        residual_loss = residual_loss_1 + residual_loss_2

        sqrt_residual_loss = np.sqrt(residual_loss)

        residual_loss_dict['loss_1'][test_index] = residual_loss_1
        residual_loss_dict['loss_2'][test_index] = residual_loss_2
        residual_loss_dict['total_loss'][test_index] = residual_loss
        residual_loss_dict['sqrt_total_loss'][test_index] = sqrt_residual_loss
        print(f'Test sample {test_index}:')
        print(f'Residual loss 1: {residual_loss_1} | Residual loss 2: {residual_loss_2}')
        print(f'Total residual loss: {residual_loss} | Sqrt residual loss: {sqrt_residual_loss}')

        print("")

    # %%

    # %%
    print(f'mean residual loss 1: {np.mean(residual_loss_dict["loss_1"]):.2e} (std: {np.std(residual_loss_dict["loss_1"]):.2e})')
    print(f'mean residual loss 2: {np.mean(residual_loss_dict["loss_2"]):.2e} (std: {np.std(residual_loss_dict["loss_2"]):.2e})')
    print(f'mean total residual loss: {np.mean(residual_loss_dict["total_loss"]):.2e} (std: {np.std(residual_loss_dict["total_loss"]):.2e})')
    print(f'mean sqrt total residual loss: {np.mean(residual_loss_dict["sqrt_total_loss"]):.2e} (std: {np.std(residual_loss_dict["sqrt_total_loss"]):.2e})')

    residual_loss_dict['mean_residual_loss_1'] = np.mean(residual_loss_dict['loss_1'])
    residual_loss_dict['std_residual_loss_1'] = np.std(residual_loss_dict['loss_1'])
    residual_loss_dict['mean_residual_loss_2'] = np.mean(residual_loss_dict['loss_2'])
    residual_loss_dict['std_residual_loss_2'] = np.std(residual_loss_dict['loss_2'])
    residual_loss_dict['mean_total_residual_loss'] = np.mean(residual_loss_dict['total_loss'])
    residual_loss_dict['std_total_residual_loss'] = np.std(residual_loss_dict['total_loss'])
    residual_loss_dict['mean_sqrt_total_residual_loss'] = np.mean(residual_loss_dict['sqrt_total_loss'])
    residual_loss_dict['std_sqrt_total_residual_loss'] = np.std(residual_loss_dict['sqrt_total_loss'])


    np.save(os.path.join(model_test_outputs_path, f"residual_loss_dict_{num_train}.npy"), residual_loss_dict)


    # %%
    ratio_list = []
    for test_index in range(num_test):
        ratio = np.sqrt(sigma_u_error_dict['squared_hdiv_h1'])[test_index] / residual_loss_dict['sqrt_total_loss'][test_index]
        ratio_list.append(ratio)
        print(f'index: {test_index} | ratio: {ratio}')

    # %%
    plt.figure(figsize=(10, 8))  # Optional: bigger figure
    plt.hist(ratio_list, bins=10)

    plt.xlabel('Value', fontsize=25) 
    plt.ylabel('Frequency', fontsize=25)
    plt.title(fr'Error / $\sqrt{{\text{{Loss}}}}$ (# test = {num_test})', fontsize=25)

    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    plt.savefig(os.path.join(model_test_outputs_path, f'ratio_histogram_hdiv_h1_{num_train}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()



    test_sigma_u_dof_finer = load_npy(test_dataset_path+'/sigma_u_dof_finer.npy')[:num_test]
    function_space_finer_config_path= repo_path + "/configs/poisson_setup2/config_data/config_function_space_finer.yaml"
    function_space_finer_args = load_yaml(function_space_finer_config_path)

    u_element = basix.ufl.element(family=function_space_finer_args["u"]["family"],
                                    cell=mesh_args["mesh_cell_type"],
                                    degree=function_space_finer_args["u"]["degree"])
    sigma_element = basix.ufl.element(family=function_space_finer_args["sigma"]["family"],
                                        cell=mesh_args["mesh_cell_type"],
                                        degree=function_space_finer_args["sigma"]["degree"])
    sigma_u_element = basix.ufl.mixed_element([sigma_element, u_element])

    finer_Vh = {
        'sigma_u': dolfinx.fem.functionspace(mesh, sigma_u_element)
    }


    finer_sigma_u_norm_dict = {
        'squared_L2': np.zeros(num_test),
        'avg_squared_L2': 0.0,
        'squared_hdiv_h1': np.zeros(num_test),
        'avg_squared_hdiv_h1': 0.0
    }


    for i in tqdm(range(num_test)):
        sigma_u_label_fc = dolfinx.fem.Function(finer_Vh['sigma_u'])
        sigma_u_label_fc.x.array[:] = test_sigma_u_dof_finer[i] 
        finer_sigma_u_norm_dict['squared_L2'][i] = compute_squared_L2_norm(sigma_u_label_fc)
    finer_sigma_u_norm_dict['avg_squared_L2'] = np.mean(finer_sigma_u_norm_dict['squared_L2'])


    for i in tqdm(range(num_test)):
        sigma_u_label_fc = dolfinx.fem.Function(finer_Vh['sigma_u'])
        sigma_u_label_fc.x.array[:] = test_sigma_u_dof_finer[i]
        sigma_label_fc = sigma_u_label_fc.sub(0).collapse()
        u_label_fc = sigma_u_label_fc.sub(1).collapse()
        finer_sigma_u_norm_dict['squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(sigma_label_fc, u_label_fc)
    finer_sigma_u_norm_dict['avg_squared_hdiv_h1'] = np.mean(finer_sigma_u_norm_dict['squared_hdiv_h1'])


    print(f"Average squared L2 norm of sigma_u: {finer_sigma_u_norm_dict['avg_squared_L2']:.2e}")
    print(f"Average squared H(div)xH1 norm of sigma_u: {finer_sigma_u_norm_dict['avg_squared_hdiv_h1']:.2e}")



    finer_sigma_u_error_dict = {
        'squared_L2': np.zeros(num_test),
        'relative_squared_L2': np.zeros(num_test),
        'squared_hdiv_h1': np.zeros(num_test),
        'relative_squared_hdiv_h1': np.zeros(num_test)
    }


    for i in tqdm(range(num_test)):
        sigma_u_label_fc = dolfinx.fem.Function(finer_Vh['sigma_u'])
        sigma_u_label_fc.x.array[:] = test_sigma_u_dof_finer[i]
        sigma_label_fc = sigma_u_label_fc.sub(0).collapse()
        u_label_fc = sigma_u_label_fc.sub(1).collapse()
    
        sigma_u_pred_fc = dolfinx.fem.Function(Vh['sigma_u'])
        sigma_u_pred_fc.x.array[:] = test_pred[i] 
        sigma_pred_fc = sigma_u_pred_fc.sub(0).collapse()
        u_pred_fc = sigma_u_pred_fc.sub(1).collapse()

        finer_sigma_u_error_dict['squared_L2'][i] = compute_squared_L2_norm(sigma_u_label_fc - sigma_u_pred_fc)
        finer_sigma_u_error_dict['relative_squared_L2'][i] = finer_sigma_u_error_dict['squared_L2'][i] / finer_sigma_u_norm_dict['avg_squared_L2']

        finer_sigma_u_error_dict['squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(sigma_label_fc - sigma_pred_fc, u_label_fc - u_pred_fc)
        finer_sigma_u_error_dict['relative_squared_hdiv_h1'][i] = finer_sigma_u_error_dict['squared_hdiv_h1'][i] / finer_sigma_u_norm_dict['avg_squared_hdiv_h1']


    finer_sigma_u_error_dict['bochner_L2'] = np.sqrt(np.mean(finer_sigma_u_error_dict['squared_L2']))
    finer_sigma_u_error_dict['relative_bochner_L2'] = np.sqrt(np.mean(finer_sigma_u_error_dict['relative_squared_L2']))
    finer_sigma_u_error_dict['bochner_hdiv_h1'] = np.sqrt(np.mean(finer_sigma_u_error_dict['squared_hdiv_h1']))
    finer_sigma_u_error_dict['relative_bochner_hdiv_h1'] = np.sqrt(np.mean(finer_sigma_u_error_dict['relative_squared_hdiv_h1']))

    finer_sigma_u_error_dict['std_L2'] = np.std(np.sqrt(finer_sigma_u_error_dict['squared_L2']))
    finer_sigma_u_error_dict['std_relative_L2'] = np.std(np.sqrt(finer_sigma_u_error_dict['relative_squared_L2']))
    finer_sigma_u_error_dict['std_hdiv_h1'] = np.std(np.sqrt(finer_sigma_u_error_dict['squared_hdiv_h1']))
    finer_sigma_u_error_dict['std_relative_hdiv_h1'] = np.std(np.sqrt(finer_sigma_u_error_dict['relative_squared_hdiv_h1']))

    print(f'sigma_u relative Bochner L2 error (std): {finer_sigma_u_error_dict["relative_bochner_L2"]:.2e} ({finer_sigma_u_error_dict["std_relative_L2"]:.2e})')
    print(f'sigma_u relative Bochner H(div) x H1 error (std): {finer_sigma_u_error_dict["relative_bochner_hdiv_h1"]:.2e} ({finer_sigma_u_error_dict["std_relative_hdiv_h1"]:.2e})')

    np.save(os.path.join(model_test_outputs_path, f"finer_sigma_u_error_dict_{num_train}.npy"), finer_sigma_u_error_dict)


    ratio_list = []
    for test_index in range(num_test):
        ratio = np.sqrt(finer_sigma_u_error_dict['squared_hdiv_h1'])[test_index] / residual_loss_dict['sqrt_total_loss'][test_index]
        ratio_list.append(ratio)
        print(f'index: {test_index} | ratio: {ratio}')


    plt.figure(figsize=(10, 8))  # Optional: bigger figure
    plt.hist(ratio_list, bins=10)

    plt.xlabel(r'Error / $\sqrt{\text{Loss}}$', fontsize=25) 
    plt.ylabel('Frequency', fontsize=25)
    plt.title('Poisson setup 2', fontsize=30)

    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    plt.savefig(os.path.join(model_test_outputs_path, f'ratio_histogram_hdiv_h1_finer_{num_train}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()