# %% [markdown]
# # FC_PINO two-stage data + physics baseline for `elasticity`
#
# Stage 1 trains the neural operator with supervised data loss on `N_1` parameter-solution pairs. Stage 2 continues from the best stage-1 checkpoint and trains with a combined objective: FC physics residual/boundary loss on `N_2` parameter instances plus supervised data loss on batches sampled from the first `N_1` labeled instances, where `N_2 > N_1`.
#
# The solution has six grid channels `(sigma_11, sigma_12, sigma_21, sigma_22, u_1, u_2)`. The Lame parameters depend on the parameter field `p` through `E = 1 + exp(p)`, so the constitutive residual is evaluated pointwise with `p`-dependent `C` and `C^{-1}` actions.

# %%
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import dolfinx
import scifem
import ufl

# Resolve the repo root (the directory containing FC_PINO/ and configs/) from this file's
# location so it is independent of the working directory and no absolute path is hard-coded.
try:
    repo_path = Path(__file__).resolve().parent.parent
except NameError:
    repo_path = Path.cwd().resolve()
if not ((repo_path / 'FC_PINO').is_dir() and (repo_path / 'configs').is_dir()):
    cwd = Path.cwd().resolve()
    repo_path = cwd.parent if cwd.name == 'FC_PINO' else cwd
fc_pino_path = repo_path / 'FC_PINO'
# Insert repo_path ahead of FC_PINO on sys.path so `from utils import ...` resolves to the
# repo's utils package rather than FC_PINO/utils.py.
# Do not add the repo's parent here: it may contain an older local neuraloperator tree that
# shadows the installed neuralop package.
for path in [str(repo_path), str(fc_pino_path)]:
    while path in sys.path:
        sys.path.remove(path)
for path in [str(fc_pino_path), str(repo_path)]:
    sys.path.insert(0, path)

from data_generation.differential_equations import ElasticityLeastSquares
from utils import evaluate_expression, load_yaml
from fc_fno import FC_FNO
from neuralop.layers.fourier_continuation import FCLegendre, FCGram

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'repo_path: {repo_path}')
print(f'device: {device}')

# %% [markdown]
# ## Configuration
#
# Two-stage FC_PINO training:
#
# 1. Train the neural operator with supervised data loss on the first `num_data_train` (`N_1`) parameter-solution instances.
# 2. Continue from that model with a combined stage-2 loss: physics loss on the first `num_physics_train` (`N_2`) parameter instances plus supervised data loss on batches from the first `N_1` labeled instances. Here `N_2 > N_1` and the physics set includes the supervised set.

# %%
mesh_config_path = repo_path / 'configs/elasticity/config_data/config_mesh.yaml'
function_space_config_path = repo_path / 'configs/elasticity/config_data/config_function_space.yaml'
train_dataset_path = repo_path / 'results/elasticity/train_dataset'
test_dataset_path = repo_path / 'results/elasticity/test_dataset'
model_train_outputs_path = repo_path / 'results/elasticity/model_train_outputs/fc_pino_two_stage_data_physics'
model_test_outputs_path = repo_path / 'results/elasticity/model_test_outputs/fc_pino_two_stage_data_physics'
model_train_outputs_path.mkdir(parents=True, exist_ok=True)
model_test_outputs_path.mkdir(parents=True, exist_ok=True)

# Two-stage training sizes. N_2 must include the first N_1 parameter instances.
num_data_train = 2000      # N_1: parameter-solution pairs used for supervised data training.
num_physics_train = 3000   # N_2: parameter instances used for physics-only training; must be > N_1.
num_train = num_data_train # Kept for downstream summaries/checkpoint compatibility.
num_valid = 500
valid_start_index = 4500
num_test = 500

# Stage iteration counts. Total iterations are reported as `iterations` for downstream summaries.
data_iterations = 6000
physics_iterations = 3000
iterations = data_iterations + physics_iterations

batch_size = 10
test_batch_size = 50
validate_every = 25
num_visualize = 3

learning_rate = 1e-3
weight_decay = 1e-2
physics_weight = 1.0
bc_weight = 1.0
stage2_data_weight = 1.0

fc_backend = 'gram'  # 'gram' or 'legendre'
fc_degree = 2
fc_cont_points = 50
n_modes = (32, 32)
hidden_channels = 128
n_layers = 4

mesh_args = load_yaml(mesh_config_path)
function_space_args = load_yaml(function_space_config_path)
num_x = mesh_args['num_x']
num_y = mesh_args['num_y']
Lx = mesh_args['upper_right_x'] - mesh_args['lower_left_x']
Ly = mesh_args['upper_right_y'] - mesh_args['lower_left_y']
assert num_physics_train > num_data_train, 'Two-stage training requires N_2 > N_1.'
assert num_physics_train <= valid_start_index, 'Move valid_start_index if the physics-training set should use more train samples.'
print(mesh_args)
print(f'N_1 supervised samples: {num_data_train}; N_2 physics samples: {num_physics_train}')


# %% [markdown]
# ## Data loading
#
# Arrays are reshaped in physical `(x_index, y_index)` order so that FC_FNO `dx` aligns with the first spatial tensor dimension and `dy` with the second.

# %%
def load_slice(path: Path, start: int, count: int):
    arr = np.load(path, mmap_mode='r')
    return np.array(arr[start:start + count])


def scalar_vertices_to_xy(values: np.ndarray) -> np.ndarray:
    # The saved scalar (p) vertex arrays are flattened y-outer/x-inner, i.e.
    # meshgrid(x, y) ('xy' indexing) row-major, unlike the tensor (sigma, u)
    # arrays which are x-outer/y-inner. Reshape to (y, x) first, then transpose
    # to the (x, y) image order used by the labels and AddSpatialCoordinates.
    return values.reshape(values.shape[0], num_y + 1, num_x + 1).transpose(0, 2, 1)


def tensor_vertices_to_cxy(values: np.ndarray) -> np.ndarray:
    num_components = values.shape[2]
    return values.reshape(values.shape[0], num_x + 1, num_y + 1, num_components).transpose(0, 3, 1, 2)


def stack_sigma_u(sigma_vertex: np.ndarray, u_vertex: np.ndarray) -> np.ndarray:
    # Output channels: [sigma_11, sigma_12, sigma_21, sigma_22, u_1, u_2].
    return np.concatenate([tensor_vertices_to_cxy(sigma_vertex), tensor_vertices_to_cxy(u_vertex)], axis=1)

# Load N_2 parameters for physics training. The first N_1 of these have labels for supervised training.
physics_p_vertex = load_slice(train_dataset_path / 'p_vertex_values.npy', 0, num_physics_train)
train_p_vertex = physics_p_vertex[:num_data_train]
train_sigma_vertex = load_slice(train_dataset_path / 'sigma_vertex_values.npy', 0, num_data_train)
train_u_vertex = load_slice(train_dataset_path / 'u_vertex_values.npy', 0, num_data_train)

valid_p_vertex = load_slice(train_dataset_path / 'p_vertex_values.npy', valid_start_index, num_valid)
valid_sigma_vertex = load_slice(train_dataset_path / 'sigma_vertex_values.npy', valid_start_index, num_valid)
valid_u_vertex = load_slice(train_dataset_path / 'u_vertex_values.npy', valid_start_index, num_valid)

test_p_vertex = load_slice(test_dataset_path / 'p_vertex_values.npy', 0, num_test)
test_sigma_vertex = load_slice(test_dataset_path / 'sigma_vertex_values.npy', 0, num_test)
test_u_vertex = load_slice(test_dataset_path / 'u_vertex_values.npy', 0, num_test)
test_p_dof = load_slice(test_dataset_path / 'p_dof.npy', 0, num_test)
test_sigma_u_dof = load_slice(test_dataset_path / 'sigma_u_dof.npy', 0, num_test)

train_p = scalar_vertices_to_xy(train_p_vertex)[:, None, :, :]
physics_p = scalar_vertices_to_xy(physics_p_vertex)[:, None, :, :]
valid_p = scalar_vertices_to_xy(valid_p_vertex)[:, None, :, :]
test_p = scalar_vertices_to_xy(test_p_vertex)[:, None, :, :]

train_y = stack_sigma_u(train_sigma_vertex, train_u_vertex)
valid_y = stack_sigma_u(valid_sigma_vertex, valid_u_vertex)
test_y = stack_sigma_u(test_sigma_vertex, test_u_vertex)

train_p_tensor = torch.as_tensor(train_p, dtype=torch.float64)
physics_p_tensor = torch.as_tensor(physics_p, dtype=torch.float64)
valid_p_tensor = torch.as_tensor(valid_p, dtype=torch.float64)
test_p_tensor = torch.as_tensor(test_p, dtype=torch.float64)
train_y_tensor = torch.as_tensor(train_y, dtype=torch.float64)
valid_y_tensor = torch.as_tensor(valid_y, dtype=torch.float64)
test_y_tensor = torch.as_tensor(test_y, dtype=torch.float64)

class AddSpatialCoordinates(nn.Module):
    def __init__(self, num_x: int, num_y: int, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        super().__init__()
        x = torch.linspace(x_min, x_max, num_x, dtype=torch.float64)
        y = torch.linspace(y_min, y_max, num_y, dtype=torch.float64)
        x_coor, y_coor = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('x_coor', x_coor[None, None, :, :])
        self.register_buffer('y_coor', y_coor[None, None, :, :])

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        x = self.x_coor.expand(batch_size, -1, -1, -1).to(device=inputs.device, dtype=inputs.dtype)
        y = self.y_coor.expand(batch_size, -1, -1, -1).to(device=inputs.device, dtype=inputs.dtype)
        return torch.cat((inputs, x, y), dim=1)

add_spatial_coordinates = AddSpatialCoordinates(
    num_x + 1,
    num_y + 1,
    x_min=mesh_args['lower_left_x'],
    x_max=mesh_args['upper_right_x'],
    y_min=mesh_args['lower_left_y'],
    y_max=mesh_args['upper_right_y'],
)
train_input_tensor = add_spatial_coordinates(train_p_tensor)
physics_input_tensor = add_spatial_coordinates(physics_p_tensor)
valid_input_tensor = add_spatial_coordinates(valid_p_tensor)
test_input_tensor = add_spatial_coordinates(test_p_tensor)

print('supervised scalar p:', tuple(train_p_tensor.shape), 'supervised model input:', tuple(train_input_tensor.shape), 'label:', tuple(train_y_tensor.shape))
print('physics scalar p:', tuple(physics_p_tensor.shape), 'physics model input:', tuple(physics_input_tensor.shape))
print('test scalar p:', tuple(test_p_tensor.shape), 'test model input:', tuple(test_input_tensor.shape), 'test label:', tuple(test_y_tensor.shape))


# %% [markdown]
# ## FEM setup and auxiliary fields
#
# The continuous residual uses the same `q` (with `z = grad(q)`) and `w` auxiliary fields as `ElasticityLeastSquares`. Both are independent of `p`.

# %%
elasticity_least_squares = ElasticityLeastSquares(mesh_args, function_space_args)
mesh = elasticity_least_squares.mesh
Vh = elasticity_least_squares.Vh

# Map dolfinx mesh-geometry vertex order to the flat `(x, y)` image-grid order used above.
dolfinx_mesh_coords = mesh.geometry.x[:, :2]
x_grid = np.linspace(mesh_args['lower_left_x'], mesh_args['upper_right_x'], num_x + 1)
y_grid = np.linspace(mesh_args['lower_left_y'], mesh_args['upper_right_y'], num_y + 1)
image_mesh_coords = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)  # Match elasticity_fno.ipynb flat vertex order.
perm = np.array([np.where(np.isclose(image_mesh_coords, row).all(axis=1))[0][0] for row in dolfinx_mesh_coords], dtype=np.int64)
max_perm_coordinate_mismatch = np.max(np.abs(image_mesh_coords[perm] - dolfinx_mesh_coords))
print(f'max mesh/image coordinate mismatch after perm: {max_perm_coordinate_mismatch:.2e}')
assert max_perm_coordinate_mismatch < 1e-12

q_fc = elasticity_least_squares.solve_q()
w_fc = elasticity_least_squares.solve_w()
q_grad_mesh = evaluate_expression(mesh, ufl.grad(q_fc), mesh.geometry.x)[1]
w_grad_mesh = evaluate_expression(mesh, ufl.grad(w_fc), mesh.geometry.x)[1]


def mesh_values_to_xy(values_at_mesh: np.ndarray) -> np.ndarray:
    values_at_mesh = np.asarray(values_at_mesh)
    if values_at_mesh.ndim == 1:
        values_at_mesh = values_at_mesh[:, None]
    flat = np.zeros((len(perm), values_at_mesh.shape[1]), dtype=np.float64)
    flat[perm, :] = values_at_mesh
    return flat.reshape(num_x + 1, num_y + 1, values_at_mesh.shape[1]).transpose(2, 0, 1)

z_grad_xy = mesh_values_to_xy(q_grad_mesh)  # z = grad(q): [z11, z12, z21, z22]
w_grad_xy = mesh_values_to_xy(w_grad_mesh)  # grad(w): [w11, w12, w21, w22]

z_grad_tensor = torch.as_tensor(z_grad_xy[None, :, :, :], dtype=torch.float64, device=device)
w_grad_tensor = torch.as_tensor(w_grad_xy[None, :, :, :], dtype=torch.float64, device=device)
print('z=grad(q):', tuple(z_grad_tensor.shape), 'grad(w):', tuple(w_grad_tensor.shape))

# Lame parameters: E = 1 + exp(p), nu = 0.4 (matching ElasticityLeastSquares.get_lame_parameters).
youngs_modulus_base = 1.0
poisson_ratio = 0.4

# %% [markdown]
# ## Loss functions and model

# %%
class RelativeL2Loss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, outputs, labels):
        diff = outputs - labels
        diff_norm = torch.linalg.vector_norm(diff.flatten(start_dim=1), dim=1)
        label_norm = torch.linalg.vector_norm(labels.flatten(start_dim=1), dim=1)
        return (diff_norm / (label_norm + self.eps)).mean()

relative_l2_loss = RelativeL2Loss()


def split_derivatives(derivs):
    if not isinstance(derivs, (list, tuple)) or len(derivs) != 2:
        raise ValueError('Expected FC_FNO to return [dx, dy] derivatives.')
    return derivs[0], derivs[1]


def lame_from_p(p):
    youngs_modulus = youngs_modulus_base + torch.exp(p)
    lambda_ = (youngs_modulus * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    return lambda_, mu


def averaged_rectangle_rule(values):
    # Averaged rectangle rule over FC-grid samples. The rectangle cell volume is
    # included explicitly even though it cancels with the sampled-domain volume.
    _, _, x_res, y_res = values.shape
    number_of_rectangle_cells = x_res * y_res
    rectangle_cell_volume = (Lx * Ly) / number_of_rectangle_cells
    sampled_domain_volume = rectangle_cell_volume * number_of_rectangle_cells
    rectangle_integral = values.flatten(start_dim=1).sum(dim=1) * rectangle_cell_volume
    return rectangle_integral / sampled_domain_volume


def fc_residual_terms(p, pred, derivs):
    # Pointwise least-squares residual densities matching ElasticityLeastSquares.compute_physical_loss_{1,2}.
    # r1 is the constitutive-energy density <T, C^{-1} T> with T = sigma + z - C(eps(u + w)).
    # r2 is the equilibrium density |div(sigma)|^2 (body force f = 0).
    dx_pred, dy_pred = split_derivatives(derivs)
    lambda_, mu = lame_from_p(p)

    sigma_11 = pred[:, 0:1]
    sigma_12 = pred[:, 1:2]
    sigma_21 = pred[:, 2:3]
    sigma_22 = pred[:, 3:4]

    grad_U_11 = dx_pred[:, 4:5] + w_grad_tensor[:, 0:1]
    grad_U_12 = dy_pred[:, 4:5] + w_grad_tensor[:, 1:2]
    grad_U_21 = dx_pred[:, 5:6] + w_grad_tensor[:, 2:3]
    grad_U_22 = dy_pred[:, 5:6] + w_grad_tensor[:, 3:4]

    eps_11 = grad_U_11
    eps_22 = grad_U_22
    eps_12 = 0.5 * (grad_U_12 + grad_U_21)
    eps_21 = eps_12
    trace_eps = eps_11 + eps_22

    C_eps_11 = 2.0 * mu * eps_11 + lambda_ * trace_eps
    C_eps_22 = 2.0 * mu * eps_22 + lambda_ * trace_eps
    C_eps_12 = 2.0 * mu * eps_12
    C_eps_21 = 2.0 * mu * eps_21

    T_11 = sigma_11 + z_grad_tensor[:, 0:1] - C_eps_11
    T_12 = sigma_12 + z_grad_tensor[:, 1:2] - C_eps_12
    T_21 = sigma_21 + z_grad_tensor[:, 2:3] - C_eps_21
    T_22 = sigma_22 + z_grad_tensor[:, 3:4] - C_eps_22
    trace_T = T_11 + T_22

    r1 = (T_11.square() + T_12.square() + T_21.square() + T_22.square()) / (2.0 * mu)
    r1 = r1 - (lambda_ / (4.0 * mu * (lambda_ + mu))) * trace_T.square()

    div_sigma_1 = dx_pred[:, 0:1] + dy_pred[:, 1:2]
    div_sigma_2 = dx_pred[:, 2:3] + dy_pred[:, 3:4]
    r2 = div_sigma_1.square() + div_sigma_2.square()

    r1_mse = averaged_rectangle_rule(r1)
    r2_mse = averaged_rectangle_rule(r2)
    return r1_mse, r2_mse


def boundary_loss(pred):
    # sigma = 0 on bottom (y min), top (y max), and right (x max); u = 0 on left (x min).
    sigma = pred[:, 0:4]
    u = pred[:, 4:6]
    loss_sigma = sigma[:, :, :, 0].square().mean() + sigma[:, :, :, -1].square().mean() + sigma[:, :, -1, :].square().mean()
    loss_u = u[:, :, 0, :].square().mean()
    return loss_sigma + loss_u


def supervised_loss_components(label, pred):
    data_loss = relative_l2_loss(pred[:, :4], label[:, :4]) + relative_l2_loss(pred[:, 4:6], label[:, 4:6])
    return {
        'total': data_loss,
        'data': data_loss,
    }


def physics_loss_components(p, pred, derivs):
    r1_mse, r2_mse = fc_residual_terms(p, pred, derivs)
    residual_loss = r1_mse.mean() + r2_mse.mean()
    bc = boundary_loss(pred)
    total = physics_weight * residual_loss + bc_weight * bc
    return {
        'total': total,
        'fc_r1_mse': r1_mse.mean(),
        'fc_r2_mse': r2_mse.mean(),
        'fc_total_mse': residual_loss,
        'bc': bc,
    }


def ensure_fcgram_npz_matrices(d_values=(2, 3, 4, 5, 6), c=25):
    src_dir = fc_pino_path / 'FC_Gram_Construction/FCGram_matrices'
    dst_dir = model_train_outputs_path / 'fcgram_matrices'
    dst_dir.mkdir(parents=True, exist_ok=True)
    for d in d_values:
        dst = dst_dir / f'FCGram_data_d{d}_c{c}.npz'
        if dst.exists():
            continue
        src = src_dir / f'FCGram_data_d{d}_C{c}.mat'
        mat = scipy.io.loadmat(src)
        np.savez(dst, ArQr=mat['ArQr'], AlQl=mat['AlQl'])
    return dst_dir


if fc_backend.lower() == 'gram':
    if FCGram is None:
        raise ImportError("FCGram is not available in this environment; set fc_backend = 'legendre' or install FCGram support.")
    fcgram_matrices_path = ensure_fcgram_npz_matrices(d_values=(fc_degree,), c=fc_cont_points // 2)
    extension_func = FCGram(d=fc_degree, n_additional_pts=fc_cont_points, matrices_path=fcgram_matrices_path).to(device)
elif fc_backend.lower() == 'legendre':
    extension_func = FCLegendre(d=fc_degree, n_additional_pts=fc_cont_points).to(device)
else:
    raise ValueError("fc_backend must be 'legendre' or 'gram'.")
model = FC_FNO(
    in_channels=3,
    out_channels=6,
    Lengths=(Lx, Ly),
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    n_layers=n_layers,
    FC_obj=extension_func,
    positional_embedding=None,
    non_linearity=F.gelu,
    projection_nonlinearity=F.tanh,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, iterations // 10), gamma=0.8)
best_data_model_path = model_train_outputs_path / f'best_data_model_params_N1_{num_data_train}_coords.pth'
best_model_path = model_train_outputs_path / f'best_physics_model_params_N1_{num_data_train}_N2_{num_physics_train}_coords.pth'
latest_model_path = model_train_outputs_path / f'latest_model_params_N1_{num_data_train}_N2_{num_physics_train}_coords.pth'
print('params:', sum(p.numel() for p in model.parameters()))


# %% [markdown]
# ## Smoke check
#
# This verifies the requested tensor shapes and derivative-channel alignment before training.

# %%
model.eval()
with torch.no_grad():
    smoke_input = train_input_tensor[:1].to(device)
    smoke_output, smoke_derivs = model(smoke_input, derivs_to_compute=['dx', 'dy'])
print('model input:', tuple(smoke_input.shape))
print('model output:', tuple(smoke_output.shape))
print('dx shape:', tuple(smoke_derivs[0].shape), 'dy shape:', tuple(smoke_derivs[1].shape))
assert tuple(smoke_input.shape[1:]) == (3, num_x + 1, num_y + 1)
assert tuple(smoke_output.shape[1:]) == (6, num_x + 1, num_y + 1)
assert tuple(smoke_derivs[0].shape[1:]) == (6, num_x + 1, num_y + 1)
assert tuple(smoke_derivs[1].shape[1:]) == (6, num_x + 1, num_y + 1)

# %% [markdown]
# ## Two-stage training
#
# Stage 1 uses only the supervised data loss on the first `N_1` parameter-solution pairs. Stage 2 starts from the best stage-1 checkpoint and uses a combined objective: FC residual plus boundary physics loss on `N_2` parameter fields, and supervised data loss on a separately sampled batch from the first `N_1` labeled fields.

# %%
def evaluate_supervised_data_loss(input_tensor, label_tensor, count):
    model.eval()
    totals = []
    with torch.no_grad():
        for start in range(0, count, test_batch_size):
            end = min(start + test_batch_size, count)
            model_input = input_tensor[start:end].to(device)
            label = label_tensor[start:end].to(device)
            pred = model(model_input)
            totals.append(supervised_loss_components(label, pred)['total'].detach().cpu())
    return torch.stack(totals).mean().item()


def evaluate_data_validation():
    return evaluate_supervised_data_loss(valid_input_tensor, valid_y_tensor, num_valid)


def evaluate_training_data_loss():
    return evaluate_supervised_data_loss(train_input_tensor, train_y_tensor, num_data_train)


def evaluate_physics_validation():
    model.eval()
    totals = []
    with torch.no_grad():
        for start in range(0, num_valid, test_batch_size):
            end = min(start + test_batch_size, num_valid)
            model_input = valid_input_tensor[start:end].to(device)
            p = valid_p_tensor[start:end].to(device)
            pred, derivs = model(model_input, derivs_to_compute=['dx', 'dy'])
            totals.append(physics_loss_components(p, pred, derivs)['total'].detach().cpu())
    return torch.stack(totals).mean().item()


loss_history = {
    'data_train': [],
    'data_valid': [],
    'physics_train': [],
    'physics_valid': [],
    'best_data_iterations': [],
    'best_physics_iterations': [],
    'num_data_train': num_data_train,
    'num_physics_train': num_physics_train,
    'data_iterations': data_iterations,
    'physics_iterations': physics_iterations,
    'stage2_data_weight': stage2_data_weight,
}
best_data_val_loss = float('inf')
best_physics_val_loss = float('inf')
start_time = time.time()

for step in range(1, data_iterations + 1):
    model.train()
    indices = torch.randint(0, num_data_train, (batch_size,))
    model_input = train_input_tensor[indices].to(device)
    label = train_y_tensor[indices].to(device)

    optimizer.zero_grad(set_to_none=True)
    pred = model(model_input)
    losses = supervised_loss_components(label, pred)
    losses['total'].backward()
    optimizer.step()
    scheduler.step()

    record = {name: value.detach().cpu().item() for name, value in losses.items()}
    record['stage'] = 'data'
    record['stage_iteration'] = step
    record['global_iteration'] = step
    loss_history['data_train'].append(record)

    if step == 1 or step % validate_every == 0 or step == data_iterations:
        val_loss = evaluate_data_validation()
        loss_history['data_valid'].append({'stage': 'data', 'stage_iteration': step, 'global_iteration': step, 'total': val_loss})
        if val_loss < best_data_val_loss:
            best_data_val_loss = val_loss
            loss_history['best_data_iterations'].append(step)
            torch.save(model.state_dict(), best_data_model_path)
        print(f"data iter {step:05d} | train data {record['data']:.3e} | valid data {val_loss:.3e}")

if best_data_model_path.exists():
    model.load_state_dict(torch.load(best_data_model_path, map_location=device, weights_only=False))
    print(f'loaded best data checkpoint before physics stage: {best_data_model_path}')

for step in range(1, physics_iterations + 1):
    model.train()
    physics_indices = torch.randint(0, num_physics_train, (batch_size,))
    data_indices = torch.randint(0, num_data_train, (batch_size,))

    physics_model_input = physics_input_tensor[physics_indices].to(device)
    physics_p = physics_p_tensor[physics_indices].to(device)
    data_model_input = train_input_tensor[data_indices].to(device)
    data_label = train_y_tensor[data_indices].to(device)

    optimizer.zero_grad(set_to_none=True)
    physics_pred, physics_derivs = model(physics_model_input, derivs_to_compute=['dx', 'dy'])
    physics_losses = physics_loss_components(physics_p, physics_pred, physics_derivs)
    data_pred = model(data_model_input)
    data_losses = supervised_loss_components(data_label, data_pred)
    total_loss = physics_losses['total'] + stage2_data_weight * data_losses['data']
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    global_step = data_iterations + step
    record = {name: value.detach().cpu().item() for name, value in physics_losses.items()}
    record['stage'] = 'physics_data'
    record['stage_iteration'] = step
    record['global_iteration'] = global_step
    record['physics_total'] = record['total']
    record['data'] = data_losses['data'].detach().cpu().item()
    record['stage2_data_weight'] = stage2_data_weight
    record['total'] = total_loss.detach().cpu().item()
    loss_history['physics_train'].append(record)

    if step == 1 or step % validate_every == 0 or step == physics_iterations:
        val_physics_loss = evaluate_physics_validation()
        val_data_loss = evaluate_data_validation()
        monitor_data_loss = evaluate_training_data_loss()
        val_loss = val_physics_loss + stage2_data_weight * val_data_loss
        loss_history['physics_valid'].append({
            'stage': 'physics_data',
            'stage_iteration': step,
            'global_iteration': global_step,
            'total': val_loss,
            'physics_total': val_physics_loss,
            'data': val_data_loss,
            'data_loss_on_N1': monitor_data_loss,
            'stage2_data_weight': stage2_data_weight,
        })
        if val_loss < best_physics_val_loss:
            best_physics_val_loss = val_loss
            loss_history['best_physics_iterations'].append(step)
            torch.save(model.state_dict(), best_model_path)
        print(
            f"physics+data iter {step:05d} | train total {record['total']:.3e} | "
            f"physics {record['physics_total']:.3e} | data {record['data']:.3e} | "
            f"fc {record['fc_total_mse']:.3e} | bc {record['bc']:.3e} | "
            f"valid total {val_loss:.3e} | valid physics {val_physics_loss:.3e} | "
            f"valid data {val_data_loss:.3e} | monitor data(N1) {monitor_data_loss:.3e}"
        )

if physics_iterations == 0 and best_data_model_path.exists():
    torch.save(torch.load(best_data_model_path, map_location='cpu', weights_only=False), best_model_path)

torch.save(model.state_dict(), latest_model_path)
np.save(model_train_outputs_path / f'loss_history_N1_{num_data_train}_N2_{num_physics_train}_stage2_data.npy', loss_history)
print(f'training elapsed: {time.time() - start_time:.1f}s')
print(f'best data validation loss: {best_data_val_loss:.3e}')
print(f'best physics validation loss: {best_physics_val_loss:.3e}')


# %% [markdown]
# ## Test inference and FC-grid residual loss

# %%
if best_model_path.exists():
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
model.eval()

test_pred_chunks = []
fc_residual_loss_dict = {
    'loss_1': np.zeros(num_test),
    'loss_2': np.zeros(num_test),
    'total_loss': np.zeros(num_test),
    'sqrt_total_loss': np.zeros(num_test),
}

with torch.no_grad():
    for start in tqdm(range(0, num_test, test_batch_size), desc='FC_PINO test inference'):
        end = min(start + test_batch_size, num_test)
        model_input = test_input_tensor[start:end].to(device)
        p = test_p_tensor[start:end].to(device)
        pred, derivs = model(model_input, derivs_to_compute=['dx', 'dy'])
        r1_mse, r2_mse = fc_residual_terms(p, pred, derivs)
        total = r1_mse + r2_mse
        fc_residual_loss_dict['loss_1'][start:end] = r1_mse.cpu().numpy()
        fc_residual_loss_dict['loss_2'][start:end] = r2_mse.cpu().numpy()
        fc_residual_loss_dict['total_loss'][start:end] = total.cpu().numpy()
        fc_residual_loss_dict['sqrt_total_loss'][start:end] = torch.sqrt(total).cpu().numpy()
        test_pred_chunks.append(pred.cpu())

test_sigma_u_vertex_values_pred = torch.cat(test_pred_chunks, dim=0).numpy().astype(np.float64)
np.save(model_test_outputs_path / 'test_pred_vertex_values_xy.npy', test_sigma_u_vertex_values_pred)
print(f"mean FC residual loss 1: {np.mean(fc_residual_loss_dict['loss_1']):.2e} (std: {np.std(fc_residual_loss_dict['loss_1']):.2e})")
print(f"mean FC residual loss 2: {np.mean(fc_residual_loss_dict['loss_2']):.2e} (std: {np.std(fc_residual_loss_dict['loss_2']):.2e})")
print(f"mean FC total residual loss: {np.mean(fc_residual_loss_dict['total_loss']):.2e} (std: {np.std(fc_residual_loss_dict['total_loss']):.2e})")

# %% [markdown]
# ## Convert point predictions to finite-element DoFs
#
# This mirrors the CG1 reconstruction and H(div)xH1 projection used in the existing FNO notebook. The projection uses the dataset mean parameter `mean_p_fc`.

# %%
mean_p_fc = dolfinx.fem.Function(Vh['p'])
mean_p_fc.x.array[:] = np.load(train_dataset_path / 'mean_p_dof.npy')

Vh_CG1_sigma11, Vh_CG1_sigma11_map = Vh['CG1_tensor'].sub(0).sub(0).collapse()
Vh_CG1_sigma12, Vh_CG1_sigma12_map = Vh['CG1_tensor'].sub(0).sub(1).collapse()
Vh_CG1_sigma21, Vh_CG1_sigma21_map = Vh['CG1_tensor'].sub(1).sub(0).collapse()
Vh_CG1_sigma22, Vh_CG1_sigma22_map = Vh['CG1_tensor'].sub(1).sub(1).collapse()
Vh_CG1_u1, Vh_CG1_u1_map = Vh['CG1_vector'].sub(0).collapse()
Vh_CG1_u2, Vh_CG1_u2_map = Vh['CG1_vector'].sub(1).collapse()
Vh_CG1_sigma, Vh_CG1_sigma_map = Vh['CG1_tensor_vector'].sub(0).collapse()
Vh_CG1_u, Vh_CG1_u_map = Vh['CG1_tensor_vector'].sub(1).collapse()

cg1_dof_to_vertex = scifem.dof_to_vertexmap(Vh['CG1'])

test_CG1_sigma_fc_pred_list = []
test_CG1_u_fc_pred_list = []
test_CG1_sigma_fc_label_list = []
test_CG1_u_fc_label_list = []
test_pred_sigma_u_dof = np.zeros((num_test, test_sigma_u_dof.shape[1]), dtype=np.float64)


def image_channel_to_cg1(channel_xy):
    fc = dolfinx.fem.Function(Vh['CG1'])
    fc.x.array[:] = channel_xy.reshape(-1, order='C')[cg1_dof_to_vertex][perm]
    return fc

for i in tqdm(range(num_test), desc='Build CG1 predictions and project predictions'):
    pred_xy = test_sigma_u_vertex_values_pred[i]

    sigma11_fc = image_channel_to_cg1(pred_xy[0])
    sigma12_fc = image_channel_to_cg1(pred_xy[1])
    sigma21_fc = image_channel_to_cg1(pred_xy[2])
    sigma22_fc = image_channel_to_cg1(pred_xy[3])
    u1_fc = image_channel_to_cg1(pred_xy[4])
    u2_fc = image_channel_to_cg1(pred_xy[5])

    sigma_fc = dolfinx.fem.Function(Vh['CG1_tensor'])
    sigma_fc.x.array[Vh_CG1_sigma11_map] = sigma11_fc.x.array[:]
    sigma_fc.x.array[Vh_CG1_sigma12_map] = sigma12_fc.x.array[:]
    sigma_fc.x.array[Vh_CG1_sigma21_map] = sigma21_fc.x.array[:]
    sigma_fc.x.array[Vh_CG1_sigma22_map] = sigma22_fc.x.array[:]

    u_fc = dolfinx.fem.Function(Vh['CG1_vector'])
    u_fc.x.array[Vh_CG1_u1_map] = u1_fc.x.array[:]
    u_fc.x.array[Vh_CG1_u2_map] = u2_fc.x.array[:]

    sigma_u_cg1 = dolfinx.fem.Function(Vh['CG1_tensor_vector'])
    sigma_u_cg1.x.array[np.sort(Vh_CG1_sigma_map)] = sigma_fc.x.array[:]
    sigma_u_cg1.x.array[np.sort(Vh_CG1_u_map)] = u_fc.x.array[:]

    sigma_u_projected = elasticity_least_squares.project_Hdiv_H1(sigma_u_cg1, mean_p_fc)
    test_pred_sigma_u_dof[i] = sigma_u_projected.x.array[:]
    test_CG1_sigma_fc_pred_list.append(sigma_fc)
    test_CG1_u_fc_pred_list.append(u_fc)

# CG1 reference functions built from the FEM ground-truth DoFs (matches elasticity_fno.ipynb).
for i in tqdm(range(num_test), desc='Build CG1 labels from FEM DoFs'):
    sigma_u_fc = dolfinx.fem.Function(Vh['sigma_u'])
    sigma_u_fc.x.array[:] = test_sigma_u_dof[i]
    sigma_label_fc = sigma_u_fc.sub(0).collapse()
    u_label_fc = sigma_u_fc.sub(1).collapse()
    u_evals = evaluate_expression(mesh, u_label_fc, mesh.geometry.x)[1]
    sigma_evals = evaluate_expression(mesh, sigma_label_fc, mesh.geometry.x)[1]

    cg1_u1 = dolfinx.fem.Function(Vh['CG1'])
    cg1_u1.x.array[:][scifem.vertex_to_dofmap(Vh['CG1'])] = u_evals[:, 0]
    cg1_u2 = dolfinx.fem.Function(Vh['CG1'])
    cg1_u2.x.array[:][scifem.vertex_to_dofmap(Vh['CG1'])] = u_evals[:, 1]
    cg1_u = dolfinx.fem.Function(Vh['CG1_vector'])
    cg1_u.x.array[:] = np.stack([cg1_u1.x.array, cg1_u2.x.array], axis=1).reshape(-1)

    cg1_sigma11 = dolfinx.fem.Function(Vh['CG1'])
    cg1_sigma11.x.array[:][scifem.vertex_to_dofmap(Vh['CG1'])] = sigma_evals[:, 0]
    cg1_sigma12 = dolfinx.fem.Function(Vh['CG1'])
    cg1_sigma12.x.array[:][scifem.vertex_to_dofmap(Vh['CG1'])] = sigma_evals[:, 1]
    cg1_sigma1 = dolfinx.fem.Function(Vh['CG1_vector'])
    cg1_sigma1.x.array[:] = np.stack([cg1_sigma11.x.array, cg1_sigma12.x.array], axis=1).reshape(-1)

    cg1_sigma21 = dolfinx.fem.Function(Vh['CG1'])
    cg1_sigma21.x.array[:][scifem.vertex_to_dofmap(Vh['CG1'])] = sigma_evals[:, 2]
    cg1_sigma22 = dolfinx.fem.Function(Vh['CG1'])
    cg1_sigma22.x.array[:][scifem.vertex_to_dofmap(Vh['CG1'])] = sigma_evals[:, 3]
    cg1_sigma2 = dolfinx.fem.Function(Vh['CG1_vector'])
    cg1_sigma2.x.array[:] = np.stack([cg1_sigma21.x.array, cg1_sigma22.x.array], axis=1).reshape(-1)

    cg1_sigma = dolfinx.fem.Function(Vh['CG1_tensor'])
    _, cg1_sigma1_map = Vh['CG1_tensor'].sub(0).collapse()
    _, cg1_sigma2_map = Vh['CG1_tensor'].sub(1).collapse()
    cg1_sigma.x.array[np.sort(cg1_sigma1_map)] = cg1_sigma1.x.array
    cg1_sigma.x.array[np.sort(cg1_sigma2_map)] = cg1_sigma2.x.array

    test_CG1_sigma_fc_label_list.append(cg1_sigma)
    test_CG1_u_fc_label_list.append(cg1_u)

np.save(model_test_outputs_path / 'test_pred_sigma_u_dof.npy', test_pred_sigma_u_dof)
print('projected prediction dofs:', test_pred_sigma_u_dof.shape)

# %% [markdown]
# ## Relative L2 and H(div) x H1 errors
#
# The H(div)xH1 norm uses the dataset mean parameter `mean_p_fc`, matching `elasticity_fno.ipynb`.

# %%
compute_squared_L2_norm = elasticity_least_squares.compute_squared_L2_norm
compute_squared_hdiv_h1_norm = elasticity_least_squares.compute_squared_hdiv_h1_norm

sigma_u_norm_dict = {
    'squared_L2': np.zeros(num_test),
    'avg_squared_L2': 0.0,
    'squared_hdiv_h1': np.zeros(num_test),
    'avg_squared_hdiv_h1': 0.0,
    'cg1_squared_L2': np.zeros(num_test),
    'avg_cg1_squared_L2': 0.0,
    'cg1_squared_hdiv_h1': np.zeros(num_test),
    'avg_cg1_squared_hdiv_h1': 0.0,
}
sigma_u_error_dict = {
    'squared_L2': np.zeros(num_test),
    'relative_squared_L2': np.zeros(num_test),
    'squared_hdiv_h1': np.zeros(num_test),
    'relative_squared_hdiv_h1': np.zeros(num_test),
    'cg1_squared_L2': np.zeros(num_test),
    'relative_cg1_squared_L2': np.zeros(num_test),
    'cg1_squared_hdiv_h1': np.zeros(num_test),
    'relative_cg1_squared_hdiv_h1': np.zeros(num_test),
}

for i in tqdm(range(num_test), desc='Projected reference norms'):
    sigma_u_label_fc = dolfinx.fem.Function(Vh['sigma_u'])
    sigma_u_label_fc.x.array[:] = test_sigma_u_dof[i]
    sigma_label_fc = sigma_u_label_fc.sub(0).collapse()
    u_label_fc = sigma_u_label_fc.sub(1).collapse()
    sigma1_label_fc = sigma_label_fc.sub(0).collapse()
    sigma2_label_fc = sigma_label_fc.sub(1).collapse()
    sigma_label_fc_ = ufl.as_vector((sigma1_label_fc, sigma2_label_fc))
    sigma_u_norm_dict['squared_L2'][i] = compute_squared_L2_norm(sigma_u_label_fc)
    sigma_u_norm_dict['squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(sigma_label_fc_, u_label_fc, mean_p_fc)

sigma_u_norm_dict['avg_squared_L2'] = np.mean(sigma_u_norm_dict['squared_L2'])
sigma_u_norm_dict['avg_squared_hdiv_h1'] = np.mean(sigma_u_norm_dict['squared_hdiv_h1'])

for i in tqdm(range(num_test), desc='CG1 reference norms'):
    cg1_sigma = test_CG1_sigma_fc_label_list[i]
    cg1_u = test_CG1_u_fc_label_list[i]
    cg1_sigma1 = cg1_sigma.sub(0).collapse()
    cg1_sigma2 = cg1_sigma.sub(1).collapse()
    cg1_sigma_ = ufl.as_tensor([cg1_sigma1, cg1_sigma2])
    sigma_u_norm_dict['cg1_squared_L2'][i] = compute_squared_L2_norm(cg1_sigma) + compute_squared_L2_norm(cg1_u)
    sigma_u_norm_dict['cg1_squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(cg1_sigma_, cg1_u, mean_p_fc)

sigma_u_norm_dict['avg_cg1_squared_L2'] = np.mean(sigma_u_norm_dict['cg1_squared_L2'])
sigma_u_norm_dict['avg_cg1_squared_hdiv_h1'] = np.mean(sigma_u_norm_dict['cg1_squared_hdiv_h1'])

for i in tqdm(range(num_test), desc='Projected prediction errors'):
    sigma_u_label_fc = dolfinx.fem.Function(Vh['sigma_u'])
    sigma_u_label_fc.x.array[:] = test_sigma_u_dof[i]
    sigma_label_fc = sigma_u_label_fc.sub(0).collapse()
    u_label_fc = sigma_u_label_fc.sub(1).collapse()
    sigma1_label_fc = sigma_label_fc.sub(0).collapse()
    sigma2_label_fc = sigma_label_fc.sub(1).collapse()

    sigma_u_pred_fc = dolfinx.fem.Function(Vh['sigma_u'])
    sigma_u_pred_fc.x.array[:] = test_pred_sigma_u_dof[i]
    sigma_pred_fc = sigma_u_pred_fc.sub(0).collapse()
    u_pred_fc = sigma_u_pred_fc.sub(1).collapse()
    sigma1_pred_fc = sigma_pred_fc.sub(0).collapse()
    sigma2_pred_fc = sigma_pred_fc.sub(1).collapse()

    difference_sigma_fc = ufl.as_vector((sigma1_label_fc - sigma1_pred_fc, sigma2_label_fc - sigma2_pred_fc))
    difference_u_fc = u_label_fc - u_pred_fc

    sigma_u_error_dict['squared_L2'][i] = compute_squared_L2_norm(sigma_u_label_fc - sigma_u_pred_fc)
    sigma_u_error_dict['relative_squared_L2'][i] = sigma_u_error_dict['squared_L2'][i] / sigma_u_norm_dict['avg_squared_L2']
    sigma_u_error_dict['squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(difference_sigma_fc, difference_u_fc, mean_p_fc)
    sigma_u_error_dict['relative_squared_hdiv_h1'][i] = sigma_u_error_dict['squared_hdiv_h1'][i] / sigma_u_norm_dict['avg_squared_hdiv_h1']

for i in tqdm(range(num_test), desc='CG1 prediction errors'):
    cg1_sigma_label = test_CG1_sigma_fc_label_list[i]
    cg1_u_label = test_CG1_u_fc_label_list[i]
    cg1_sigma_pred = test_CG1_sigma_fc_pred_list[i]
    cg1_u_pred = test_CG1_u_fc_pred_list[i]

    cg1_sigma1_label = cg1_sigma_label.sub(0).collapse()
    cg1_sigma2_label = cg1_sigma_label.sub(1).collapse()
    cg1_sigma1_pred = cg1_sigma_pred.sub(0).collapse()
    cg1_sigma2_pred = cg1_sigma_pred.sub(1).collapse()
    difference_cg1_sigma = ufl.as_tensor([cg1_sigma1_label - cg1_sigma1_pred, cg1_sigma2_label - cg1_sigma2_pred])

    sigma_u_error_dict['cg1_squared_L2'][i] = compute_squared_L2_norm(cg1_sigma_label - cg1_sigma_pred) + compute_squared_L2_norm(cg1_u_label - cg1_u_pred)
    sigma_u_error_dict['relative_cg1_squared_L2'][i] = sigma_u_error_dict['cg1_squared_L2'][i] / sigma_u_norm_dict['avg_cg1_squared_L2']
    sigma_u_error_dict['cg1_squared_hdiv_h1'][i] = compute_squared_hdiv_h1_norm(difference_cg1_sigma, cg1_u_label - cg1_u_pred, mean_p_fc)
    sigma_u_error_dict['relative_cg1_squared_hdiv_h1'][i] = sigma_u_error_dict['cg1_squared_hdiv_h1'][i] / sigma_u_norm_dict['avg_cg1_squared_hdiv_h1']

sigma_u_error_dict['bochner_L2'] = np.sqrt(np.mean(sigma_u_error_dict['squared_L2']))
sigma_u_error_dict['relative_bochner_L2'] = np.sqrt(np.mean(sigma_u_error_dict['relative_squared_L2']))
sigma_u_error_dict['bochner_hdiv_h1'] = np.sqrt(np.mean(sigma_u_error_dict['squared_hdiv_h1']))
sigma_u_error_dict['relative_bochner_hdiv_h1'] = np.sqrt(np.mean(sigma_u_error_dict['relative_squared_hdiv_h1']))
sigma_u_error_dict['std_relative_L2'] = np.std(np.sqrt(sigma_u_error_dict['relative_squared_L2']))
sigma_u_error_dict['std_relative_hdiv_h1'] = np.std(np.sqrt(sigma_u_error_dict['relative_squared_hdiv_h1']))

sigma_u_error_dict['cg1_bochner_L2'] = np.sqrt(np.mean(sigma_u_error_dict['cg1_squared_L2']))
sigma_u_error_dict['relative_cg1_bochner_L2'] = np.sqrt(np.mean(sigma_u_error_dict['relative_cg1_squared_L2']))
sigma_u_error_dict['cg1_bochner_hdiv_h1'] = np.sqrt(np.mean(sigma_u_error_dict['cg1_squared_hdiv_h1']))
sigma_u_error_dict['relative_cg1_bochner_hdiv_h1'] = np.sqrt(np.mean(sigma_u_error_dict['relative_cg1_squared_hdiv_h1']))
sigma_u_error_dict['std_relative_cg1_L2'] = np.std(np.sqrt(sigma_u_error_dict['relative_cg1_squared_L2']))
sigma_u_error_dict['std_relative_cg1_hdiv_h1'] = np.std(np.sqrt(sigma_u_error_dict['relative_cg1_squared_hdiv_h1']))

np.save(model_test_outputs_path / 'sigma_u_norm_dict.npy', sigma_u_norm_dict)
np.save(model_test_outputs_path / 'sigma_u_error_dict.npy', sigma_u_error_dict)

print(f"sigma_u relative Bochner L2 error (std): {sigma_u_error_dict['relative_bochner_L2']:.2e} ({sigma_u_error_dict['std_relative_L2']:.2e})")
print(f"sigma_u relative Bochner H(div) x H1 error (std): {sigma_u_error_dict['relative_bochner_hdiv_h1']:.2e} ({sigma_u_error_dict['std_relative_hdiv_h1']:.2e})")
print(f"sigma_u relative CG1 Bochner L2 error (std): {sigma_u_error_dict['relative_cg1_bochner_L2']:.2e} ({sigma_u_error_dict['std_relative_cg1_L2']:.2e})")
print(f"sigma_u relative CG1 Bochner H(div) x H1 error (std): {sigma_u_error_dict['relative_cg1_bochner_hdiv_h1']:.2e} ({sigma_u_error_dict['std_relative_cg1_hdiv_h1']:.2e})")

# %% [markdown]
# ## DoF-assembled residual loss

# %%
compute_physical_loss_1 = elasticity_least_squares.compute_physical_loss_1
compute_physical_loss_2 = elasticity_least_squares.compute_physical_loss_2

dof_residual_loss_dict = {
    'loss_1': np.zeros(num_test),
    'loss_2': np.zeros(num_test),
    'total_loss': np.zeros(num_test),
    'sqrt_total_loss': np.zeros(num_test),
}
cg1_residual_loss_dict = {
    'loss_1': np.zeros(num_test),
    'loss_2': np.zeros(num_test),
    'total_loss': np.zeros(num_test),
    'sqrt_total_loss': np.zeros(num_test),
}

for test_index in tqdm(range(num_test), desc='DoF residual loss'):
    pred_fc = dolfinx.fem.Function(Vh['sigma_u'])
    pred_fc.x.array[:] = test_pred_sigma_u_dof[test_index]
    pred_sigma_fc = pred_fc.sub(0).collapse()
    pred_sigma1_fc, pred_sigma2_fc = ufl.split(pred_sigma_fc)
    pred_sigma_fc_ = ufl.as_vector((pred_sigma1_fc, pred_sigma2_fc))
    pred_u_fc = pred_fc.sub(1).collapse()

    p_fc = dolfinx.fem.Function(Vh['p'])
    p_fc.x.array[:] = test_p_dof[test_index]

    residual_loss_1 = compute_physical_loss_1(pred_sigma_fc_, pred_u_fc, p_fc)
    residual_loss_2 = compute_physical_loss_2(pred_sigma_fc_, pred_u_fc, p_fc)
    residual_loss = residual_loss_1 + residual_loss_2

    dof_residual_loss_dict['loss_1'][test_index] = residual_loss_1
    dof_residual_loss_dict['loss_2'][test_index] = residual_loss_2
    dof_residual_loss_dict['total_loss'][test_index] = residual_loss
    dof_residual_loss_dict['sqrt_total_loss'][test_index] = np.sqrt(residual_loss)

for test_index in tqdm(range(num_test), desc='CG1 residual loss'):
    cg1_sigma_pred = test_CG1_sigma_fc_pred_list[test_index]
    cg1_u_pred = test_CG1_u_fc_pred_list[test_index]
    cg1_sigma1_pred = cg1_sigma_pred.sub(0).collapse()
    cg1_sigma2_pred = cg1_sigma_pred.sub(1).collapse()
    cg1_sigma_pred_ = ufl.as_tensor([cg1_sigma1_pred, cg1_sigma2_pred])

    p_fc = dolfinx.fem.Function(Vh['p'])
    p_fc.x.array[:] = test_p_dof[test_index]

    residual_loss_1 = compute_physical_loss_1(cg1_sigma_pred_, cg1_u_pred, p_fc)
    residual_loss_2 = compute_physical_loss_2(cg1_sigma_pred_, cg1_u_pred, p_fc)
    residual_loss = residual_loss_1 + residual_loss_2

    cg1_residual_loss_dict['loss_1'][test_index] = residual_loss_1
    cg1_residual_loss_dict['loss_2'][test_index] = residual_loss_2
    cg1_residual_loss_dict['total_loss'][test_index] = residual_loss
    cg1_residual_loss_dict['sqrt_total_loss'][test_index] = np.sqrt(residual_loss)

np.save(model_test_outputs_path / 'fc_grid_residual_loss_dict.npy', fc_residual_loss_dict)
np.save(model_test_outputs_path / 'dof_residual_loss_dict.npy', dof_residual_loss_dict)
np.save(model_test_outputs_path / 'cg1_residual_loss_dict.npy', cg1_residual_loss_dict)

print(f"mean DoF residual loss 1: {np.mean(dof_residual_loss_dict['loss_1']):.2e} (std: {np.std(dof_residual_loss_dict['loss_1']):.2e})")
print(f"mean DoF residual loss 2: {np.mean(dof_residual_loss_dict['loss_2']):.2e} (std: {np.std(dof_residual_loss_dict['loss_2']):.2e})")
print(f"mean DoF total residual loss: {np.mean(dof_residual_loss_dict['total_loss']):.2e} (std: {np.std(dof_residual_loss_dict['total_loss']):.2e})")
print(f"mean CG1 residual loss 1: {np.mean(cg1_residual_loss_dict['loss_1']):.2e} (std: {np.std(cg1_residual_loss_dict['loss_1']):.2e})")
print(f"mean CG1 residual loss 2: {np.mean(cg1_residual_loss_dict['loss_2']):.2e} (std: {np.std(cg1_residual_loss_dict['loss_2']):.2e})")
print(f"mean CG1 total residual loss: {np.mean(cg1_residual_loss_dict['total_loss']):.2e} (std: {np.std(cg1_residual_loss_dict['total_loss']):.2e})")

# %% [markdown]
# ## Visualizations
#
# Parameter images and FEM-evaluated reference/prediction/difference plots for the projected solution fields, following `elasticity_fno.ipynb`.

# %%
def plot_ref_pred_diff(x, y, ref_f_grid_evals, pred_f_grid_evals, diff_f_grid_evals, variable_name,
                        levels=100,
                        ref_pred_format='%.3f',
                        ref_pred_colorbar_pad=0.02,
                        diff_colorbar_pad=0.01,
                        tick_labelsize=15,
                        colorbar_labelsize=15):

    vmin = min(ref_f_grid_evals.min(), pred_f_grid_evals.min())
    vmax = max(ref_f_grid_evals.max(), pred_f_grid_evals.max())

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

    cf0 = axs[0].tricontourf(x, y, ref_f_grid_evals, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax)
    axs[0].set_title(fr'Reference {variable_name}', fontsize=18)
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].set_aspect(1.0, adjustable='box')

    cf1 = axs[1].tricontourf(x, y, pred_f_grid_evals, levels=levels, cmap='turbo', vmin=vmin, vmax=vmax)
    axs[1].set_title(fr'Prediction {variable_name}', fontsize=18)
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].set_aspect(1.0, adjustable='box')

    cbar_shared = fig.colorbar(cf1, ax=[axs[0], axs[1]], format=ref_pred_format, pad=ref_pred_colorbar_pad, aspect=40)
    cbar_shared.ax.tick_params(labelsize=colorbar_labelsize)
    cbar_shared.locator = ticker.MaxNLocator(nbins=5)
    cbar_shared.update_ticks()

    cf2 = axs[2].tricontourf(x, y, diff_f_grid_evals, levels=levels, cmap='turbo')
    axs[2].set_title(fr'Difference {variable_name}', fontsize=18)
    axs[2].set_xticklabels([])
    axs[2].set_yticklabels([])
    axs[2].set_aspect(1.0, adjustable='box')

    cbar_diff = fig.colorbar(cf2, ax=axs[2], pad=diff_colorbar_pad)
    cbar_diff.ax.tick_params(labelsize=colorbar_labelsize)
    cbar_diff.locator = ticker.MaxNLocator(nbins=5)
    cbar_diff.update_ticks()

    for ax in axs:
        ax.tick_params(left=False, bottom=False)

    return fig

visualization_dir = model_test_outputs_path / 'visualizations'
visualization_dir.mkdir(parents=True, exist_ok=True)
num_visualize_actual = min(num_visualize, num_test)

x = mesh.geometry.x[:, 0]
y = mesh.geometry.x[:, 1]

for i in range(num_visualize_actual):
    test_p_fc = dolfinx.fem.Function(Vh['p'])
    test_p_fc.x.array[:] = test_p_dof[i]
    fig, ax = plt.subplots(figsize=(10, 5))
    tric = ax.tricontourf(x, y, evaluate_expression(mesh, test_p_fc, mesh.geometry.x)[1][:, 0], cmap='turbo', levels=100)
    cbar = fig.colorbar(tric, ax=ax, fraction=0.0235, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title('parameter', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0, adjustable='box')
    fig.savefig(visualization_dir / f'test_p_{i}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

sigma_component_titles = [r'$\sigma_{11}^{\circ}$', r'$\sigma_{12}^{\circ}$', r'$\sigma_{21}^{\circ}$', r'$\sigma_{22}^{\circ}$']
sigma_component_files = ['sigma_11', 'sigma_12', 'sigma_21', 'sigma_22']
u_component_titles = [r'$u_1^{\circ}$', r'$u_2^{\circ}$']
u_component_files = ['u1', 'u2']

for test_sample_index in range(num_visualize_actual):
    pred_sigma_u_fc = dolfinx.fem.Function(Vh['sigma_u'])
    pred_sigma_u_fc.x.array[:] = test_pred_sigma_u_dof[test_sample_index]
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

    for c in range(2):
        fig = plot_ref_pred_diff(x, y, ref_u_grid_evals[:, c], pred_u_grid_evals[:, c], diff_u_grid_evals[:, c], u_component_titles[c],
                                 levels=100, ref_pred_format='%.2f', ref_pred_colorbar_pad=0.02, diff_colorbar_pad=0.02)
        fig.savefig(visualization_dir / f'{u_component_files[c]}_ref_pred_diff_{test_sample_index}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    for c in range(4):
        fig = plot_ref_pred_diff(x, y, ref_sigma_grid_evals[:, c], pred_sigma_grid_evals[:, c], diff_sigma_grid_evals[:, c], sigma_component_titles[c],
                                 levels=100, ref_pred_format='%.1f', ref_pred_colorbar_pad=0.02, diff_colorbar_pad=0.02)
        fig.savefig(visualization_dir / f'{sigma_component_files[c]}_ref_pred_diff_{test_sample_index}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f'Saved projected visualization figures for {num_visualize_actual} sample(s) to {visualization_dir}')

# %% [markdown]
# ## Save summary

# %%
def summarize_loss_dict(loss_dict):
    return {
        'loss_1_mean': float(np.mean(loss_dict['loss_1'])),
        'loss_1_std': float(np.std(loss_dict['loss_1'])),
        'loss_2_mean': float(np.mean(loss_dict['loss_2'])),
        'loss_2_std': float(np.std(loss_dict['loss_2'])),
        'total_loss_mean': float(np.mean(loss_dict['total_loss'])),
        'total_loss_std': float(np.std(loss_dict['total_loss'])),
        'sqrt_total_loss_mean': float(np.mean(loss_dict['sqrt_total_loss'])),
        'sqrt_total_loss_std': float(np.std(loss_dict['sqrt_total_loss'])),
    }

metrics_summary = {
    'num_train': num_train,
    'num_valid': num_valid,
    'num_test': num_test,
    'iterations': iterations,
    'relative_bochner_L2': float(sigma_u_error_dict['relative_bochner_L2']),
    'std_relative_L2': float(sigma_u_error_dict['std_relative_L2']),
    'relative_bochner_hdiv_h1': float(sigma_u_error_dict['relative_bochner_hdiv_h1']),
    'std_relative_hdiv_h1': float(sigma_u_error_dict['std_relative_hdiv_h1']),
    'relative_cg1_bochner_L2': float(sigma_u_error_dict['relative_cg1_bochner_L2']),
    'std_relative_cg1_L2': float(sigma_u_error_dict['std_relative_cg1_L2']),
    'relative_cg1_bochner_hdiv_h1': float(sigma_u_error_dict['relative_cg1_bochner_hdiv_h1']),
    'std_relative_cg1_hdiv_h1': float(sigma_u_error_dict['std_relative_cg1_hdiv_h1']),
    'fc_grid_quadrature': 'averaged rectangle rule: sum(pointwise squared residual) * rectangle_cell_volume / (rectangle_cell_volume * number_of_rectangle_cells)',
    'fc_grid_residual_mse': summarize_loss_dict(fc_residual_loss_dict),
    'dof_residual_loss': summarize_loss_dict(dof_residual_loss_dict),
    'cg1_residual_loss': summarize_loss_dict(cg1_residual_loss_dict),
    'num_data_train': num_data_train,
    'num_physics_train': num_physics_train,
    'data_iterations': data_iterations,
    'physics_iterations': physics_iterations,
    'best_model_path': str(best_model_path),
    'latest_model_path': str(latest_model_path),
}

summary_path = model_test_outputs_path / 'metrics_summary.json'
with open(summary_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(json.dumps(metrics_summary, indent=2))
print(f'saved summary to {summary_path}')
