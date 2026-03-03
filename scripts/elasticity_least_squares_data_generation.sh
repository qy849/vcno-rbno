#!/bin/bash

# bash elasticity_least_squares_data_generation.sh | tee ../logs/elasticity_least_squares_data_generation.log

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"

mesh_config_path="$repo_path/configs/elasticity/config_data/config_mesh.yaml"
function_space_config_path="$repo_path/configs/elasticity/config_data/config_function_space.yaml"
function_space_finer_config_path="$repo_path/configs/elasticity/config_data/config_function_space_finer.yaml"


output_reduced_basis_config_path="$repo_path/configs/elasticity/config_data/config_output_reduced_basis.yaml"

train_dataset_path="$repo_path/results/elasticity/train_dataset"
test_dataset_path="$repo_path/results/elasticity/test_dataset"

elasticity_least_squares_path="$repo_path/data_generation/differential_equations/elasticity_least_squares.py"
elasticity_least_squares_evaluating_grid_points_path="$repo_path/data_generation/differential_equations/elasticity_least_squares_evaluating_grid_points.py"
elasticity_least_squares_finer_path="$repo_path/data_generation/differential_equations/elasticity_least_squares_finer.py"
elasticity_least_squares_evaluating_grid_points_finer_path="$repo_path/data_generation/differential_equations/elasticity_least_squares_evaluating_grid_points_finer.py"

elasticity_least_squares_auxiliary_pde_path="$repo_path/data_generation/differential_equations/elasticity_least_squares_auxiliary_pde.py"
elasticity_least_squares_hdiv_h1_pod_basis_path="$repo_path/data_generation/output_reduced_basis/elasticity_least_squares_hdiv_h1_pod_basis.py"
elasticity_least_squares_hdiv_h1_reduced_loss_weight_path="$repo_path/data_generation/reduced_loss_weight/elasticity_least_squares_hdiv_h1_reduced_loss_weight.py"
elasticity_least_squares_reference_reduced_minimizers_path="$repo_path/data_generation/reference_reduced_minimizers/reference_reduced_minimizers.py"



python "$elasticity_least_squares_auxiliary_pde_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --train_dataset_path "$train_dataset_path" \
    --test_dataset_path "$test_dataset_path"


for dataset_path in "$train_dataset_path" "$test_dataset_path"; do
    mpirun -n 50 python "$elasticity_least_squares_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" 
done


for dataset_path in "$train_dataset_path" "$test_dataset_path"; do
    mpirun -n 50 python "$elasticity_least_squares_evaluating_grid_points_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path"
done


python "$elasticity_least_squares_hdiv_h1_pod_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --output_reduced_basis_config_path "$output_reduced_basis_config_path" \
    --train_dataset_path "$train_dataset_path" \
    --test_dataset_path "$test_dataset_path"

for dataset_path in "$train_dataset_path" "$test_dataset_path"; do
    mpirun -n 50 python "$elasticity_least_squares_hdiv_h1_reduced_loss_weight_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --output_reduced_basis_config_path "$output_reduced_basis_config_path" \
        --dataset_path "$dataset_path" 
    done

for dataset_path in "$train_dataset_path" "$test_dataset_path"; do
    python "$elasticity_least_squares_reference_reduced_minimizers_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --output_reduced_basis_config_path "$output_reduced_basis_config_path" \
        --dataset_path "$dataset_path" 
done


mpirun -n 50 python "$elasticity_least_squares_finer_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_finer_config_path" \
    --dataset_path "$test_dataset_path" 


mpirun -n 50 python "$elasticity_least_squares_evaluating_grid_points_finer_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_finer_config_path" \
    --dataset_path "$test_dataset_path"