#!/bin/bash

set -euo pipefail

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"
results_path="$repo_path/results"

problems=(
    "elasticity"
    "poisson_setup1"
    "poisson_setup2"
)

dataset_dirs=(
    "train_dataset"
    "test_dataset"
)

model_output_dirs=(
    "model_train_outputs"
    "model_test_outputs"
)

model_names=(
    "fno"
    "pcanet"
    "pcanet_new_basis"
    "rbno_physics_loss"
    "rbno_physics_loss_pod_data_loss"
    "rbno_pod_data_loss"
)

mkdir -p "$results_path"

for problem in "${problems[@]}"; do
    problem_results_path="$results_path/$problem"
    mkdir -p "$problem_results_path"

    for dataset_dir in "${dataset_dirs[@]}"; do
        mkdir -p "$problem_results_path/$dataset_dir"
    done

    for model_output_dir in "${model_output_dirs[@]}"; do
        model_output_path="$problem_results_path/$model_output_dir"
        mkdir -p "$model_output_path"

        for model_name in "${model_names[@]}"; do
            mkdir -p "$model_output_path/$model_name"
        done
    done
done

printf 'Created results directory tree under %s\n' "$results_path"
