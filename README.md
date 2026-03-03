# vcno-rbno

This repository provides the code and data workflow for generating datasets, reproducing analysis, and training models for the VCNO-RBNO experiments on Poisson and elasticity problem setups.

## Repository layout

- `configs/`: YAML configuration files for meshes, function spaces, and reduced-basis settings.
- `data_generation/`: PDE solvers, probability-measure generation, reduced-basis construction, reduced loss weights, and reference minimizers.
- `scripts/`: shell entry points for creating directories, downloading published data, and running the main data generation pipelines.
- `results/`: generated data. This directory is not committed to Git.
- `models/`: neural network model definitions used by the notebooks.
- `analysis/`: post-processing notebooks and saved figures.
- `train/`: model training notebooks and supporting utilities.
- `utils/`: shared IO, plotting, MPI, and helper utilities.

## Environment setup

This repository does not yet include a pinned `environment.yml`, so the instructions below are the recommended starting point based on the current codebase. The notebooks in this repo use a Jupyter kernel named `fenicsx`.

### 1. Create and activate a conda environment

If you already have a working FEniCSx environment, you can reuse it. Otherwise:

```bash
conda create -n fenicsx python=3.11
conda activate fenicsx
```

### 2. Install FEniCSx, MPI, and scientific Python packages

```bash
conda install -c conda-forge fenics-dolfinx mpich scifem pyvista matplotlib scipy pyyaml tqdm jupyterlab ipykernel
```

### 3. Install PyTorch

GPU-enabled PyTorch is recommended for the training notebooks.

Example for NVIDIA CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Replace `cu126` with the CUDA build that matches your machine. The safest way to choose that tag is the official PyTorch install selector:

`https://pytorch.org/get-started/locally/`

### 4. Register the notebook kernel

```bash
python -m ipykernel install --user --name fenicsx --display-name "fenicsx"
```

### 5. Verify the environment

Run this once from the repository root:

```bash
python - <<'PY'
import dolfinx
import torch
from mpi4py import MPI
print("dolfinx:", dolfinx.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
print("mpi size:", MPI.COMM_WORLD.size)
PY
```

## Quick start

All commands below are meant to be run from the repository root:

```bash
cd /path/to/vcno-rbno
```

Create the expected `results/` directory tree:

```bash
bash scripts/create_results_directories.sh
```

## Download published input data

The public starting data is hosted on Hugging Face. Download it into `results/` with:

```bash
bash scripts/download_results.sh https://huggingface.co/datasets/yqiu327/vcno-rbno-results/resolve/main
```

You can also download only selected problem folders:

```bash
bash scripts/download_results.sh https://huggingface.co/datasets/yqiu327/vcno-rbno-results/resolve/main elasticity poisson_setup2
```

What this does:

- creates the expected `results/` directory tree if it does not already exist
- downloads the published archive parts and checksum manifest
- verifies file integrity with SHA256
- reconstructs the downloaded data back into `results/`

The download script requires either `curl` or `wget`.

## Generate the remaining data

After the initial data is in `results/`, run the shell pipelines in `scripts/` to generate the remaining derived data used by training and analysis.

### Important MPI note

The data generation scripts currently use `mpirun -n 50`. Change that value in the scripts if your workstation or cluster allocation is different.

### Elasticity

```bash
bash scripts/elasticity_least_squares_data_generation.sh
```

This pipeline generates the remaining elasticity outputs, including:

- auxiliary PDE quantities
- least-squares solution data
- grid-point evaluations
- POD reduced bases
- reduced loss weights
- reference reduced minimizers
- finer-grid test data

### Poisson setup 1

```bash
bash scripts/poisson_least_squares_setup1_data_generation.sh
```

The first probability-measure generation block in this script is currently commented out, so the script assumes the starting dataset already exists under `results/poisson_setup1/`. This is consistent with the Hugging Face download workflow.

### Poisson setup 2

```bash
bash scripts/poisson_least_squares_setup2_data_generation.sh
```

As with setup 1, the initial probability-measure generation block is currently commented out, so the script starts from the downloaded input data already placed in `results/poisson_setup2/`.

## Analysis notebooks

Analysis notebooks live in `analysis/`, again grouped by problem:

- `analysis/elasticity/`
- `analysis/poisson_setup1/`
- `analysis/poisson_setup2/`

These notebooks are used for:

- visualization of samples and reconstructed fields
- low-rank approximation studies
- trailing eigenvalue analysis
- reduced-basis loss analysis
- reconstruction error analysis
- wall-clock time comparisons

Saved figures are kept next to the notebooks in the same folders.

## Training notebooks

Model training notebooks live in `train/`, grouped by problem:

- `train/elasticity/`
- `train/poisson_setup1/`
- `train/poisson_setup2/`

Examples include:

- `train/elasticity/elasticity_fno.ipynb`
- `train/elasticity/elasticity_rbno_physics_loss.ipynb`
- `train/poisson_setup1/poisson_setup1_pcanet.ipynb`
- `train/poisson_setup2/poisson_setup2_rbno_pod_data_loss.ipynb`

The shared training utilities are:

- `train/train_loss.py`
- `train/train_utils.py`
- `train/soap.py`

Before opening the notebooks:

- activate the same `fenicsx` environment
- launch Jupyter from the repository root so local imports resolve cleanly
- confirm that the required data already exists under `results/`
- verify that PyTorch sees your GPU if you plan to train on GPU

Launch Jupyter with:

```bash
jupyter lab
```

Then select the `fenicsx` kernel inside the notebook.

## Recommended end-to-end workflow

For a new user, the most straightforward sequence is:

1. Create and activate the `fenicsx` environment.
2. Install the required packages and register the `fenicsx` Jupyter kernel.
3. Run `bash scripts/create_results_directories.sh`.
4. Run `bash scripts/download_results.sh https://huggingface.co/datasets/yqiu327/vcno-rbno-results/resolve/main`.
5. Run the problem-specific data generation script you need.
6. Open the matching notebook in `analysis/` to inspect the generated data and reproduced figures.
7. Open the matching notebook in `train/` to train or inspect models.

## Updating the public data snapshot

If you maintain the Hugging Face dataset and want to publish a new `results/` snapshot:

```bash
bash scripts/package_results_for_download.sh /tmp/vcno-rbno-results-download
hf upload-large-folder yqiu327/vcno-rbno-results /tmp/vcno-rbno-results-download --repo-type dataset
```

The download script in this repository is designed to consume the files produced by `scripts/package_results_for_download.sh`.

## Citation

If this repository is helpful in your research, please cite:

```bibtex
@article{qiu2025variationally,
  title={Variationally correct operator learning: Reduced basis neural operator with a posteriori error estimation},
  author={Qiu, Yuan and Dahmen, Wolfgang and Chen, Peng},
  journal={arXiv preprint arXiv:2512.21319},
  year={2025}
}
```
