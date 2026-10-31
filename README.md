# ASMSA v2

**Analysis and Sampling of Molecular Simulations by adversarial Autoencoders**

ASMSA learns low-dimensional collective variables from molecular-dynamics trajectories. It featurizes molecular geometry, trains an autoencoder, and can export latent coordinates for analysis and enhanced sampling. The workflow supports single molecules and complexes, including protein-ligand systems where inter-component distances encode binding poses.

## Get started

The quickest way to see the end-to-end idea is the synthetic demo. It creates its own protein-ligand trajectory, so no simulation data or GROMACS installation is needed for this first notebook. With [uv](https://docs.astral.sh/uv/), run from the repository root:

```bash
uv sync --extra full
uv run --extra full jupyter lab notebooks/00-quickstart.ipynb
```

For GPU acceleration, install the PyTorch build appropriate to your CUDA environment before or after installing ASMSA.

## Workflow

Use the numbered notebooks for a real project:

| Notebook | Purpose |
| --- | --- |
| [00 Quickstart](notebooks/00-quickstart.ipynb) | Explore the pipeline with generated data. |
| [01 Setup](notebooks/01-setup.ipynb) | Define the system and project configuration. |
| [02 Prepare](notebooks/02-prepare.ipynb) | Featurize the trajectory, split the data, and standardize features. |
| [03 Tune](notebooks/03-tune.ipynb) | Search model and training settings. Optuna enables TPE sampling and pruning; without it, tuning falls back to random search. |
| [04 Train](notebooks/04-train.ipynb) | Train and evaluate the autoencoder. |
| [05 PLUMED](notebooks/05-plumed.ipynb) | Export latent collective variables for PLUMED. |
| [06 MD](notebooks/06-md.ipynb) | Run and inspect molecular-dynamics sampling. |
| [07 Analysis](notebooks/07-analysis.ipynb) | Analyze and cluster latent-space states. |

The notebooks share project settings and generated files under `notebooks/run/`. Start at 01 for your own trajectory; 00 is a standalone demonstration.

## Installation

ASMSA requires Python 3.10 or newer. From the repository root, use uv to create a project environment with the full notebook dependencies:

```bash
uv sync --extra full
```

If you prefer pip, install into your active environment with:

```bash
python -m pip install -e ".[full]"
```

Optional dependency groups are available when you only need part of the workflow:

| Extra | Includes |
| --- | --- |
| `dev` | pytest and Jupyter |
| `tune` | Optuna and pandas |
| `analysis` | scikit-learn, hdbscan, and pandas |
| `viz` | Plotly, py3Dmol, and ipywidgets |
| `full` | All notebook and development extras |

For development and tests without the optional analysis, tuning, and visualization libraries, use `uv sync --extra dev` or `python -m pip install -e ".[dev]"`. To combine extras, use `uv sync --extra dev --extra tune` or `python -m pip install -e ".[dev,tune]"`. GROMACS is needed for the MD workflow, and PLUMED must be available in the GROMACS installation for PLUMED-biased simulations. They are separate system tools, not Python dependencies.

The Docker image installs GROMACS and the full Python extras. Build and launch it with:

```bash
docker build -t asmsa_v2 -f docker/Dockerfile .
docker run --rm --gpus all -p 8888:8888 -v "$PWD:/workspace" asmsa_v2
```

The image builds GROMACS 2026.3 with PLUMED's PyTorch module and includes CUDA-enabled PyTorch 2.5.1. The default GROMACS build targets CUDA architecture 75; override it for your GPU, for example `--build-arg CUDA_ARCH=86` for an RTX 3060. Building or running the GPU-enabled image requires the NVIDIA CUDA toolkit image and, at runtime, NVIDIA Container Toolkit on the host. To build GROMACS without CUDA, use `--build-arg GMX_GPU=OFF`; this does not change the CUDA-enabled PyTorch package.

## Project layout

- `src/asmsa_v2/` — featurization, data preparation, models, training, tuning, analysis, export, and MD helpers.
- `notebooks/` — the guided workflow and quickstart demo.
- `notebooks/demo_data/` — small trajectory and topology used by the notebooks.
- `notebooks/run/` — local project configuration and generated run artifacts.
- `docker/` — Docker image and notebook entrypoint.

## Tests

Install the development dependencies, then run the test suite from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```
