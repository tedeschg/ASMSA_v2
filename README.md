# ASMSA v2

Analysis and Sampling of Molecular Simulations by adversarial Autoencoders — PyTorch rewrite.

v2 fixes three concrete problems in the v1 AAE:

1. The discriminator loss is now a real BCE-with-logits with spectral-normalized disc layers (v1 had a comment claiming BCE but implemented an unstable raw-logits + random-scaling hybrid).
2. Three separate optimizers for AE, discriminator, and the encoder-adversarial step, so Adam moments stay clean (v1 shared one optimizer across all three).
3. A single AAE can now be jointly trained on frames from multiple molecules with different feature-vector lengths, via per-molecule input/output adapters feeding a shared trunk and discriminator (v1 was single-molecule only).

The featurization layer (`asmsa_v2.featurize.molecule`, `asmsa_v2.featurize.mol_model`) is reused as-is from v1 — it was already PyTorch.

## Install

### Docker (recommended — includes GROMACS)

```bash
docker build -t asmsa_v2 -f docker/Dockerfile .
docker run --gpus all --rm -p 8888:8888 -v $PWD:/workspace asmsa_v2
```

Open the Jupyter URL printed in the terminal. For CPU-only, swap the pip index in `docker/Dockerfile` to the `cpu` channel.

### Local (pip)

```bash
pip install -e .[dev]
```

GROMACS must be installed separately (`apt-get install gromacs` on Ubuntu).

## Quickstart

See `notebooks/01-train-multi-mol.ipynb` — trains one AAE jointly on two molecules with different atom counts and plots the shared latent space.

```python
from asmsa_v2 import AAE, Trainer, TrainConfig, IsotropicNormalPrior, MultiMolDataset
from torch.utils.data import DataLoader

ds = MultiMolDataset({"ala2": X_a, "chignolin": X_b})
loader = DataLoader(ds, batch_sampler=ds.round_robin_sampler(batch_size=128))

model = AAE(feature_dims={"ala2": X_a.shape[1], "chignolin": X_b.shape[1]}, latent_dim=2)
Trainer(model, prior=IsotropicNormalPrior(2), config=TrainConfig(epochs=50)).fit(loader)
```

## Tests

```bash
pytest tests/
```
