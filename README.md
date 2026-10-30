# ASMSA v2

Analysis and Sampling of Molecular Simulations by adversarial Autoencoders — PyTorch rewrite.

v2 is designed around the **protein-ligand (or protein-protein) complex** use case: train an AAE on an MD trajectory of a complex and the learned 2-D latent space clusters binding poses / conformational states. Compared to v1 it fixes three concrete problems and adds one capability:

1. The discriminator is a real BCE-with-logits critic with spectral-normalized layers (v1 claimed BCE in a comment but implemented an unstable raw-logits + random-scaling hybrid).
2. Three separate Adam optimizers for AE, discriminator, and the encoder-adversarial step keep moments cleanly separated across objectives (v1 shared one optimizer across all three).
3. The featurizer now understands **complexes**: alongside the standard intra-component bonds/angles/dihedrals, `ComplexMolecule` emits inter-component distance features between named atom subsets of the two parts of the system — the signal that encodes the binding pose.

The featurization layer (`asmsa_v2.featurize.molecule`, `.mol_model`) is reused as-is from v1 (it was already PyTorch); `ComplexMolecule` is a thin wrapper on top.

As a free bonus from the architecture, the AAE can also be trained jointly on multiple separately-topologized molecules by passing more than one entry to `feature_dims` — useful later if you want a universal latent space across several protein-ligand pairs.

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

See `notebooks/01-train-complex.ipynb` for the full worked example (synthetic protein+ligand trajectory, two binding poses, latent-space clustering).

```python
import mdtraj as md
from asmsa_v2 import (
    AAE, Trainer, TrainConfig, IsotropicNormalPrior,
    ComplexMolecule, MultiMolDataset, multi_mol_collate, featurize_frames,
)
from torch.utils.data import DataLoader

# 1. featurize the complex trajectory
traj = md.load('traj.xtc', top='complex.pdb')
cm = ComplexMolecule(
    pdb='complex.pdb', n_atoms=traj.n_atoms,
    components={
        'protein': traj.top.select('protein'),
        'ligand':  traj.top.select('resname LIG'),
    },
    diheds=my_dihed_indices,
    inter_subsets={                              # which atoms contribute inter-distances
        'protein': traj.top.select('name CA'),
        'ligand':  traj.top.select('resname LIG and not symbol H'),
    },
    inter_mode='dense',
)
X = featurize_frames(traj.xyz, cm.get_model())   # [n_frames, n_features]

# 2. train
ds = MultiMolDataset({'complex': X})
loader = DataLoader(ds, batch_sampler=ds.round_robin_sampler(batch_size=128),
                    collate_fn=multi_mol_collate)
model = AAE(feature_dims=ds.feature_dims, latent_dim=2)
Trainer(model, prior=IsotropicNormalPrior(2),
        config=TrainConfig(epochs=50)).fit(loader)
```

## Tests

```bash
pytest tests/
```
