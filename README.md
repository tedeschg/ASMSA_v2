# ASMSA v2

**A**nalysis and **S**ampling of **M**olecular **S**imulations by adversarial **A**utoencoders — PyTorch rewrite.

Train an autoencoder on an MD trajectory, and its low-dimensional latent space becomes both an
interpretable map of the conformational landscape and a set of **collective variables** you can
hand to PLUMED to drive enhanced sampling.

v2 keeps v1's goal and protocol, reimplements the model in PyTorch, and is built around the
**protein-ligand (or protein-protein) complex** case while still supporting a single molecule.

---

## Pipeline

Eight notebooks in `notebooks/`, each one stage. They share state through a validated
`asmsa.yaml` written by the first one.

| Notebook | Does |
|---|---|
| [`00-quickstart`](notebooks/00-quickstart.ipynb) | The whole idea in ~2 minutes on a synthetic complex. No data or GROMACS needed. |
| [`01-setup`](notebooks/01-setup.ipynb) | Describe the system once, in a `Project`. Preflight-check the inputs. |
| [`02-prepare`](notebooks/02-prepare.ipynb) | GROMACS topology, train/val/test split, featurization, standardization. |
| [`03-tune`](notebooks/03-tune.ipynb) | Hyperparameter search (Optuna), scored on latent quality. |
| [`04-train`](notebooks/04-train.ipynb) | Train with validation and model selection; judge whether the latent is usable. |
| [`05-plumed`](notebooks/05-plumed.ipynb) | Export the CV to TorchScript, resolve atom numbering, write `plumed.dat`. |
| [`06-md`](notebooks/06-md.ipynb) | Solvate, equilibrate, run metadynamics, compute free energies. |
| [`07-analysis`](notebooks/07-analysis.ipynb) | Cluster the latent into states, export structures, explore interactively. |

Notebooks 00–05 and 07 run end to end on the built-in synthetic demo system, so the pipeline is
usable before you have data and without GROMACS installed. `06-md` needs real GROMACS and PLUMED;
without them it runs in dry-run mode, printing the protocol and exporting it as a shell script.

---

## Install

### Local

```bash
pip install -e '.[full]'
```

GROMACS and PLUMED are separate, and only needed for `06-md`. PLUMED must be built with the
**pytorch module enabled** (`--enable-modules=pytorch`), otherwise `PYTORCH_MODEL_CV` does not
exist and the learned CV cannot be used.

### Docker (bundles GROMACS)

```bash
docker build -t asmsa_v2 -f docker/Dockerfile .
docker run --gpus all --rm -p 8888:8888 -v $PWD:/workspace asmsa_v2
```

Swap the pip index in `docker/Dockerfile` to the `cpu` channel for CPU-only. The models here are
small; v1's own notes observed that a GPU buys little.

---

## What changed from v1

### Fixed

- **The discriminator is a real critic.** v1's comment claimed binary cross-entropy but it
  implemented `mean(logits * U(1, 1.05))` — an unbounded linear objective with random scaling,
  not BCE. v2 uses BCE-with-logits plus a non-saturating generator loss, on spectral-normalized
  layers, with an optional R1 gradient penalty.
- **Optimizer state is separated.** v1 shared one Adam across the autoencoder, discriminator and
  adversarial-encoder steps, so the moment estimates of three different objectives were mixed.
- **Features are standardized again.** On a realistic complex, covalent-bond features have a
  standard deviation around 0.007 nm while inter-component distances span nanometres — a ratio of
  ~200×, with the narrow blocks holding under 0.01% of the total variance. v1 standardized and
  folded the transform into the exported model; that step must not be skipped, or the
  reconstruction loss sees only the widest block.
- **Validation is first class.** Early stopping on a *training* loss cannot detect overfitting,
  and under an adversarial objective it is not even monotone. v2 trains against a held-out split,
  restores the best weights, and uses a non-shuffling validation loader so the metric is
  reproducible between passes.
- **The encoder runs once per batch.** Three forward passes per batch also meant the encoder's
  BatchNorm running statistics were updated three times for one batch, so the buffers used at
  evaluation time averaged three differently-normalized views of the same data.
- **Atom numbering is resolved, not guessed.** `pdb2gmx` adds hydrogens and reorders atoms, so
  PLUMED's `ATOMS=` list must enumerate atoms in the featurizer's order while referring to
  topology indices. v1 derived this from an index file's sort order and advised checking it by
  hand. v2 matches on `(chain, residue number, residue name, atom name)` with force-field alias
  handling, names any atom it cannot match, and refuses to write input it cannot justify.
- Smaller things: the batch sampler reported more batches than it produced; `TrainConfig.batch_size`
  was silently ignored; force-field angle reference values were plain tensors, so `.to('cuda')`
  left them on the CPU and GPU featurization crashed; `encode_all` flipped a model out of `eval()`.

### Added

- **Non-adversarial latent matching.** `latent_reg='mmd'` (Wasserstein autoencoder, Tolstikhin
  et al., ICLR 2018) and `'sliced'` (Kolouri et al., 2018) match the aggregate posterior to the
  prior with no minimax game, no second learning rate to balance, and much less tendency to
  collapse a 2-D latent onto a line. The classic AAE remains available as `'adversarial'`.
- **Time-lagged reconstruction.** `time_lag > 0` reconstructs the frame τ steps ahead, so the
  latent encodes the *slow* degrees of freedom rather than whichever carry the most instantaneous
  variance (Wehmeyer & Noé, JCP 148, 241703 (2018)). For enhanced-sampling CVs this is usually
  what you want.
- **Metrics that answer distinct questions**: reconstruction, k-NN KL prior match (v1's
  model-selection criterion, preserved exactly), **VAMP-2** score and implied timescales
  (Mardt et al., Nat. Commun. 9, 5 (2018)) for slow-dynamics content, and a participation-ratio
  effective dimension that detects latent collapse. `evaluate_latent` renders them as a table
  with verdicts.
- **Hyperparameter tuning that survives the rewrite.** v1's keras-tuner + Kubernetes tuner was
  welded to TensorFlow. v2 uses Optuna locally, with TPE sampling and pruning, architectures
  parameterized in v1's seed/layer terms so published ranges transfer, and results as portable
  JSON.
- **Free energies two ways.** `plumed sum_hills`, and a pure-NumPy reweighting of the biased
  trajectory using the recorded bias and `c(t)` (Tiwary & Parrinello, JPC B 119, 736 (2015)).
  Agreement between the two is the strongest convergence evidence available.
- **Inter-component features.** `ComplexMolecule` emits distances across the interface alongside
  the usual intra-component internal coordinates. These are what encode the binding pose; without
  them the latent cannot tell where the ligand sits. `auto_features` picks the atom subsets
  (`ca_heavy`, `contact`, `all`).
- **Multi-molecule training.** The discriminator sees only `z`, so passing several entries to
  `feature_dims` trains one shared latent space across separately-topologized systems.
- **Local-first execution behind a runner interface**, so the notebooks work off-cluster, with a
  resumable MD protocol that skips steps whose outputs already exist.

### Deliberately not carried over

- **Kubernetes job submission** for GROMACS and distributed tuning. v1 required one specific
  cluster. Execution sits behind `asmsa_v2.md.Runner`, so a cluster backend can be added without
  touching the notebooks.
- **v1's sparse-masked multi-model trick**, which packed several architectures into one masked
  graph. Replaced by explicit per-molecule adapters.
- **`with_density`** (matching latent density to a conformational-space density estimate).
  `with_cv1_bias` is kept.
- **OPES.** PLUMED's current recommendation, but `06-md` emits well-tempered METAD only, for
  continuity with v1's protocols and existing `HILLS` files.

---

## Library quickstart

```python
import mdtraj as md
import asmsa_v2 as asmsa

traj = md.load('traj.xtc', top='complex.pdb')

# 1. featurize: intra-component internal coordinates + inter-component distances
molecule = asmsa.auto_features(
    traj,
    components={'protein': traj.top.select('protein'),
                'ligand':  traj.top.select('resname LIG')},
    inter_strategy='ca_heavy',          # or 'contact' for a large protein
    extract_dihedrals=True, pdb_for_dihedrals='complex.pdb',
)
X = asmsa.featurize_frames(traj.xyz, molecule.get_model())

# 2. split chronologically, and standardize on the training split only
split = asmsa.split_frames(len(X), strategy='chronological')
scaler = asmsa.FeatureScaler().fit(X[split.train])
Xs = scaler.transform(X)

# 3. train with validation and best-weight restoration
train_ds = asmsa.MultiMolDataset({'complex': Xs[split.train]})
val_ds = asmsa.MultiMolDataset({'complex': Xs[split.val]})

model = asmsa.AAE(feature_dims=train_ds.feature_dims, latent_dim=2)
trainer = asmsa.Trainer(
    model,
    prior=asmsa.IsotropicNormalPrior(2),
    config=asmsa.TrainConfig(epochs=300, latent_reg='mmd', lam_latent=10.0),
    callbacks=[asmsa.EarlyStopping(patience=30)],
)
trainer.fit(train_ds.loader(batch_size=128), val_ds.eval_loader())

# 4. judge it
z = trainer.encode_all({'complex': Xs})['complex'].numpy()
print(asmsa.evaluate_latent(z, prior=asmsa.IsotropicNormalPrior(2), vamp_lag=20))

# 5. export the collective variable for PLUMED
from asmsa_v2.export import export_cv_models, map_pdb_to_topology, cv_grid, write_plumed_dat

paths = export_cv_models(model, molecule.get_model(), scaler,
                         n_atoms=traj.n_atoms, out_dir='models', name='cv')
atom_map = map_pdb_to_topology('complex.pdb', 'topol.gro')   # raises on any mismatch
write_plumed_dat('plumed.dat', paths['plumed'].name, atom_map, cv_grid(z))
```

The exported model contains the **whole chain** — featurization, standardization, encoder — because
PLUMED only ever passes raw coordinates. `export_cv_models` traces in eval mode on CPU and verifies
the result against the eager model frame by frame, raising rather than writing a model that would
bias the wrong coordinate.

---

## Tests

```bash
pytest tests/
```

The numerically consequential parts are checked against known answers rather than just for
not-crashing: VAMP-2 against an AR(1) process with an analytic relaxation time, the KL estimator
against identical and shifted distributions, FES reweighting against an analytically known
double well, and the PLUMED atom map against the reordering v1's notebook documents for trp-cage.

---

## Caveats

- The synthetic demo system (`asmsa_v2.demo`) is a **geometric mock**, not a force-field
  simulation. Bond lengths are plausible and it exercises every stage, but no physical conclusion
  may be drawn from it.
- A 2-D latent is the default because metadynamics must grid the CV space and because you can look
  at it. Higher `latent_dim` works, but the sampling cost grows steeply.
- `06-md`'s execution path requires GROMACS and PLUMED and has not been exercised in an
  environment where they are installed.

## License

MIT.
