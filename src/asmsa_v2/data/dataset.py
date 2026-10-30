"""Dataset / sampler / collate for joint multi-molecule AAE training.

Each training batch contains frames from exactly one molecule — the forward
pass through the per-molecule adapter stays simple (no padding, no masking).
Molecules are served in round-robin order; smaller trajectories wrap around.

Per-frame auxiliary targets (``extras``) ride along with the features so the
trainer can pick up a CV1 bias target or a time-lagged companion frame
without a separate data path.
"""
from __future__ import annotations

from typing import Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _as_2d(t, name: str) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=torch.float32)
    if t.ndim != 2:
        raise ValueError(
            f"tensor for {name!r} must be 2-D [n_frames, n_features], "
            f"got shape {tuple(t.shape)}"
        )
    return t


class MultiMolDataset(Dataset):
    """Frames of one or more molecules, each with its own feature dimension.

    Parameters
    ----------
    tensors : mapping ``mol_id -> [n_frames, n_features]``
    extras : optional mapping ``mol_id -> {key -> [n_frames, ...]}``
        Per-frame auxiliary tensors delivered alongside each batch. Recognised
        by the trainer: ``"cv1"`` (scalar bias target) and ``"x_lag"``
        (time-lagged target features, same width as ``tensors[mol_id]``).
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        extras: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    ):
        if not tensors:
            raise ValueError("MultiMolDataset needs at least one molecule tensor")
        self.mol_ids: list[str] = list(tensors.keys())
        self._tensors: dict[str, torch.Tensor] = {
            mol_id: _as_2d(t, mol_id) for mol_id, t in tensors.items()
        }

        self._extras: dict[str, dict[str, torch.Tensor]] = {}
        for mol_id in self.mol_ids:
            per_mol = dict((extras or {}).get(mol_id, {}))
            n = self._tensors[mol_id].shape[0]
            for key, val in per_mol.items():
                v = torch.as_tensor(val, dtype=torch.float32)
                if v.shape[0] != n:
                    raise ValueError(
                        f"extras[{mol_id!r}][{key!r}] has {v.shape[0]} rows but "
                        f"{mol_id!r} has {n} frames"
                    )
                per_mol[key] = v
            self._extras[mol_id] = per_mol

        self.feature_dims: dict[str, int] = {
            m: t.shape[1] for m, t in self._tensors.items()
        }

        # flat global index -> (mol_id, local_index)
        self._lengths = [self._tensors[m].shape[0] for m in self.mol_ids]
        self._offsets = np.cumsum([0] + self._lengths)
        self._N = int(self._offsets[-1])

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, dict[str, torch.Tensor]]:
        mi = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local = idx - int(self._offsets[mi])
        mol_id = self.mol_ids[mi]
        extras = {k: v[local] for k, v in self._extras[mol_id].items()}
        return self._tensors[mol_id][local], mol_id, extras

    def tensor(self, mol_id: str) -> torch.Tensor:
        """Full feature tensor for one molecule."""
        return self._tensors[mol_id]

    def tensors(self) -> dict[str, torch.Tensor]:
        """All feature tensors, suitable for ``Trainer.encode_all``."""
        return dict(self._tensors)

    def n_frames(self, mol_id: str) -> int:
        return int(self._tensors[mol_id].shape[0])

    def indices_for(self, mol_id: str) -> np.ndarray:
        mi = self.mol_ids.index(mol_id)
        start = int(self._offsets[mi])
        return np.arange(start, start + self._lengths[mi])

    def round_robin_sampler(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
    ):
        return RoundRobinBatchSampler(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )

    def loader(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
        **dataloader_kwargs,
    ):
        """Shorthand for the correct ``DataLoader`` wiring, for **training**.

        Getting the batch sampler and the collate function paired up correctly
        is easy to get wrong by hand, so the dataset offers it directly. For
        validation or test data use :meth:`eval_loader` instead.
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            self,
            batch_sampler=self.round_robin_sampler(
                batch_size=batch_size, shuffle=shuffle,
                drop_last=drop_last, generator=generator,
            ),
            collate_fn=multi_mol_collate,
            **dataloader_kwargs,
        )

    def eval_loader(self, batch_size: int = 512, **dataloader_kwargs):
        """Loader for validation / test data: no shuffling, no dropped frames.

        Both defaults matter. A shuffling validation loader makes the
        validation metric depend on the batch partition, so the same weights
        score differently on each pass and "best epoch" becomes partly noise.
        ``drop_last=True`` would silently discard up to ``batch_size - 1``
        frames, which on a small validation split is a large fraction of it.
        """
        return self.loader(batch_size=batch_size, shuffle=False,
                           drop_last=False, **dataloader_kwargs)

    def __repr__(self) -> str:
        dims = ", ".join(f"{m}: {self._tensors[m].shape[0]}x{self.feature_dims[m]}"
                         for m in self.mol_ids)
        return f"MultiMolDataset({dims})"


class RoundRobinBatchSampler(Sampler[list[int]]):
    """Yields batches of indices where every index in a batch belongs to the
    same molecule. Molecules are iterated in round-robin order; an epoch is
    defined as one full pass through the largest molecule's trajectory.

    Molecules that yield no batches at all (fewer frames than ``batch_size``
    with ``drop_last=True``) are excluded from both the iteration and the
    reported length, so ``len(sampler)`` always equals the number of batches
    actually produced.
    """

    def __init__(
        self,
        dataset: MultiMolDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        if generator is None:
            self.generator.manual_seed(torch.initial_seed() & 0xFFFF_FFFF)

        self._n_batches_per_mol = {
            m: self._count_batches(self.dataset.n_frames(m))
            for m in self.dataset.mol_ids
        }
        self._active = [m for m, k in self._n_batches_per_mol.items() if k > 0]
        if not self._active:
            raise ValueError(
                f"no molecule has at least batch_size={self.batch_size} frames "
                f"(smallest needed for drop_last={self.drop_last}); "
                f"frame counts: "
                f"{ {m: self.dataset.n_frames(m) for m in self.dataset.mol_ids} }"
            )

    def _count_batches(self, n: int) -> int:
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _mol_batches(self, mol_id: str) -> list[list[int]]:
        idx = self.dataset.indices_for(mol_id)
        if self.shuffle:
            perm = torch.randperm(len(idx), generator=self.generator).numpy()
            idx = idx[perm]
        batches = [
            idx[i:i + self.batch_size].tolist()
            for i in range(0, len(idx), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        per_mol = {m: self._mol_batches(m) for m in self._active}
        max_len = max(len(b) for b in per_mol.values())
        for step in range(max_len):
            for mol_id in self._active:
                batches = per_mol[mol_id]
                yield batches[step % len(batches)]

    def __len__(self) -> int:
        max_len = max(self._n_batches_per_mol[m] for m in self._active)
        return max_len * len(self._active)


def multi_mol_collate(batch):
    """Collate items sharing one ``mol_id`` into ``(X, mol_id, extras)``.

    The shared-``mol_id`` invariant is guaranteed by
    :class:`RoundRobinBatchSampler`; it is asserted here because a mixed batch
    would silently train the wrong adapter.
    """
    xs, mol_ids, extras = zip(*batch)
    mol_id = mol_ids[0]
    if not all(m == mol_id for m in mol_ids):
        raise ValueError(
            "mixed-molecule batch; use MultiMolDataset.loader() or "
            "RoundRobinBatchSampler as the DataLoader's batch_sampler"
        )
    keys = extras[0].keys() if extras else ()
    stacked_extras = {k: torch.stack([e[k] for e in extras], dim=0) for k in keys}
    return torch.stack(xs, dim=0), mol_id, stacked_extras


def make_time_lagged_dataset(
    tensors: Mapping[str, torch.Tensor],
    lag: int = 1,
    extras: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> MultiMolDataset:
    """Build a dataset whose reconstruction target is the frame ``lag`` steps ahead.

    A time-lagged autoencoder reconstructs :math:`x_{t+\\tau}` from
    :math:`x_t`, which makes the latent space encode the *slow* degrees of
    freedom rather than whatever happens to carry the most instantaneous
    variance (Wehmeyer & Noe, J. Chem. Phys. 148, 241703 (2018)). For
    enhanced-sampling collective variables that is usually what you want.

    Frames are truncated to ``n_frames - lag`` per molecule, and the lagged
    companion is attached as the ``"x_lag"`` extra. ``tensors`` must be in
    trajectory order (do not pre-shuffle).
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    base: dict[str, torch.Tensor] = {}
    new_extras: dict[str, dict[str, torch.Tensor]] = {}
    for mol_id, t in tensors.items():
        t = _as_2d(t, mol_id)
        if t.shape[0] <= lag:
            raise ValueError(
                f"{mol_id!r} has {t.shape[0]} frames, too few for lag={lag}"
            )
        base[mol_id] = t[:-lag]
        per_mol = {"x_lag": t[lag:]}
        for k, v in dict((extras or {}).get(mol_id, {})).items():
            per_mol[k] = torch.as_tensor(v, dtype=torch.float32)[:-lag]
        new_extras[mol_id] = per_mol
    return MultiMolDataset(base, extras=new_extras)
