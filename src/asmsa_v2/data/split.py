"""Train / validation / test splitting for MD trajectories.

Consecutive MD frames are highly correlated, so a naive random split leaks
information: a validation frame 2 fs away from a training frame is, for all
practical purposes, the same structure. The validation loss then measures
memorization rather than generalization.

Three strategies are offered:

``chronological``
    Contiguous blocks in trajectory order (train | val | test). Correlation
    leakage is confined to the two block boundaries, so this is the
    recommended default and the standard choice in the kinetic-model
    literature.
``strided``
    Interleaved by stride. Each split covers the whole trajectory, which
    helps when the simulation has a slow systematic drift, but neighbouring
    frames land in different splits so the leakage is maximal. Use only with
    ``decorrelate_rmsd``.
``random``
    Shuffle then slice, reproducing v1's behaviour. Kept for comparability
    with published v1 results; same leakage caveat as ``strided``.

:func:`decorrelate_by_rmsd` additionally reproduces v1's filter that drops
training frames lying within an RMSD cutoff of any validation/test frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitIndices:
    """Frame indices for each split, as indices into the original trajectory."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str

    def __post_init__(self):
        self.train = np.asarray(self.train, dtype=np.int64)
        self.val = np.asarray(self.val, dtype=np.int64)
        self.test = np.asarray(self.test, dtype=np.int64)

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "val": self.val.size, "test": self.test.size}

    def summary(self) -> str:
        tot = sum(self.sizes.values()) or 1
        parts = ", ".join(
            f"{k}={v} ({v / tot * 100:.1f}%)" for k, v in self.sizes.items()
        )
        return f"[{self.strategy}] {tot} frames -> {parts}"

    def _repr_html_(self) -> str:
        tot = sum(self.sizes.values()) or 1
        rows = "".join(
            f"<tr><td style='padding:2px 10px'><b>{k}</b></td>"
            f"<td style='padding:2px 10px;text-align:right'>{v}</td>"
            f"<td style='padding:2px 10px;text-align:right'>{v / tot * 100:.1f}%</td></tr>"
            for k, v in self.sizes.items()
        )
        return (
            f"<table><caption style='text-align:left'><b>Split</b> "
            f"(strategy: <code>{self.strategy}</code>, {tot} frames)</caption>"
            f"<tr><th>split</th><th>frames</th><th>share</th></tr>{rows}</table>"
        )

    def __repr__(self) -> str:
        return f"SplitIndices({self.summary()})"


def split_frames(
    n_frames: int,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    strategy: str = "chronological",
    seed: int | None = 0,
) -> SplitIndices:
    """Split ``n_frames`` into train/val/test index arrays.

    Fractions are normalized if they do not sum to 1, so ``0.7/0.15/0.15``
    and ``7/1.5/1.5`` behave identically and the brittle float equality
    assertion from v1's notebook is unnecessary.
    """
    if n_frames < 3:
        raise ValueError(f"need at least 3 frames to split, got {n_frames}")
    fracs = np.array([train, val, test], dtype=np.float64)
    if np.any(fracs < 0):
        raise ValueError("split fractions must be non-negative")
    if fracs.sum() <= 0:
        raise ValueError("split fractions must not all be zero")
    fracs = fracs / fracs.sum()

    n_train = int(round(n_frames * fracs[0]))
    n_val = int(round(n_frames * fracs[1]))
    # give any rounding remainder to test so the splits always tile exactly
    n_train = min(n_train, n_frames)
    n_val = min(n_val, n_frames - n_train)

    if strategy == "chronological":
        order = np.arange(n_frames)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_frames)
    elif strategy == "strided":
        # round-robin deal: 0->train, 1->train, ... proportional to fractions
        order = np.empty(n_frames, dtype=np.int64)
        counts = np.array([n_train, n_val, n_frames - n_train - n_val])
        cursor = np.concatenate([[0], np.cumsum(counts)])
        pos = cursor[:3].copy()
        # deal frames to the split whose fill ratio is currently lowest
        filled = np.zeros(3)
        for i in range(n_frames):
            ratios = np.where(counts > 0, filled / np.maximum(counts, 1), np.inf)
            k = int(np.argmin(ratios))
            order[pos[k]] = i
            pos[k] += 1
            filled[k] += 1
    else:
        raise ValueError(
            f"unknown strategy {strategy!r}; use 'chronological', 'strided' or 'random'"
        )

    return SplitIndices(
        train=np.sort(order[:n_train]),
        val=np.sort(order[n_train:n_train + n_val]),
        test=np.sort(order[n_train + n_val:]),
        strategy=strategy,
    )


def decorrelate_by_rmsd(
    split: SplitIndices,
    traj,
    cutoff_nm: float = 0.05,
    atom_indices=None,
    against: tuple[str, ...] = ("val", "test"),
) -> SplitIndices:
    """Drop training frames within ``cutoff_nm`` RMSD of any held-out frame.

    This is v1's dataset-bias filter, reimplemented on mdtraj so it no longer
    needs GROMACS ``gmx rms`` and XPM matrix parsing. ``traj`` must be an
    mdtraj Trajectory whose frame indexing matches ``split``.

    Only the training split shrinks; validation and test are untouched.
    """
    import mdtraj as md

    if atom_indices is None:
        atom_indices = traj.top.select("name CA")
        if len(atom_indices) < 3:
            atom_indices = traj.top.select("element != H")
        if len(atom_indices) == 0:
            atom_indices = np.arange(traj.n_atoms)
    atom_indices = np.asarray(atom_indices, dtype=np.int64)

    held = np.concatenate(
        [getattr(split, name) for name in against if getattr(split, name).size]
    ) if against else np.array([], dtype=np.int64)
    if held.size == 0 or split.train.size == 0:
        return split

    held_traj = traj[held]
    keep = np.ones(split.train.size, dtype=bool)
    # md.rmsd gives distances from every held-out frame to one reference frame,
    # so loop over training frames and take the minimum over held-out frames.
    for i, fi in enumerate(split.train):
        d = md.rmsd(held_traj, traj, frame=int(fi), atom_indices=atom_indices)
        if d.min() <= cutoff_nm:
            keep[i] = False

    return SplitIndices(
        train=split.train[keep],
        val=split.val,
        test=split.test,
        strategy=f"{split.strategy}+rmsd_decorrelated({cutoff_nm}nm)",
    )


def lag_pairs(n_frames: int, lag: int, valid_indices=None) -> np.ndarray:
    """Return ``[n_pairs, 2]`` index pairs ``(t, t+lag)`` for time-lagged training.

    If ``valid_indices`` is given (e.g. a split's train indices), only pairs
    whose *both* endpoints are in that set are returned, so a time-lagged
    model never sees a validation frame during training.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if lag >= n_frames:
        raise ValueError(f"lag {lag} >= n_frames {n_frames}")
    t = np.arange(n_frames - lag, dtype=np.int64)
    pairs = np.stack([t, t + lag], axis=1)
    if valid_indices is not None:
        allowed = np.zeros(n_frames, dtype=bool)
        allowed[np.asarray(valid_indices, dtype=np.int64)] = True
        pairs = pairs[allowed[pairs[:, 0]] & allowed[pairs[:, 1]]]
    return pairs
