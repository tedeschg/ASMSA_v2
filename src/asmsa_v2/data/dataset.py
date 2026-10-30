"""Dataset / sampler / collate for joint multi-molecule AAE training.

Each training batch contains frames from exactly one molecule — the forward
pass through the per-molecule adapter stays simple (no padding, no masking).
Molecules are served in round-robin order; smaller trajectories wrap around.
"""
from typing import Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MultiMolDataset(Dataset):
    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        if not tensors:
            raise ValueError("MultiMolDataset needs at least one molecule tensor")
        self.mol_ids: list[str] = list(tensors.keys())
        self._tensors: dict[str, torch.Tensor] = {}
        for mol_id, t in tensors.items():
            t = torch.as_tensor(t, dtype=torch.float32)
            if t.ndim != 2:
                raise ValueError(f"tensor for {mol_id!r} must be 2-D [n_frames, n_features], "
                                 f"got shape {tuple(t.shape)}")
            self._tensors[mol_id] = t

        self.feature_dims: dict[str, int] = {m: t.shape[1] for m, t in self._tensors.items()}

        # flat global index -> (mol_id, local_index)
        self._lengths = [self._tensors[m].shape[0] for m in self.mol_ids]
        self._offsets = np.cumsum([0] + self._lengths)
        self._N = int(self._offsets[-1])

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        mi = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local = idx - int(self._offsets[mi])
        mol_id = self.mol_ids[mi]
        return self._tensors[mol_id][local], mol_id

    def indices_for(self, mol_id: str) -> np.ndarray:
        mi = self.mol_ids.index(mol_id)
        start = int(self._offsets[mi])
        return np.arange(start, start + self._lengths[mi])

    def round_robin_sampler(self, batch_size: int, shuffle: bool = True,
                            drop_last: bool = True, generator: torch.Generator | None = None):
        return RoundRobinBatchSampler(
            self, batch_size=batch_size, shuffle=shuffle,
            drop_last=drop_last, generator=generator,
        )


class RoundRobinBatchSampler(Sampler[list[int]]):
    """Yields batches of indices where every index in a batch belongs to the
    same molecule. Molecules are iterated in round-robin order; an epoch is
    defined as one full pass through the largest molecule's trajectory.
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
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        if generator is None:
            self.generator.manual_seed(torch.initial_seed() & 0xFFFF_FFFF)

    def _mol_batches(self, mol_id: str) -> list[list[int]]:
        idx = self.dataset.indices_for(mol_id)
        if self.shuffle:
            perm = torch.randperm(len(idx), generator=self.generator).numpy()
            idx = idx[perm]
        batches = [idx[i:i + self.batch_size].tolist()
                   for i in range(0, len(idx), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        per_mol = {m: self._mol_batches(m) for m in self.dataset.mol_ids}
        if not any(per_mol.values()):
            return
        max_len = max(len(b) for b in per_mol.values())
        for step in range(max_len):
            for mol_id in self.dataset.mol_ids:
                batches = per_mol[mol_id]
                if not batches:
                    continue
                yield batches[step % len(batches)]

    def __len__(self) -> int:
        per_mol_lens = []
        for m in self.dataset.mol_ids:
            n = self.dataset._lengths[self.dataset.mol_ids.index(m)]
            k = n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
            per_mol_lens.append(k)
        return max(per_mol_lens) * len(self.dataset.mol_ids) if per_mol_lens else 0


def multi_mol_collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, str]:
    """Collate assumes all items in a batch share the same ``mol_id`` (enforced
    by :class:`RoundRobinBatchSampler`). Returns ``(stacked_x, mol_id)``.
    """
    xs, mol_ids = zip(*batch)
    mol_id = mol_ids[0]
    assert all(m == mol_id for m in mol_ids), "mixed-molecule batch — use RoundRobinBatchSampler"
    return torch.stack(xs, dim=0), mol_id
