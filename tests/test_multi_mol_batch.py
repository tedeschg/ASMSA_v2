import torch
from torch.utils.data import DataLoader

from asmsa_v2 import MultiMolDataset, multi_mol_collate


def _make_ds():
    return MultiMolDataset({
        "molA": torch.randn(500, 20),
        "molB": torch.randn(300, 35),
    })


def test_batch_is_single_molecule():
    ds = _make_ds()
    loader = DataLoader(ds, batch_sampler=ds.round_robin_sampler(batch_size=64),
                        collate_fn=multi_mol_collate)
    seen = set()
    n = 0
    for x, mol_id, extras in loader:
        assert isinstance(mol_id, str)
        assert x.ndim == 2
        assert x.shape[1] == ds.feature_dims[mol_id]
        assert extras == {}
        seen.add(mol_id)
        n += 1
    assert seen == {"molA", "molB"}
    # len() must match the number of batches actually produced
    assert n == len(loader)


def test_round_robin_cycles():
    ds = _make_ds()
    sampler = ds.round_robin_sampler(batch_size=64, shuffle=False)
    batches = list(sampler)
    # First two batches should be from different molecules
    assert batches[0] != batches[1]
    # Decode mol-id by range membership
    def mol_of(ix):
        for i, m in enumerate(ds.mol_ids):
            lo = int(ds._offsets[i])
            hi = int(ds._offsets[i + 1])
            if lo <= ix < hi:
                return m
        raise AssertionError
    assert mol_of(batches[0][0]) != mol_of(batches[1][0])
