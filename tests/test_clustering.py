import numpy as np
import pytest

from asmsa_v2 import cluster_latent, ClusterResult


def _two_cluster_z(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(-2, -2), scale=0.3, size=(100, 2))
    b = rng.normal(loc=(+2, +2), scale=0.3, size=(100, 2))
    return np.vstack([a, b])


def test_kmeans_recovers_two_clusters():
    z = _two_cluster_z()
    r = cluster_latent(z, method="kmeans", k_range=(2, 3, 4))
    assert isinstance(r, ClusterResult)
    assert len(r.cluster_ids) == 2
    # Each cluster has ~100 frames
    for cid in r.cluster_ids:
        assert 80 <= r.sizes[cid] <= 120


def test_representatives_are_close_to_centroid():
    z = _two_cluster_z()
    r = cluster_latent(z, method="kmeans", k_range=(2,))
    for cid, rep in r.representatives.items():
        d = np.linalg.norm(z[rep] - r.centroids[cid])
        assert d < 1.0


def test_summary_renders():
    z = _two_cluster_z()
    r = cluster_latent(z, method="kmeans", k_range=(2,))
    s = r.summary()
    assert "cluster" in s.lower()
    assert str(len(r.cluster_ids)) in s


def test_invalid_method_raises():
    z = _two_cluster_z()
    with pytest.raises(ValueError, match="unknown method"):
        cluster_latent(z, method="dbscan_xyz")
