"""Pose-cluster analysis on a trained AAE's latent space.

Prefers HDBSCAN (density-based, no need to pick k, natively marks noise
frames as cluster id ``-1``). Falls back to sklearn KMeans with k chosen
by silhouette score if HDBSCAN isn't installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    hdbscan = None
    _HDBSCAN_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None
    silhouette_score = None
    _SKLEARN_AVAILABLE = False


def _to_np(z) -> np.ndarray:
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    return np.asarray(z)


@dataclass
class ClusterResult:
    labels: np.ndarray                      # per-frame cluster id (``-1`` = noise)
    cluster_ids: list[int]                  # sorted non-noise ids
    centroids: dict[int, np.ndarray]        # id -> 2-D centroid in latent
    representatives: dict[int, int]         # id -> frame index closest to centroid
    sizes: dict[int, int]                   # id -> number of frames
    method: str

    def summary(self) -> str:
        lines = [f"[{self.method}] {len(self.cluster_ids)} cluster(s)"]
        for cid in self.cluster_ids:
            lines.append(
                f"  cluster {cid:>3}: {self.sizes[cid]:>5} frames, "
                f"rep=frame {self.representatives[cid]}, "
                f"centroid=({self.centroids[cid][0]:+.2f}, {self.centroids[cid][1]:+.2f})"
            )
        noise = int(np.sum(self.labels == -1))
        if noise:
            lines.append(f"  noise:        {noise:>5} frames")
        return "\n".join(lines)


def cluster_latent(
    z,
    min_cluster_size: int = 20,
    method: str | None = None,
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    seed: int = 0,
) -> ClusterResult:
    """Cluster latent points into pose groups.

    Parameters
    ----------
    z : array ``[n_frames, latent_dim]``
    min_cluster_size : int
        Only used for HDBSCAN. Smaller → more, smaller clusters.
    method : ``"hdbscan"``, ``"kmeans"``, or ``None`` (auto: HDBSCAN if available).
    k_range : sequence of int
        Candidate k values when using KMeans; the best silhouette wins.
    """
    z = _to_np(z)
    if z.ndim != 2:
        raise ValueError(f"expected 2-D z, got shape {z.shape}")

    if method is None:
        method = "hdbscan" if _HDBSCAN_AVAILABLE else "kmeans"
    if method == "hdbscan":
        if not _HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan not installed; `pip install hdbscan` or pass method='kmeans'")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size))
        labels = clusterer.fit_predict(z)
    elif method == "kmeans":
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed; `pip install scikit-learn`")
        best_score, best_labels, best_k = -1.0, None, None
        for k in k_range:
            if k >= z.shape[0]:
                continue
            km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(z)
            s = silhouette_score(z, km.labels_) if k > 1 else -1.0
            if s > best_score:
                best_score, best_labels, best_k = s, km.labels_, k
        if best_labels is None:
            best_labels = np.zeros(z.shape[0], dtype=int)
        labels = best_labels
        method = f"kmeans(k={best_k}, silhouette={best_score:.3f})"
    else:
        raise ValueError(f"unknown method {method!r}; use 'hdbscan' or 'kmeans'")

    cluster_ids = sorted(int(c) for c in np.unique(labels) if c != -1)
    centroids: dict[int, np.ndarray] = {}
    representatives: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for cid in cluster_ids:
        mask = labels == cid
        pts = z[mask]
        centroid = pts.mean(axis=0)
        d = np.linalg.norm(pts - centroid, axis=1)
        local_rep = int(np.argmin(d))
        global_rep = int(np.where(mask)[0][local_rep])
        centroids[cid] = centroid
        representatives[cid] = global_rep
        sizes[cid] = int(mask.sum())

    return ClusterResult(
        labels=np.asarray(labels),
        cluster_ids=cluster_ids,
        centroids=centroids,
        representatives=representatives,
        sizes=sizes,
        method=method,
    )
