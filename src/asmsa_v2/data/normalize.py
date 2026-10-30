"""Feature standardization.

The featurizer emits blocks on wildly different scales: covalent bond lengths
have a standard deviation of ~0.007 nm while inter-component distances in a
protein-ligand complex span several nm (a ratio of ~200x on a realistic
complex). An unweighted MSE reconstruction loss is therefore dominated almost
entirely by the widest block, and the narrow blocks become invisible to the
autoencoder — they contribute ~0.001% of the total variance.

v1 standardized features with a train-set mean/scale before training and
folded the same transform into the TorchScript model handed to PLUMED. This
module restores that, as a reusable object that can

- fit on the training split only (no leakage into validation/test),
- round-trip to disk so later pipeline stages reuse the exact same transform,
- be exported as a ``torch.nn.Module`` for inclusion in a traced model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_EPS = 1e-8


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


class FeatureScaler:
    """Per-feature affine standardization ``(x - center) / scale``.

    Parameters
    ----------
    method : {"standard", "robust", "none"}
        ``"standard"`` uses mean / standard deviation (v1 behaviour).
        ``"robust"`` uses median / inter-quartile range, which is preferable
        when a few frames contain unfolded or dissociated outlier geometries
        that would otherwise inflate the scale of every feature.
        ``"none"`` is a pass-through, useful for ablations.
    clip_scale : float
        Features whose scale falls below this are treated as constant and
        given ``scale = 1`` so they pass through as a pure offset instead of
        being amplified into numerical noise.
    """

    def __init__(self, method: str = "standard", clip_scale: float = 1e-6):
        if method not in ("standard", "robust", "none"):
            raise ValueError(
                f"unknown method {method!r}; use 'standard', 'robust' or 'none'"
            )
        self.method = method
        self.clip_scale = float(clip_scale)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_constant_: int = 0

    # ---------------------------------------------------------------- fitting
    def fit(self, X) -> "FeatureScaler":
        """Fit on ``X`` of shape ``[n_frames, n_features]`` (train split only)."""
        Xn = _as_np(X)
        if Xn.ndim != 2:
            raise ValueError(f"expected 2-D [n_frames, n_features], got {Xn.shape}")
        if Xn.shape[0] < 2:
            raise ValueError("need at least 2 frames to estimate a scale")

        if self.method == "none":
            center = np.zeros(Xn.shape[1])
            scale = np.ones(Xn.shape[1])
        elif self.method == "standard":
            center = Xn.mean(axis=0)
            scale = Xn.std(axis=0)
        else:  # robust
            center = np.median(Xn, axis=0)
            q75, q25 = np.percentile(Xn, [75, 25], axis=0)
            scale = (q75 - q25) / 1.349  # IQR -> sigma for a normal distribution

        constant = scale < self.clip_scale
        self.n_constant_ = int(constant.sum())
        scale = np.where(constant, 1.0, scale)

        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def _check_fitted(self):
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler is not fitted; call .fit(X_train) first")

    # ------------------------------------------------------------- transforms
    def transform(self, X):
        """Standardize ``X``. Returns the same type (torch or numpy) it got."""
        self._check_fitted()
        if isinstance(X, torch.Tensor):
            c = torch.as_tensor(self.center_, dtype=X.dtype, device=X.device)
            s = torch.as_tensor(self.scale_, dtype=X.dtype, device=X.device)
            return (X - c) / s
        Xn = np.asarray(X, dtype=np.float32)
        return (Xn - self.center_) / self.scale_

    def inverse_transform(self, Z):
        """Map standardized features back to physical units."""
        self._check_fitted()
        if isinstance(Z, torch.Tensor):
            c = torch.as_tensor(self.center_, dtype=Z.dtype, device=Z.device)
            s = torch.as_tensor(self.scale_, dtype=Z.dtype, device=Z.device)
            return Z * s + c
        Zn = np.asarray(Z, dtype=np.float32)
        return Zn * self.scale_ + self.center_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> Path:
        """Persist to a ``.npz``. Later stages must reuse this exact transform."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            center=self.center_,
            scale=self.scale_,
            method=np.array(self.method),
            clip_scale=np.array(self.clip_scale),
            n_constant=np.array(self.n_constant_),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        d = np.load(Path(path), allow_pickle=False)
        obj = cls(method=str(d["method"]), clip_scale=float(d["clip_scale"]))
        obj.center_ = d["center"].astype(np.float32)
        obj.scale_ = d["scale"].astype(np.float32)
        obj.n_constant_ = int(d["n_constant"])
        return obj

    # -------------------------------------------------------------- reporting
    @property
    def n_features(self) -> int:
        self._check_fitted()
        return int(self.center_.shape[0])

    def as_module(self) -> "StandardizeModule":
        """Return a ``nn.Module`` applying this transform, for TorchScript export."""
        self._check_fitted()
        return StandardizeModule(self.center_, self.scale_)

    def report(self, X=None, feature_blocks: dict[str, tuple[int, int]] | None = None) -> str:
        """Human-readable summary; pass ``X`` to include the pre-scaling spread."""
        self._check_fitted()
        lines = [
            f"FeatureScaler(method={self.method!r}, n_features={self.n_features})",
            f"  constant features clamped to scale=1: {self.n_constant_}",
        ]
        if X is not None:
            Xn = _as_np(X)
            sd = Xn.std(axis=0)
            nz = sd[sd > self.clip_scale]
            if nz.size:
                lines.append(
                    f"  raw per-feature std: min={nz.min():.5g} max={nz.max():.5g} "
                    f"(ratio {nz.max() / nz.min():.0f}x)"
                )
            if feature_blocks:
                var_total = float((sd ** 2).sum()) or 1.0
                lines.append("  raw variance share by block:")
                for name, (a, b) in feature_blocks.items():
                    share = float((sd[a:b] ** 2).sum()) / var_total * 100.0
                    lines.append(
                        f"    {name:<16} {share:7.3f}%   "
                        f"(std {sd[a:b].mean():.5g} avg over {b - a} features)"
                    )
        return "\n".join(lines)

    def __repr__(self) -> str:
        n = self.n_features if self.center_ is not None else "unfitted"
        return f"FeatureScaler(method={self.method!r}, n_features={n})"


class StandardizeModule(nn.Module):
    """``(x - center) / scale`` as a traceable module with registered buffers.

    Registered as buffers (not plain attributes) so ``.to(device)`` moves them
    and ``torch.jit.trace`` captures them as constants of the saved model.
    """

    def __init__(self, center, scale):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center) / self.scale
