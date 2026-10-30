"""Maximum-mean-discrepancy latent penalty (WAE-MMD).

Tolstikhin, Bousquet, Gelly & Schoelkopf, "Wasserstein Auto-Encoders",
ICLR 2018, replace the adversarial latent matching of an AAE with a direct
kernel two-sample penalty between the aggregate posterior q(z) and the prior
p(z). There is no discriminator, so there is no minimax game to balance, no
mode collapse, and no need to tune a second learning rate — which is a large
practical win for the 2-D latents used for collective variables, where the
sample size per batch is big relative to the latent dimension and the MMD
estimator is consequently accurate.

The inverse-multiquadratic kernel is used, as recommended in the paper: it
has much heavier tails than an RBF, so gradients do not vanish once the two
distributions are already close.
"""
from __future__ import annotations

import torch


def _imq_kernel(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """Inverse multiquadratic kernel ``C / (C + ||x - y||^2)``."""
    d2 = torch.cdist(x, y, p=2.0) ** 2
    return scale / (scale + d2)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    d2 = torch.cdist(x, y, p=2.0) ** 2
    return torch.exp(-d2 / (2.0 * scale))


_KERNELS = {"imq": _imq_kernel, "rbf": _rbf_kernel}


def mmd_loss(
    z_posterior: torch.Tensor,
    z_prior: torch.Tensor,
    kernel: str = "imq",
    scales: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    unbiased: bool = True,
) -> torch.Tensor:
    """Squared MMD between encoder output and prior samples.

    Parameters
    ----------
    z_posterior : ``[n, d]``
        Encoder outputs for a batch (the aggregate posterior sample).
    z_prior : ``[m, d]``
        Samples drawn from the prior.
    kernel : {"imq", "rbf"}
    scales : tuple of float
        Bandwidth multipliers. The base bandwidth is ``2 * d``, following the
        WAE reference implementation, and the penalty is summed over scales so
        the estimator is sensitive to mismatch at several length scales at
        once instead of depending on one hand-picked bandwidth.
    unbiased : bool
        Drop the diagonal self-similarity terms (the standard unbiased
        U-statistic estimator). With ``False`` the biased V-statistic is used,
        which is always non-negative and is sometimes preferred for stability
        at small batch sizes.

    Returns
    -------
    torch.Tensor
        Scalar. Zero iff the two samples are indistinguishable under the
        kernel; can be slightly negative with ``unbiased=True``.
    """
    if z_posterior.ndim != 2 or z_prior.ndim != 2:
        raise ValueError("mmd_loss expects 2-D [n, d] tensors")
    if z_posterior.shape[1] != z_prior.shape[1]:
        raise ValueError(
            f"latent dim mismatch: posterior {z_posterior.shape[1]} vs "
            f"prior {z_prior.shape[1]}"
        )
    if kernel not in _KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; use one of {list(_KERNELS)}")
    kfn = _KERNELS[kernel]

    n, d = z_posterior.shape
    m = z_prior.shape[0]
    if n < 2 or m < 2:
        return z_posterior.new_zeros(())

    base = 2.0 * d
    total = z_posterior.new_zeros(())
    for s in scales:
        scale = base * float(s)
        k_qq = kfn(z_posterior, z_posterior, scale)
        k_pp = kfn(z_prior, z_prior, scale)
        k_qp = kfn(z_posterior, z_prior, scale)

        if unbiased:
            # remove diagonal (self-similarity) terms
            qq = (k_qq.sum() - k_qq.diagonal().sum()) / (n * (n - 1))
            pp = (k_pp.sum() - k_pp.diagonal().sum()) / (m * (m - 1))
        else:
            qq = k_qq.mean()
            pp = k_pp.mean()
        qp = k_qp.mean()
        total = total + qq + pp - 2.0 * qp

    return total
