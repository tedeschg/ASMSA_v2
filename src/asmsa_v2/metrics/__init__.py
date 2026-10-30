"""Latent-space metrics: KL divergence, VAMP-2 scoring, and quality reports."""

from .kl import kl_divergence_knn, prior_match_score
from .vamp import (
    implied_timescales,
    koopman_singular_values,
    vamp2_scan,
    vamp2_score,
)
from .quality import QualityReport, effective_dimension, evaluate_latent    

__all__ = [
    "QualityReport",
    "effective_dimension",
    "evaluate_latent",
    "implied_timescales",
    "kl_divergence_knn",
    "koopman_singular_values",
    "prior_match_score",
    "vamp2_scan",
    "vamp2_score",
]
