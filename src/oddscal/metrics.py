"""Probability quality metrics for multiclass probabilistic forecasts."""
from __future__ import annotations

import numpy as np

__all__ = ["brier_score", "log_loss", "expected_calibration_error"]


def _check(y_true, proba):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or len(y) != len(p):
        raise ValueError("proba must be (n_samples, n_classes) matching y_true")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    return y, p


def brier_score(y_true, proba) -> float:
    """Multiclass Brier score, averaged over classes (lower is better).

    For K equiprobable classes a random forecast scores (K-1)/K**2,
    e.g. 0.2222 for K=3.
    """
    y, p = _check(y_true, proba)
    k = p.shape[1]
    onehot = np.eye(k)[y]
    return float(np.mean((p - onehot) ** 2))  # mean over samples and classes


def log_loss(y_true, proba, eps: float = 1e-15) -> float:
    """Multiclass logarithmic loss (cross-entropy), natural log."""
    y, p = _check(y_true, proba)
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def expected_calibration_error(y_true, proba, class_index: int, n_bins: int = 10) -> float:
    """ECE for one class: mean |empirical frequency - mean predicted| over
    equal-frequency probability bins."""
    y, p = _check(y_true, proba)
    conf = p[:, class_index]
    hit = (y == class_index).astype(float)
    order = np.argsort(conf)
    conf, hit = conf[order], hit[order]
    bins = np.array_split(np.arange(len(conf)), n_bins)
    err, n = 0.0, len(conf)
    for b in bins:
        if len(b) == 0:
            continue
        err += len(b) / n * abs(hit[b].mean() - conf[b].mean())
    return float(err)
