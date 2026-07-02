"""Removing the bookmaker margin (de-vigging) from betting odds."""
from __future__ import annotations

import numpy as np

__all__ = ["implied_probabilities", "overround"]


def overround(odds) -> float:
    """Bookmaker overround: sum of raw inverse odds (1.0 = fair market)."""
    odds = np.asarray(odds, dtype=float)
    _validate(odds)
    return float(np.sum(1.0 / odds))


def implied_probabilities(odds, method: str = "shin"):
    """Convert decimal odds for one market into fair probabilities.

    Parameters
    ----------
    odds : array-like of shape (n_outcomes,)
        Decimal odds, each > 1.0. Works for any number of outcomes
        (2 for tennis, 3 for football 1X2, more for outrights).
    method : {"shin", "multiplicative", "power"}
        * ``multiplicative`` - proportional normalisation.
        * ``shin`` (default) - Shin (1993) model; corrects the
          favourite-longshot bias by attributing part of the margin
          to insider trading.
        * ``power`` - finds k such that sum((1/odds)**k) == 1.

    Returns
    -------
    numpy.ndarray of shape (n_outcomes,), summing to 1.0.
    """
    odds = np.asarray(odds, dtype=float)
    _validate(odds)
    r = 1.0 / odds
    if method == "multiplicative":
        return r / r.sum()
    if method == "power":
        return _power(r)
    if method == "shin":
        return _shin(r)
    raise ValueError(f"unknown method: {method!r}")


def _validate(odds) -> None:
    if odds.ndim != 1 or odds.size < 2:
        raise ValueError("odds must be a 1-D array with at least 2 outcomes")
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("all odds must be finite and greater than 1.0")


def _power(r, n_iter: int = 60):
    lo, hi = 1.0, 12.0
    for _ in range(n_iter):
        k = 0.5 * (lo + hi)
        if np.sum(r**k) > 1.0:
            lo = k
        else:
            hi = k
    p = r ** (0.5 * (lo + hi))
    return p / p.sum()


def _shin(r, n_iter: int = 80):
    B = r.sum()

    def probs(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / B) - z) / (2.0 * (1.0 - z))

    lo, hi = 1e-9, 0.5
    if probs(hi).sum() > 1.0:  # extreme margins - widen bracket
        hi = 0.999
    for _ in range(n_iter):
        z = 0.5 * (lo + hi)
        if probs(z).sum() > 1.0:
            lo = z
        else:
            hi = z
    p = probs(0.5 * (lo + hi))
    return p / p.sum()
