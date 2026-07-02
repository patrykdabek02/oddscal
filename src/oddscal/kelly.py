"""Kelly criterion stake sizing."""
from __future__ import annotations

import numpy as np

__all__ = ["kelly_fraction", "log_growth"]


def kelly_fraction(p: float, odds: float, fraction: float = 1.0, cap: float | None = None) -> float:
    """Optimal fraction of bankroll for a bet at decimal ``odds`` with
    estimated win probability ``p``.

    ``fraction`` scales the full-Kelly stake (0.25 = quarter Kelly, the
    common practical choice given noisy probability estimates). ``cap``
    optionally limits the stake (e.g. 0.05 = max 5% of bankroll).
    Returns 0.0 when the edge is non-positive.
    """
    if not 0.0 < p < 1.0:
        return 0.0
    b = odds - 1.0
    if b <= 0.0:
        return 0.0
    f = max(0.0, (p * b - (1.0 - p)) / b) * fraction
    if cap is not None:
        f = min(f, cap)
    return float(f)


def log_growth(p: float, odds: float, fraction: float = 1.0, cap: float | None = None) -> float:
    """Expected logarithmic growth of bankroll per bet at the (scaled)
    Kelly stake. Useful for ranking candidate bets on the same match."""
    f = kelly_fraction(p, odds, fraction, cap)
    if f <= 0.0:
        return 0.0
    b = odds - 1.0
    return float(p * np.log1p(f * b) + (1.0 - p) * np.log1p(-f))
