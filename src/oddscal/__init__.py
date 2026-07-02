"""oddscal - de-vigging, probability calibration metrics and Kelly staking
for sports betting research.

Extracted from the experimental part of a bachelor's thesis on football
betting market efficiency (University of Economics in Katowice, 2026).
"""
from .devig import implied_probabilities, overround
from .kelly import kelly_fraction, log_growth
from .metrics import brier_score, expected_calibration_error, log_loss

__version__ = "0.1.0"
__all__ = [
    "implied_probabilities", "overround",
    "kelly_fraction", "log_growth",
    "brier_score", "log_loss", "expected_calibration_error",
]
