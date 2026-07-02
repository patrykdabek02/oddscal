import numpy as np
import pytest

from oddscal import brier_score, expected_calibration_error, log_loss


def test_perfect_forecast_scores_zero():
    y = np.array([0, 1, 2, 1])
    p = np.eye(3)[y]
    assert brier_score(y, p) == pytest.approx(0.0)
    assert log_loss(y, p) == pytest.approx(0.0, abs=1e-10)


def test_uniform_forecast_three_classes():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, size=30_000)
    p = np.full((len(y), 3), 1 / 3)
    assert brier_score(y, p) == pytest.approx(0.2222, abs=1e-3)
    assert log_loss(y, p) == pytest.approx(np.log(3), abs=1e-9)


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.05, 0.95, size=50_000)
    y = (rng.uniform(size=len(conf)) < conf).astype(int)
    p = np.column_stack([1 - conf, conf])
    assert expected_calibration_error(y, p, class_index=1) < 0.02


def test_ece_overconfident_is_large():
    y = np.zeros(1000, dtype=int)
    p = np.column_stack([np.full(1000, 0.1), np.full(1000, 0.9)])
    assert expected_calibration_error(y, p, class_index=1) > 0.8
