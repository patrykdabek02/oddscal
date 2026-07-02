import numpy as np
import pytest

from oddscal import implied_probabilities, overround


ODDS_1X2 = [1.30, 5.50, 11.00]  # heavy favourite, ~5% margin


@pytest.mark.parametrize("method", ["multiplicative", "shin", "power"])
def test_probabilities_sum_to_one(method):
    p = implied_probabilities(ODDS_1X2, method=method)
    assert p.shape == (3,)
    assert np.all(p > 0)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)


def test_overround_positive():
    assert overround(ODDS_1X2) > 1.0
    assert overround([2.0, 2.0]) == pytest.approx(1.0)


def test_shin_corrects_favourite_longshot_bias():
    mult = implied_probabilities(ODDS_1X2, method="multiplicative")
    shin = implied_probabilities(ODDS_1X2, method="shin")
    assert shin[0] > mult[0]      # favourite gets MORE probability
    assert shin[-1] < mult[-1]    # longshot gets LESS


def test_two_outcome_market():
    p = implied_probabilities([1.90, 1.90], method="shin")
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert p[0] == pytest.approx(p[1], abs=1e-9)


@pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [2.0], [2.0, -3.0, 4.0]])
def test_invalid_odds_raise(bad):
    with pytest.raises(ValueError):
        implied_probabilities(bad)
