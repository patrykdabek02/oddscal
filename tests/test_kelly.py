import pytest

from oddscal import kelly_fraction, log_growth


def test_textbook_value():
    # p=0.6 at even odds (b=1): f* = (0.6*1 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_no_edge_returns_zero():
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.3, 2.0) == 0.0


def test_fraction_and_cap():
    assert kelly_fraction(0.6, 2.0, fraction=0.25) == pytest.approx(0.05)
    assert kelly_fraction(0.9, 5.0, cap=0.05) == 0.05


def test_log_growth_positive_iff_edge():
    assert log_growth(0.6, 2.0) > 0.0
    assert log_growth(0.4, 2.0) == 0.0


def test_degenerate_probabilities():
    assert kelly_fraction(0.0, 2.0) == 0.0
    assert kelly_fraction(1.0, 2.0) == 0.0
