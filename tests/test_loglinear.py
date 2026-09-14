"""Log-linear extrapolation of sigma, and when it refuses to be used.

The last development period of a square triangle has one link ratio, so its
sigma cannot be estimated. Mack's minimum rule always produces a number;
extrapolating the decay of log(sigma) is the better-motivated alternative when
the decay is real, which is exactly what the significance test decides.
"""

from pathlib import Path

import numpy as np
import pytest

from mackpy import mack_chain_ladder, read_triangle_csv
from mackpy.mack import _extrapolate_sigma_loglinear, _loglinear_fit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raa():
    return read_triangle_csv(ROOT / "data" / "raa.csv").to_numpy()


def test_default_is_mack(raa):
    assert mack_chain_ladder(raa).est_sigma == "mack"
    assert mack_chain_ladder(raa).sigma_source[-1] == "mack"


def test_raa_decay_is_significant_so_extrapolation_is_used(raa):
    res = mack_chain_ladder(raa, est_sigma="log-linear")
    assert res.sigma_source[-1] == "log-linear"
    assert res.sigma_source[:-1] == ("data",) * 8


def test_extrapolated_sigma_differs_from_the_minimum_rule(raa):
    a = mack_chain_ladder(raa, est_sigma="mack")
    b = mack_chain_ladder(raa, est_sigma="log-linear")
    assert a.sigma[-1] != pytest.approx(b.sigma[-1])
    # only the last period can differ
    np.testing.assert_allclose(a.sigma[:-1], b.sigma[:-1], rtol=1e-12)


def test_the_choice_moves_every_origin_that_is_not_run_off(raa):
    """sigma of the last development period is on every origin's path.

    Only the oldest origin period, already fully developed, is untouched: every
    other one must still be projected through that final step, so a different
    sigma there moves its standard error. The factors themselves do not change,
    so the reserves are identical.
    """
    a = mack_chain_ladder(raa, est_sigma="mack")
    b = mack_chain_ladder(raa, est_sigma="log-linear")
    np.testing.assert_allclose(a.ibnr, b.ibnr, rtol=1e-12)
    assert a.mack_se[0] == b.mack_se[0] == 0.0
    assert np.all(a.mack_se[1:] != b.mack_se[1:])
    # the extrapolated sigma is the smaller one here, so the errors shrink
    assert np.all(b.mack_se[1:] < a.mack_se[1:])
    assert b.total_mack_se < a.total_mack_se


def test_fit_matches_least_squares_computed_independently():
    sigma2 = np.array([100.0, 60.0, 36.0, 20.0, 12.0, np.nan])
    a, b, p, df = _loglinear_fit(sigma2)

    x = np.arange(5, dtype=float)
    y = np.log(np.sqrt(sigma2[:5]))
    slope, intercept = np.polyfit(x, y, 1)
    assert b == pytest.approx(slope, rel=1e-10)
    assert a == pytest.approx(intercept, rel=1e-10)
    assert df == 3
    assert 0.0 <= p <= 1.0


def test_a_clean_decay_is_extrapolated():
    sigma2 = np.array([100.0, 60.0, 36.0, 20.0, 12.0, np.nan])
    out, source = _extrapolate_sigma_loglinear(sigma2)
    assert source[-1] == "log-linear"
    assert 0.0 < out[-1] < sigma2[4]


def test_a_flat_sequence_falls_back_to_mack():
    """No decay means no slope to extrapolate, so Mack's rule takes over."""
    sigma2 = np.array([50.0, 51.0, 49.5, 50.5, 50.0, np.nan])
    out, source = _extrapolate_sigma_loglinear(sigma2)
    assert source[-1] == "mack-fallback"
    assert out[-1] == pytest.approx(min(50.0**2 / 50.5, 50.5, 50.0))


def test_an_increasing_sequence_falls_back_to_mack():
    sigma2 = np.array([10.0, 20.0, 35.0, 60.0, 90.0, np.nan])
    out, source = _extrapolate_sigma_loglinear(sigma2)
    assert source[-1] == "mack-fallback"


def test_too_few_points_to_test_a_slope_falls_back():
    sigma2 = np.array([100.0, 40.0, np.nan])
    assert _loglinear_fit(sigma2) is None
    out, source = _extrapolate_sigma_loglinear(sigma2)
    assert source[-1] == "mack-fallback"
    assert np.isfinite(out[-1])


def test_nothing_missing_means_nothing_to_decide():
    sigma2 = np.array([100.0, 60.0, 36.0])
    out, source = _extrapolate_sigma_loglinear(sigma2)
    np.testing.assert_allclose(out, sigma2, rtol=0)
    assert source == ("data", "data", "data")


def test_both_spellings_are_accepted(raa):
    a = mack_chain_ladder(raa, est_sigma="log-linear")
    b = mack_chain_ladder(raa, est_sigma="loglinear")
    assert a.sigma[-1] == b.sigma[-1]


@pytest.mark.parametrize("n", [5, 8, 12])
def test_log_linear_keeps_the_mse_decomposition(n):
    from tests.test_properties import make_triangle

    res = mack_chain_ladder(make_triangle(n), est_sigma="log-linear")
    np.testing.assert_allclose(
        res.mack_se**2, res.process_risk**2 + res.parameter_risk**2, rtol=1e-10
    )
