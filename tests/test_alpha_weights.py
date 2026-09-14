"""Mack's general variance assumption: the exponent alpha and the weights.

The three values of alpha correspond to three familiar estimators of the
development factor, and the tests below compute each of them directly rather
than trusting the implementation to agree with itself.
"""

from pathlib import Path

import numpy as np
import pytest

from mackpy import mack_chain_ladder, read_triangle_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raa():
    return read_triangle_csv(ROOT / "data" / "raa.csv").to_numpy()


def link_ratios(tri, k):
    rows = ~np.isnan(tri[:, k]) & ~np.isnan(tri[:, k + 1])
    return tri[rows, k], tri[rows, k + 1]


def test_alpha_zero_is_the_simple_average_of_link_ratios(raa):
    res = mack_chain_ladder(raa, alpha=0)
    for k in range(raa.shape[1] - 1):
        c_k, c_next = link_ratios(raa, k)
        assert res.f[k] == pytest.approx(float(np.mean(c_next / c_k)), rel=1e-12)


def test_alpha_one_is_volume_weighted(raa):
    res = mack_chain_ladder(raa, alpha=1)
    for k in range(raa.shape[1] - 1):
        c_k, c_next = link_ratios(raa, k)
        assert res.f[k] == pytest.approx(float(c_next.sum() / c_k.sum()), rel=1e-12)


def test_alpha_two_is_the_regression_estimator(raa):
    """alpha = 2 weights each ratio by C^2, i.e. ordinary least squares."""
    res = mack_chain_ladder(raa, alpha=2)
    for k in range(raa.shape[1] - 1):
        c_k, c_next = link_ratios(raa, k)
        expected = float(np.sum(c_k * c_next) / np.sum(c_k**2))
        assert res.f[k] == pytest.approx(expected, rel=1e-12)


def test_alpha_one_is_the_default(raa):
    a = mack_chain_ladder(raa)
    b = mack_chain_ladder(raa, alpha=1)
    np.testing.assert_allclose(a.f, b.f, rtol=0)
    np.testing.assert_allclose(a.mack_se, b.mack_se, rtol=0)


def test_the_three_alphas_differ(raa):
    fs = [mack_chain_ladder(raa, alpha=a).f for a in (0, 1, 2)]
    assert not np.allclose(fs[0], fs[1])
    assert not np.allclose(fs[1], fs[2])


def test_sigma_scales_with_the_weighting(raa):
    """sigma_k^2 is the weighted residual variance, so it must follow alpha."""
    for alpha in (0, 1, 2):
        res = mack_chain_ladder(raa, alpha=alpha)
        for k in range(raa.shape[1] - 2):
            c_k, c_next = link_ratios(raa, k)
            weight = c_k**alpha
            expected = float(
                np.sum(weight * (c_next / c_k - res.f[k]) ** 2) / (len(c_k) - 1)
            )
            assert res.sigma[k] ** 2 == pytest.approx(expected, rel=1e-10)


def test_unit_weights_change_nothing(raa):
    a = mack_chain_ladder(raa)
    b = mack_chain_ladder(raa, weights=np.ones_like(raa))
    np.testing.assert_allclose(a.f, b.f, rtol=0)
    np.testing.assert_allclose(a.mack_se, b.mack_se, rtol=0)
    assert a.total_mack_se == b.total_mack_se


def test_zero_weight_drops_exactly_one_ratio(raa):
    """Zeroing w[i, k] must give the factor computed without that row."""
    w = np.ones_like(raa)
    w[3, 0] = 0.0
    res = mack_chain_ladder(raa, weights=w)

    rows = ~np.isnan(raa[:, 0]) & ~np.isnan(raa[:, 1])
    rows[3] = False
    expected = float(raa[rows, 1].sum() / raa[rows, 0].sum())
    assert res.f[0] == pytest.approx(expected, rel=1e-12)

    # every later period is untouched
    plain = mack_chain_ladder(raa)
    np.testing.assert_allclose(res.f[1:], plain.f[1:], rtol=1e-12)


def test_zero_weight_lowers_the_observation_count(raa):
    w = np.ones_like(raa)
    w[:, 0] = 0.0
    w[0, 0] = 1.0
    w[1, 0] = 1.0
    res = mack_chain_ladder(raa, weights=w)
    assert res.n_obs[0] == 2


def test_weight_of_a_single_ratio_forces_sigma_extrapolation(raa):
    """One usable ratio leaves sigma_0 unestimable, so it is extrapolated."""
    w = np.ones_like(raa)
    w[1:, 0] = 0.0
    res = mack_chain_ladder(raa, weights=w)
    assert res.sigma_source[0] == "mack"
    assert np.isfinite(res.sigma[0])


def test_weights_reject_wrong_shape_and_sign(raa):
    with pytest.raises(ValueError, match="shape"):
        mack_chain_ladder(raa, weights=np.ones((3, 3)))
    w = np.ones_like(raa)
    w[2, 2] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        mack_chain_ladder(raa, weights=w)


def test_rejects_unsupported_alpha(raa):
    with pytest.raises(ValueError, match="alpha"):
        mack_chain_ladder(raa, alpha=3)


def test_a_whole_period_without_usable_ratios_is_an_error(raa):
    w = np.ones_like(raa)
    w[:, 0] = 0.0
    with pytest.raises(ValueError, match="usable link ratio"):
        mack_chain_ladder(raa, weights=w)
