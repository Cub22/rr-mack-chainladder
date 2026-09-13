"""Properties that must hold whatever the input triangle.

These tests do not depend on any reference implementation.  They catch the
kind of mistake that a single worked example can hide: an index shifted by
one, a variance term attached to the wrong column, a formula that happens to
be right for a ten-by-ten triangle and wrong for a five-by-five one.
"""

from pathlib import Path

import numpy as np
import pytest

from mackpy import cum_to_incr, incr_to_cum, mack_chain_ladder, read_triangle_csv
from mackpy.mack import _extrapolate_sigma_mack

ROOT = Path(__file__).resolve().parents[1]


def make_triangle(n: int, seed: int = 0) -> np.ndarray:
    """A random but well-behaved cumulative triangle of size ``n``."""
    rng = np.random.default_rng(seed)
    incr = rng.gamma(shape=4.0, scale=250.0, size=(n, n))
    incr *= np.linspace(1.0, 0.05, n)[None, :]
    cum = np.cumsum(incr, axis=1)
    for i in range(n):
        cum[i, n - i :] = np.nan
    return cum


@pytest.fixture(scope="module")
def raa():
    return read_triangle_csv(ROOT / "data" / "raa.csv").to_numpy()


@pytest.mark.parametrize("n", [4, 5, 7, 10, 15])
def test_projection_follows_the_factors(n):
    tri = make_triangle(n)
    res = mack_chain_ladder(tri)
    for i in range(n):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        expected = tri[i, last] * np.prod(res.f[last:])
        assert res.ultimate[i] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("n", [4, 5, 7, 10, 15])
def test_observed_part_is_untouched(n):
    tri = make_triangle(n)
    res = mack_chain_ladder(tri)
    observed = ~np.isnan(tri)
    np.testing.assert_allclose(res.full_triangle[observed], tri[observed], rtol=0)


@pytest.mark.parametrize("n", [4, 5, 7, 10, 15])
def test_mse_decomposes(n):
    """Mack.S.E^2 = process risk^2 + parameter risk^2, by origin."""
    tri = make_triangle(n)
    res = mack_chain_ladder(tri)
    np.testing.assert_allclose(
        res.mack_se**2,
        res.process_risk**2 + res.parameter_risk**2,
        rtol=1e-10,
    )


@pytest.mark.parametrize("n", [4, 5, 7, 10, 15])
def test_total_decomposes(n):
    tri = make_triangle(n)
    res = mack_chain_ladder(tri)
    assert res.total_mack_se**2 == pytest.approx(
        res.total_process_risk**2 + res.total_parameter_risk**2, rel=1e-10
    )


@pytest.mark.parametrize("n", [5, 8, 12])
def test_closed_form_mse_agrees_with_the_recursion(n):
    """Cross-check the recursion against Mack's (1993) closed form.

    mse(R_i) = C_{i,n}^2 * sum_k (sigma_k^2 / f_k^2) * (1/C_{i,k} + 1/S_k)

    The implementation uses the recursive formulation of Mack (1999); the two
    must coincide, and computing the closed form here independently is the
    point of the test.
    """
    tri = make_triangle(n)
    res = mack_chain_ladder(tri)
    full, f, sigma2, s = res.full_triangle, res.f, res.sigma**2, res.s
    for i in range(n):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        if last >= n - 1:
            continue
        acc = sum(
            (sigma2[k] / f[k] ** 2) * (1.0 / full[i, k] + 1.0 / s[k])
            for k in range(last, n - 1)
        )
        expected = full[i, n - 1] ** 2 * acc
        assert res.mack_se[i] ** 2 == pytest.approx(expected, rel=1e-9)


def test_fully_developed_origin_has_no_error(raa):
    res = mack_chain_ladder(raa)
    assert res.ibnr[0] == 0.0
    assert res.mack_se[0] == 0.0
    assert res.process_risk[0] == 0.0
    assert res.parameter_risk[0] == 0.0


@pytest.mark.parametrize("n", [4, 6, 10])
def test_scale_equivariance(n):
    """Multiplying the triangle by c multiplies reserve and s.e. by c.

    Mack's model is scale invariant in this sense: sigma_k^2 scales with c and
    so the standard error scales linearly, not quadratically.
    """
    tri = make_triangle(n)
    c = 1000.0
    a = mack_chain_ladder(tri)
    b = mack_chain_ladder(tri * c)
    np.testing.assert_allclose(b.f, a.f, rtol=1e-12)
    np.testing.assert_allclose(b.ibnr, a.ibnr * c, rtol=1e-10)
    np.testing.assert_allclose(b.mack_se, a.mack_se * c, rtol=1e-10)
    assert b.total_mack_se == pytest.approx(a.total_mack_se * c, rel=1e-10)


def test_deterministic_triangle_has_zero_process_risk():
    """If every link ratio in a column is identical, sigma_k is zero."""
    n = 6
    f_true = np.array([2.0, 1.5, 1.2, 1.1, 1.05])
    cum = np.zeros((n, n))
    cum[:, 0] = np.arange(1, n + 1) * 100.0
    for k in range(n - 1):
        cum[:, k + 1] = cum[:, k] * f_true[k]
    for i in range(n):
        cum[i, n - i :] = np.nan
    res = mack_chain_ladder(cum)
    np.testing.assert_allclose(res.f, f_true, rtol=1e-12)
    np.testing.assert_allclose(res.sigma, np.zeros(n - 1), atol=1e-9)
    np.testing.assert_allclose(res.mack_se, np.zeros(n), atol=1e-9)
    assert res.total_mack_se == pytest.approx(0.0, abs=1e-9)


def test_incremental_round_trip(raa):
    back = incr_to_cum(cum_to_incr(raa))
    observed = ~np.isnan(raa)
    np.testing.assert_allclose(back[observed], raa[observed], rtol=1e-12)
    assert np.isnan(back[~observed]).all()


def test_mack_sigma_rule():
    """sigma_{n-1}^2 = min(sigma_{n-2}^4/sigma_{n-3}^2, sigma_{n-3}^2, sigma_{n-2}^2)."""
    sigma2 = np.array([100.0, 25.0, 16.0, np.nan])
    out = _extrapolate_sigma_mack(sigma2)
    assert out[3] == pytest.approx(min(16.0**2 / 25.0, 25.0, 16.0))
    np.testing.assert_allclose(out[:3], sigma2[:3])


def test_rejects_gapped_triangle():
    tri = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, np.nan, np.nan]])
    with pytest.raises(ValueError, match="gap"):
        mack_chain_ladder(tri)


def test_rejects_unknown_est_sigma(raa):
    with pytest.raises(ValueError, match="est_sigma"):
        mack_chain_ladder(raa, est_sigma="whatever")


def test_loglinear_runs_and_differs_from_mack(raa):
    """The log-linear option is exercised, but only for not crashing.

    Its agreement with R is deliberately not asserted anywhere; see the
    docstring of ``_extrapolate_sigma_loglinear``.
    """
    a = mack_chain_ladder(raa, est_sigma="mack")
    b = mack_chain_ladder(raa, est_sigma="log-linear")
    np.testing.assert_allclose(a.f, b.f, rtol=1e-12)
    np.testing.assert_allclose(a.ibnr, b.ibnr, rtol=1e-12)
    assert np.isfinite(b.total_mack_se)
