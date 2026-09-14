"""Tests for the Student t tail probability used by the log-linear fit.

The anchors below are quantiles whose tail probabilities are known exactly or
are standard table values, so the test does not depend on the implementation it
is checking.
"""

import math

import pytest

from mackpy.tdist import betainc, t_sf


def test_median():
    for df in (1, 2, 5, 30, 1000):
        assert t_sf(0.0, df) == pytest.approx(0.5, abs=1e-12)


def test_cauchy_closed_form():
    """With one degree of freedom the t distribution is Cauchy."""
    for t in (-3.0, -0.5, 0.0, 1.0, 2.5, 7.0):
        expected = 0.5 - math.atan(t) / math.pi
        assert t_sf(t, 1) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_symmetry():
    for df in (1, 3, 7, 40):
        for t in (0.3, 1.1, 2.9, 5.0):
            assert t_sf(-t, df) == pytest.approx(1.0 - t_sf(t, df), rel=1e-12)


@pytest.mark.parametrize(
    "df,quantile,tail",
    [
        (5, 2.015048372669157, 0.05),
        (10, 2.228138851986273, 0.025),
        (30, 1.6972608865939574, 0.05),
    ],
)
def test_table_quantiles(df, quantile, tail):
    """Standard critical values: the tail beyond t_{df, 1-p} is p."""
    assert t_sf(quantile, df) == pytest.approx(tail, abs=1e-9)


def test_large_df_approaches_the_normal():
    for t in (0.5, 1.0, 1.96, 3.0):
        normal = 0.5 * math.erfc(t / math.sqrt(2.0))
        assert t_sf(t, 2_000_000) == pytest.approx(normal, abs=1e-6)


def test_monotone_in_t():
    values = [t_sf(t, 7) for t in (-3.0, -1.0, 0.0, 1.0, 3.0)]
    assert values == sorted(values, reverse=True)


def test_betainc_endpoints_and_symmetry():
    assert betainc(2.0, 3.0, 0.0) == 0.0
    assert betainc(2.0, 3.0, 1.0) == 1.0
    for a, b, x in [(0.5, 0.5, 0.3), (2.0, 5.0, 0.6), (7.0, 1.5, 0.2)]:
        assert betainc(a, b, x) == pytest.approx(1.0 - betainc(b, a, 1.0 - x), rel=1e-12)


def test_betainc_uniform_case():
    """I_x(1, 1) is the uniform distribution function."""
    for x in (0.1, 0.37, 0.9):
        assert betainc(1.0, 1.0, x) == pytest.approx(x, rel=1e-12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        betainc(1.0, 1.0, 1.5)
    with pytest.raises(ValueError):
        t_sf(1.0, 0)
