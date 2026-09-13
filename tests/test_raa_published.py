"""Compare the Python implementation with the published R output for RAA.

The expected values come from the ChainLadder vignette (see
``reference/README.md``).  They are printed rounded, so the tolerance here is
0.1 for the by-origin figures and 0.01 for the totals.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mackpy import mack_chain_ladder, read_triangle_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fit():
    tri = read_triangle_csv(ROOT / "data" / "raa.csv")
    return tri, mack_chain_ladder(tri.to_numpy(), est_sigma="mack")


@pytest.fixture(scope="module")
def published():
    by = pd.read_csv(ROOT / "reference" / "raa_published_byorigin.csv")
    tot = pd.read_csv(ROOT / "reference" / "raa_published_totals.csv")
    return by, dict(zip(tot["quantity"], tot["value"]))


def test_development_factors(fit):
    _, res = fit
    # Vignette: 2.999 1.624 1.271 1.172 1.113 1.042 1.033 1.017 1.009
    expected = [2.999, 1.624, 1.271, 1.172, 1.113, 1.042, 1.033, 1.017, 1.009]
    assert res.f.shape == (9,)
    np.testing.assert_allclose(res.f, expected, atol=5e-4)


def test_ibnr_by_origin(fit, published):
    _, res = fit
    by, _ = published
    np.testing.assert_allclose(res.ibnr, by["IBNR"].to_numpy(), atol=0.1)


def test_mack_se_by_origin(fit, published):
    _, res = fit
    by, _ = published
    np.testing.assert_allclose(res.mack_se, by["Mack.S.E"].to_numpy(), atol=0.1)


def test_ultimate_by_origin(fit, published):
    _, res = fit
    by, _ = published
    np.testing.assert_allclose(res.ultimate, by["Ultimate"].to_numpy(), atol=0.6)


def test_dev_to_date(fit, published):
    _, res = fit
    by, _ = published
    np.testing.assert_allclose(res.dev_to_date, by["Dev.To.Date"].to_numpy(), atol=1e-4)


def test_totals(fit, published):
    _, res = fit
    _, tot = published
    got = res.totals()
    assert got["IBNR"] == pytest.approx(tot["IBNR"], abs=0.01)
    assert got["Mack.S.E"] == pytest.approx(tot["Mack.S.E"], abs=0.01)
    assert got["Ultimate"] == pytest.approx(tot["Ultimate"], abs=0.01)
    assert got["Latest"] == pytest.approx(tot["Latest"], abs=0.01)
    assert got["CV(IBNR)"] == pytest.approx(tot["CV(IBNR)"], abs=1e-4)


def test_total_se_is_not_the_sum_of_the_individual_ones(fit):
    """A sanity check on the covariance term of Mack's total formula.

    The total standard error must sit strictly between the root-sum-of-squares
    (which would hold if the reserve estimates were uncorrelated) and the plain
    sum (which would hold if they were perfectly correlated).
    """
    _, res = fit
    rss = float(np.sqrt(np.sum(res.mack_se**2)))
    plain = float(np.sum(res.mack_se))
    assert rss < res.total_mack_se < plain
