"""Compare the Python implementation with R output generated in this environment.

These tests read the CSV files written by ``R/export_reference.R``. The cases
are discovered from whatever that script produced, so adding a triangle on the
R side automatically widens the comparison here with no change to this file.
If the script has not been run the tests skip, and the CI workflow has a step
that fails when the files are absent so that a skip cannot pass for a pass.

The comparison is at full double precision: the CSV files are not rounded, so
a disagreement beyond the tolerances below is a real disagreement.

Cases with ``est.sigma = "log-linear"`` are marked ``informational``. The
Python implementation deliberately does not reproduce R's fallback logic (see
``mackpy.mack._extrapolate_sigma_loglinear``), so those comparisons are run and
reported but are not allowed to fail the build.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mackpy import mack_chain_ladder, read_triangle_csv

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "reference" / "generated"

RTOL = 1e-8
RTOL_SIGMA = 1e-6

CASE_RE = re.compile(r"^(?P<name>.+)_a(?P<alpha>\d)_(?P<sigma>Mack|loglinear)_byorigin\.csv$")


def discover_cases():
    """Every (triangle, alpha, est_sigma) the R script left behind."""
    if not GENERATED.is_dir():
        return []
    cases = []
    for path in sorted(GENERATED.glob("*_byorigin.csv")):
        match = CASE_RE.match(path.name)
        if not match:
            continue
        name = match.group("name")
        if not (GENERATED / f"{name}_triangle.csv").exists():
            continue
        alpha = int(match.group("alpha"))
        tag = match.group("sigma")
        est_sigma = "mack" if tag == "Mack" else "log-linear"
        marks = [pytest.mark.informational] if est_sigma == "log-linear" else []
        cases.append(
            pytest.param(
                name, alpha, est_sigma, tag,
                id=f"{name}-alpha{alpha}-{tag}",
                marks=marks,
            )
        )
    return cases


CASES = discover_cases()

if not CASES:
    CASES = [
        pytest.param(
            None, None, None, None,
            id="no-reference",
            marks=pytest.mark.skip(
                reason="R reference not generated; run `make reference`"
            ),
        )
    ]


def load(name: str, alpha: int, tag: str):
    stem = f"{name}_a{alpha}_{tag}"
    tri = read_triangle_csv(GENERATED / f"{name}_triangle.csv")
    byorigin = pd.read_csv(GENERATED / f"{stem}_byorigin.csv")
    totals = pd.read_csv(GENERATED / f"{stem}_totals.csv").set_index("quantity")["value"]
    factors = pd.read_csv(GENERATED / f"{stem}_factors.csv")
    return tri, byorigin, totals, factors


def fit(tri, alpha, est_sigma):
    return mack_chain_ladder(tri.to_numpy(), est_sigma=est_sigma, alpha=alpha)


@pytest.mark.parametrize("name,alpha,est_sigma,tag", CASES)
def test_factors_match_r(name, alpha, est_sigma, tag):
    tri, _, _, factors = load(name, alpha, tag)
    res = fit(tri, alpha, est_sigma)
    np.testing.assert_allclose(res.f, factors["f"].to_numpy(), rtol=RTOL)
    np.testing.assert_allclose(res.sigma, factors["sigma"].to_numpy(), rtol=RTOL_SIGMA)
    np.testing.assert_allclose(res.f_se, factors["f.se"].to_numpy(), rtol=RTOL_SIGMA)


@pytest.mark.parametrize("name,alpha,est_sigma,tag", CASES)
def test_reserves_match_r(name, alpha, est_sigma, tag):
    tri, byorigin, _, _ = load(name, alpha, tag)
    res = fit(tri, alpha, est_sigma)
    np.testing.assert_allclose(res.ultimate, byorigin["Ultimate"].to_numpy(), rtol=RTOL)
    np.testing.assert_allclose(
        res.ibnr, byorigin["IBNR"].to_numpy(), rtol=RTOL, atol=1e-9
    )
    np.testing.assert_allclose(
        res.mack_se, byorigin["Mack.S.E"].to_numpy(), rtol=RTOL_SIGMA, atol=1e-9
    )


@pytest.mark.parametrize("name,alpha,est_sigma,tag", CASES)
def test_risk_decomposition_matches_r(name, alpha, est_sigma, tag):
    tri, byorigin, _, _ = load(name, alpha, tag)
    if "ProcessRisk" not in byorigin:
        pytest.skip("reference file has no risk decomposition")
    res = fit(tri, alpha, est_sigma)
    np.testing.assert_allclose(
        res.process_risk, byorigin["ProcessRisk"].to_numpy(),
        rtol=RTOL_SIGMA, atol=1e-9,
    )
    np.testing.assert_allclose(
        res.parameter_risk, byorigin["ParameterRisk"].to_numpy(),
        rtol=RTOL_SIGMA, atol=1e-9,
    )


@pytest.mark.parametrize("name,alpha,est_sigma,tag", CASES)
def test_totals_match_r(name, alpha, est_sigma, tag):
    tri, _, totals, _ = load(name, alpha, tag)
    res = fit(tri, alpha, est_sigma)
    got = res.totals()
    assert got["IBNR"] == pytest.approx(totals["IBNR"], rel=RTOL)
    assert got["Ultimate"] == pytest.approx(totals["Ultimate"], rel=RTOL)
    assert got["Mack.S.E"] == pytest.approx(totals["Mack.S.E"], rel=RTOL_SIGMA)
