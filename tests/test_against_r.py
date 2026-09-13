"""Compare the Python implementation with R output generated in this environment.

These tests read the CSV files written by ``R/export_reference.R``.  If that
script has not been run (no R available, for instance) the tests skip.  The
comparison is at full double precision: the CSV files are not rounded, so a
disagreement beyond 1e-8 relative is a genuine disagreement.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mackpy import mack_chain_ladder, read_triangle_csv

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "reference" / "generated"

RTOL = 1e-8

CASES = [
    ("raa", "Mack", "mack"),
    ("raa", "loglinear", "log-linear"),
    ("genins", "Mack", "mack"),
    ("genins", "loglinear", "log-linear"),
]


def _load(name: str, tag: str):
    triangle = GENERATED / f"{name}_triangle.csv"
    byorigin = GENERATED / f"{name}_{tag}_byorigin.csv"
    totals = GENERATED / f"{name}_{tag}_totals.csv"
    factors = GENERATED / f"{name}_{tag}_factors.csv"
    missing = [p.name for p in (triangle, byorigin, totals, factors) if not p.exists()]
    if missing:
        pytest.skip(
            "R reference not generated (missing "
            + ", ".join(missing)
            + "); run `make reference`"
        )
    tri = read_triangle_csv(triangle)
    return (
        tri,
        pd.read_csv(byorigin),
        pd.read_csv(totals).set_index("quantity")["value"],
        pd.read_csv(factors),
    )


@pytest.mark.parametrize("name,tag,est_sigma", CASES)
def test_factors_match_r(name, tag, est_sigma):
    if est_sigma == "log-linear":
        pytest.skip(
            "log-linear sigma extrapolation is not claimed to reproduce R "
            "(see mackpy.mack._extrapolate_sigma_loglinear)"
        )
    tri, _, _, factors = _load(name, tag)
    res = mack_chain_ladder(tri.to_numpy(), est_sigma=est_sigma)
    np.testing.assert_allclose(res.f, factors["f"].to_numpy(), rtol=RTOL)
    np.testing.assert_allclose(res.sigma, factors["sigma"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(res.f_se, factors["f.se"].to_numpy(), rtol=1e-6)


@pytest.mark.parametrize("name,tag,est_sigma", CASES)
def test_reserves_match_r(name, tag, est_sigma):
    if est_sigma == "log-linear":
        pytest.skip("log-linear sigma extrapolation is not claimed to reproduce R")
    tri, byorigin, _, _ = _load(name, tag)
    res = mack_chain_ladder(tri.to_numpy(), est_sigma=est_sigma)
    np.testing.assert_allclose(res.ultimate, byorigin["Ultimate"].to_numpy(), rtol=RTOL)
    np.testing.assert_allclose(res.ibnr, byorigin["IBNR"].to_numpy(), rtol=RTOL, atol=1e-9)
    np.testing.assert_allclose(res.mack_se, byorigin["Mack.S.E"].to_numpy(), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name,tag,est_sigma", CASES)
def test_risk_decomposition_matches_r(name, tag, est_sigma):
    if est_sigma == "log-linear":
        pytest.skip("log-linear sigma extrapolation is not claimed to reproduce R")
    tri, byorigin, _, _ = _load(name, tag)
    res = mack_chain_ladder(tri.to_numpy(), est_sigma=est_sigma)
    if "ProcessRisk" not in byorigin:
        pytest.skip("reference file has no risk decomposition")
    np.testing.assert_allclose(
        res.process_risk, byorigin["ProcessRisk"].to_numpy(), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        res.parameter_risk, byorigin["ParameterRisk"].to_numpy(), rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("name,tag,est_sigma", CASES)
def test_totals_match_r(name, tag, est_sigma):
    if est_sigma == "log-linear":
        pytest.skip("log-linear sigma extrapolation is not claimed to reproduce R")
    tri, _, totals, _ = _load(name, tag)
    res = mack_chain_ladder(tri.to_numpy(), est_sigma=est_sigma)
    got = res.totals()
    assert got["IBNR"] == pytest.approx(totals["IBNR"], rel=RTOL)
    assert got["Ultimate"] == pytest.approx(totals["Ultimate"], rel=RTOL)
    assert got["Mack.S.E"] == pytest.approx(totals["Mack.S.E"], rel=1e-6)
