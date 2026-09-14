"""The tail factor of Mack (1999).

A tail factor is treated as one extra development period. That framing makes
the expected behaviour sharp: with a known tail everything simply scales, and
any uncertainty attached to the tail can only increase the standard error.
"""

from pathlib import Path

import numpy as np
import pytest

from mackpy import mack_chain_ladder, read_triangle_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raa():
    return read_triangle_csv(ROOT / "data" / "raa.csv").to_numpy()


def test_tail_of_one_is_the_no_tail_case(raa):
    a = mack_chain_ladder(raa)
    b = mack_chain_ladder(raa, tail=1.0)
    assert not b.has_tail
    np.testing.assert_allclose(a.ultimate, b.ultimate, rtol=0)
    np.testing.assert_allclose(a.mack_se, b.mack_se, rtol=0)
    assert a.total_mack_se == b.total_mack_se


def test_tail_adds_a_development_period(raa):
    plain = mack_chain_ladder(raa)
    with_tail = mack_chain_ladder(raa, tail=1.05)
    assert with_tail.has_tail
    assert len(with_tail.f) == len(plain.f) + 1
    assert with_tail.f[-1] == 1.05
    assert with_tail.full_triangle.shape[1] == plain.full_triangle.shape[1] + 1
    assert with_tail.sigma_source[-1] == "tail"


def test_known_tail_scales_ultimates_and_errors(raa):
    """With a tail of known size everything downstream is multiplied by it."""
    tail = 1.05
    plain = mack_chain_ladder(raa)
    with_tail = mack_chain_ladder(raa, tail=tail)
    np.testing.assert_allclose(with_tail.ultimate, plain.ultimate * tail, rtol=1e-12)
    np.testing.assert_allclose(with_tail.mack_se, plain.mack_se * tail, rtol=1e-10)
    assert with_tail.total_mack_se == pytest.approx(
        plain.total_mack_se * tail, rel=1e-10
    )


def test_known_tail_raises_the_reserve_of_every_origin(raa):
    plain = mack_chain_ladder(raa)
    with_tail = mack_chain_ladder(raa, tail=1.05)
    assert np.all(with_tail.ibnr >= plain.ibnr - 1e-9)
    # even the oldest origin period, complete without a tail, now has a reserve
    assert plain.ibnr[0] == 0.0
    assert with_tail.ibnr[0] > 0.0


def test_tail_uncertainty_increases_the_standard_error(raa):
    certain = mack_chain_ladder(raa, tail=1.05)
    uncertain = mack_chain_ladder(raa, tail=1.05, tail_se=0.02)
    assert np.all(uncertain.mack_se >= certain.mack_se - 1e-9)
    assert uncertain.total_mack_se > certain.total_mack_se
    np.testing.assert_allclose(uncertain.ultimate, certain.ultimate, rtol=1e-12)


def test_tail_sigma_feeds_process_risk_and_tail_se_feeds_parameter_risk(raa):
    base = mack_chain_ladder(raa, tail=1.05)
    proc = mack_chain_ladder(raa, tail=1.05, tail_sigma=50.0)
    param = mack_chain_ladder(raa, tail=1.05, tail_se=0.02)

    assert np.all(proc.process_risk >= base.process_risk - 1e-9)
    np.testing.assert_allclose(proc.parameter_risk, base.parameter_risk, rtol=1e-10)

    assert np.all(param.parameter_risk >= base.parameter_risk - 1e-9)
    np.testing.assert_allclose(param.process_risk, base.process_risk, rtol=1e-10)


def test_the_oldest_origin_gains_error_only_from_the_tail(raa):
    """Origin 1981 is fully run off, so all of its error comes from the tail."""
    res = mack_chain_ladder(raa, tail=1.05, tail_se=0.02)
    assert res.mack_se[0] > 0.0
    expected = res.full_triangle[0, -2] * 0.02
    assert res.parameter_risk[0] == pytest.approx(expected, rel=1e-10)


def test_uncertainty_without_a_factor_still_counts(raa):
    """tail = 1 with a standard error is a tail: known size, unknown accuracy."""
    res = mack_chain_ladder(raa, tail=1.0, tail_se=0.02)
    assert res.has_tail
    plain = mack_chain_ladder(raa)
    np.testing.assert_allclose(res.ultimate, plain.ultimate, rtol=1e-12)
    assert res.total_mack_se > plain.total_mack_se


def test_mse_still_decomposes_with_a_tail(raa):
    res = mack_chain_ladder(raa, tail=1.07, tail_sigma=40.0, tail_se=0.03)
    np.testing.assert_allclose(
        res.mack_se**2, res.process_risk**2 + res.parameter_risk**2, rtol=1e-10
    )
    assert res.total_mack_se**2 == pytest.approx(
        res.total_process_risk**2 + res.total_parameter_risk**2, rel=1e-10
    )


def test_total_still_exceeds_the_root_sum_of_squares(raa):
    res = mack_chain_ladder(raa, tail=1.05, tail_se=0.02)
    rss = float(np.sqrt(np.sum(res.mack_se**2)))
    assert rss < res.total_mack_se < float(np.sum(res.mack_se))


def test_rejects_impossible_tails(raa):
    with pytest.raises(ValueError, match="tail factor"):
        mack_chain_ladder(raa, tail=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        mack_chain_ladder(raa, tail=1.05, tail_se=-0.1)
