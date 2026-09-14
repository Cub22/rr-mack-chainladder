"""Mack's distribution-free chain ladder.

Reference
---------
Mack, T. (1993). Distribution-free calculation of the standard error of chain
ladder reserve estimates. *ASTIN Bulletin* 23(2), 213-225.

Mack, T. (1999). The standard error of chain ladder reserve estimates:
recursive calculation and inclusion of a tail factor. *ASTIN Bulletin* 29(2),
361-366.  The recursion of the 1999 paper is what is implemented here; for the
individual origin periods and no tail factor it is algebraically identical to
the closed-form mean squared error of the 1993 paper, which
``tests/test_properties.py`` checks by computing both.

Model
-----
With C_{i,k} the cumulative claims of origin period i after k development
periods and F_{i,k} = C_{i,k+1} / C_{i,k}:

    (CL1)  E[F_{i,k} | C_{i,1}, ..., C_{i,k}]   = f_k
    (CL2)  Var(F_{i,k} | C_{i,1}, ..., C_{i,k}) = sigma_k^2 / (w_{i,k} C_{i,k}^alpha)
    (CL3)  origin periods are independent

Estimators
----------
    f_k       = sum_i w_{i,k} C_{i,k}^alpha F_{i,k} / sum_i w_{i,k} C_{i,k}^alpha
    sigma_k^2 = 1/(I_k - 1) * sum_i w_{i,k} C_{i,k}^alpha (F_{i,k} - f_k)^2
    S_k       = sum_i w_{i,k} C_{i,k}^alpha

alpha = 1 with unit weights is the volume weighted chain ladder and the
default of R's ``ChainLadder::MackChainLadder``; alpha = 0 gives the simple
average of the link ratios and alpha = 2 the ordinary regression estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tdist import t_sf
from .triangle import check_triangle, latest_diagonal

__all__ = ["MackResult", "mack_chain_ladder", "LOGLINEAR_ALPHA"]

#: The slope of the log-linear sigma regression is used only if its one-sided
#: p-value is below this; otherwise Mack's rule is used instead.
LOGLINEAR_ALPHA = 0.05


@dataclass
class MackResult:
    """Output of :func:`mack_chain_ladder`.

    Arrays indexed by development period have length ``n - 1`` without a tail
    factor and ``n`` with one, the last entry then being the tail.
    """

    triangle: np.ndarray
    full_triangle: np.ndarray
    f: np.ndarray
    sigma: np.ndarray
    f_se: np.ndarray
    s: np.ndarray
    latest: np.ndarray
    ultimate: np.ndarray
    ibnr: np.ndarray
    process_risk: np.ndarray
    parameter_risk: np.ndarray
    mack_se: np.ndarray
    total_ibnr: float
    total_process_risk: float
    total_parameter_risk: float
    total_mack_se: float
    est_sigma: str = "mack"
    alpha: int = 1
    tail: float = 1.0
    has_tail: bool = False
    sigma_source: tuple = ()
    n_obs: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def dev_to_date(self) -> np.ndarray:
        return self.latest / self.ultimate

    def summary(self):
        """Return a :class:`pandas.DataFrame` shaped like R's ``summary()$ByOrigin``."""
        import pandas as pd

        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(self.ibnr > 0, self.mack_se / self.ibnr, np.nan)
        return pd.DataFrame(
            {
                "Latest": self.latest,
                "Dev.To.Date": self.dev_to_date,
                "Ultimate": self.ultimate,
                "IBNR": self.ibnr,
                "Mack.S.E": self.mack_se,
                "CV(IBNR)": cv,
            }
        )

    def totals(self) -> dict:
        latest = float(np.nansum(self.latest))
        ultimate = float(np.nansum(self.ultimate))
        return {
            "Latest": latest,
            "Dev": latest / ultimate,
            "Ultimate": ultimate,
            "IBNR": self.total_ibnr,
            "Mack.S.E": self.total_mack_se,
            "CV(IBNR)": self.total_mack_se / self.total_ibnr,
        }


def _prepare_weights(tri: np.ndarray, weights) -> np.ndarray:
    if weights is None:
        return np.ones_like(tri)
    w = np.asarray(weights, dtype=float)
    if w.shape != tri.shape:
        raise ValueError(f"weights must have shape {tri.shape}, got {w.shape}")
    finite = w[~np.isnan(w)]
    if finite.size and np.any(finite < 0):
        raise ValueError("weights must be non-negative")
    return np.nan_to_num(w, nan=0.0)


def _usable(tri: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Rows contributing a link ratio in development period ``k``."""
    return ~np.isnan(tri[:, k]) & ~np.isnan(tri[:, k + 1]) & (w[:, k] > 0)


def _development_factors(tri: np.ndarray, w: np.ndarray, alpha: int):
    """Factors, the weight totals S_k, and the number of ratios per period."""
    n = tri.shape[1]
    f = np.full(n - 1, np.nan)
    s = np.full(n - 1, np.nan)
    n_obs = np.zeros(n - 1, dtype=int)
    for k in range(n - 1):
        rows = _usable(tri, w, k)
        n_obs[k] = int(rows.sum())
        if n_obs[k] == 0:
            continue
        c_k = tri[rows, k]
        weight = w[rows, k] * c_k**alpha
        s[k] = weight.sum()
        f[k] = float(np.sum(weight * (tri[rows, k + 1] / c_k)) / s[k])
    return f, s, n_obs


def _sigma_from_data(tri: np.ndarray, w: np.ndarray, alpha: int, f: np.ndarray):
    """sigma_k^2 for those development periods with at least two ratios."""
    n = tri.shape[1]
    sigma2 = np.full(n - 1, np.nan)
    for k in range(n - 1):
        rows = _usable(tri, w, k)
        i_k = int(rows.sum())
        if i_k < 2:
            continue
        c_k = tri[rows, k]
        weight = w[rows, k] * c_k**alpha
        ratio = tri[rows, k + 1] / c_k
        sigma2[k] = float(np.sum(weight * (ratio - f[k]) ** 2) / (i_k - 1))
    return sigma2


def _extrapolate_sigma_mack(sigma2: np.ndarray) -> np.ndarray:
    """Mack's rule for development periods that cannot be estimated.

    sigma_{n-1}^2 = min( sigma_{n-2}^4 / sigma_{n-3}^2,
                         min(sigma_{n-3}^2, sigma_{n-2}^2) )

    Applied left to right, so a triangle missing more than one estimate at the
    end is still handled.
    """
    out = sigma2.copy()
    for k in range(len(out)):
        if not np.isnan(out[k]):
            continue
        if k < 2 or np.isnan(out[k - 1]) or np.isnan(out[k - 2]):
            out[k] = 0.0
            continue
        a, b = out[k - 2], out[k - 1]
        if a <= 0:
            out[k] = min(a, b)
        else:
            out[k] = min(b**2 / a, min(a, b))
    return out


def _loglinear_fit(sigma2: np.ndarray):
    """Fit log(sigma_k) = a + b k by least squares on the estimable periods.

    Returns ``(a, b, p_value, df)``, the p-value being one-sided against the
    alternative that the slope is negative.  Returns ``None`` when fewer than
    three points are available: a line through two points has no residual
    degrees of freedom, so its slope cannot be tested.
    """
    known = np.flatnonzero(~np.isnan(sigma2) & (sigma2 > 0))
    if known.size < 3:
        return None
    x = known.astype(float)
    y = np.log(np.sqrt(sigma2[known]))
    x_bar, y_bar = x.mean(), y.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx == 0.0:
        return None
    b = float(np.sum((x - x_bar) * (y - y_bar)) / sxx)
    a = float(y_bar - b * x_bar)
    df = int(x.size - 2)
    resid = y - (a + b * x)
    s2 = float(np.sum(resid**2) / df)
    if s2 <= 0.0:
        return a, b, 0.0, df  # a perfect fit is as significant as it gets
    se_b = np.sqrt(s2 / sxx)
    t_stat = b / se_b
    return a, b, float(t_sf(-t_stat, df)), df


def _extrapolate_sigma_loglinear(sigma2: np.ndarray):
    """Log-linear extrapolation with a significance check and a fallback.

    Mack's suggestion is to extrapolate the decay of sigma_k by fitting a
    straight line to log(sigma_k).  That is only sensible if the decay is real,
    so the slope is tested against zero and Mack's minimum rule is used instead
    when the slope is not significantly negative at the five per cent level.

    R's ``est.sigma = "log-linear"`` follows the same idea and also falls back
    to Mack's rule when the fit is poor, but the criterion it applies is its
    own.  Agreement with R is therefore checked as an informational test rather
    than asserted; see ``tests/test_against_r.py``.

    Returns the completed ``sigma^2`` and a label per development period saying
    where each value came from.
    """
    out = sigma2.copy()
    source = ["data" if not np.isnan(v) else "pending" for v in sigma2]
    missing = np.flatnonzero(np.isnan(out))
    if missing.size == 0:
        return out, tuple(source)

    fit = _loglinear_fit(sigma2)
    if fit is None or fit[2] > LOGLINEAR_ALPHA:
        filled = _extrapolate_sigma_mack(sigma2)
        for k in missing:
            out[k] = filled[k]
            source[k] = "mack-fallback"
        return out, tuple(source)

    a, b = fit[0], fit[1]
    for k in missing:
        out[k] = float(np.exp(a + b * float(k)) ** 2)
        source[k] = "log-linear"
    return out, tuple(source)


def mack_chain_ladder(
    triangle,
    est_sigma: str = "mack",
    alpha: int = 1,
    weights=None,
    tail: float = 1.0,
    tail_sigma: float = 0.0,
    tail_se: float = 0.0,
) -> MackResult:
    """Fit the Mack chain ladder to a cumulative run-off triangle.

    Parameters
    ----------
    triangle
        Array-like of shape ``(m, n)`` with ``NaN`` in the unobserved part.
    est_sigma
        How to obtain ``sigma`` for development periods with fewer than two
        ratios: ``"mack"`` (Mack's extrapolation rule, the setting under which
        agreement with R is asserted) or ``"log-linear"``.
    alpha
        Exponent in the variance assumption, 0, 1 or 2.  ``1`` is the volume
        weighted chain ladder and the R default.
    weights
        Optional array of the same shape as the triangle.  A zero excludes that
        ratio from the estimation entirely, which is the usual way of dropping
        an outlying development.
    tail
        Tail factor applied beyond the last observed development period.  The
        default of 1.0 means no tail.
    tail_sigma, tail_se
        Uncertainty attached to the tail factor: ``tail_sigma`` enters the
        process error exactly as any other ``sigma_k``, ``tail_se`` the
        estimation error as any other ``f.se_k``.  Both default to zero, which
        treats the tail factor as known - convenient, and almost certainly
        optimistic.

    Returns
    -------
    MackResult
    """
    tri = np.asarray(triangle, dtype=float)
    check_triangle(tri)
    if alpha not in (0, 1, 2):
        raise ValueError("alpha must be 0, 1 or 2")
    if tail <= 0:
        raise ValueError("tail factor must be positive")
    if tail_sigma < 0 or tail_se < 0:
        raise ValueError("tail uncertainty must be non-negative")
    m, n = tri.shape

    w = _prepare_weights(tri, weights)

    f, s, n_obs = _development_factors(tri, w, alpha)
    if np.isnan(f).any():
        raise ValueError("some development period has no usable link ratio")

    sigma2_data = _sigma_from_data(tri, w, alpha, f)
    if est_sigma == "mack":
        sigma2 = _extrapolate_sigma_mack(sigma2_data)
        source = tuple("data" if not np.isnan(v) else "mack" for v in sigma2_data)
    elif est_sigma in ("log-linear", "loglinear"):
        sigma2, source = _extrapolate_sigma_loglinear(sigma2_data)
    else:
        raise ValueError("est_sigma must be 'mack' or 'log-linear'")
    sigma = np.sqrt(sigma2)
    f_se = sigma / np.sqrt(s)

    # A tail factor is one more development period with its own factor and its
    # own uncertainty; everything downstream then runs unchanged.
    has_tail = tail != 1.0 or tail_sigma > 0.0 or tail_se > 0.0
    if has_tail:
        f = np.append(f, tail)
        sigma = np.append(sigma, tail_sigma)
        sigma2 = np.append(sigma2, tail_sigma**2)
        f_se = np.append(f_se, tail_se)
        s = np.append(s, np.nan)
        source = source + ("tail",)
        full = np.column_stack([tri, np.full(m, np.nan)])
        n_full = n + 1
    else:
        full = tri.copy()
        n_full = n

    # --- projection of the lower right part ------------------------------
    for i in range(m):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        for k in range(last, n_full - 1):
            full[i, k + 1] = full[i, k] * f[k]

    latest = latest_diagonal(tri)
    ultimate = full[:, n_full - 1]
    ibnr = ultimate - latest

    # --- prediction error, recursively (Mack 1999) ------------------------
    # process:   PR_{k+1}^2 = f_k^2 PR_k^2 + sigma_k^2 C_k^(2-alpha) / w_k
    # parameter: PA_{k+1}^2 = f_k^2 PA_k^2 + f.se_k^2 C_k^2
    process = np.zeros(m)
    parameter = np.zeros(m)
    for i in range(m):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        proc2 = 0.0
        param2 = 0.0
        for k in range(last, n_full - 1):
            c_ik = full[i, k]
            weight = w[i, k] if k < n - 1 else 1.0
            if weight <= 0:
                weight = 1.0
            proc2 = f[k] ** 2 * proc2 + sigma2[k] * c_ik ** (2 - alpha) / weight
            param2 = f[k] ** 2 * param2 + f_se[k] ** 2 * c_ik**2
        process[i] = np.sqrt(proc2)
        parameter[i] = np.sqrt(param2)
    mse = process**2 + parameter**2
    mack_se = np.sqrt(mse)

    # --- total over all origin periods ------------------------------------
    # The estimation errors of different origin periods share the same
    # estimated factors, so the total carries a covariance term:
    #   mse(total) = sum_i mse_i
    #              + sum_i Chat_{i,n} (sum_{j>i} Chat_{j,n}) sum_k 2 f.se_k^2/f_k^2
    total_mse = float(np.sum(mse))
    for i in range(m):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        if last >= n_full - 1:
            continue
        later = float(np.sum(ultimate[i + 1 :]))
        if later == 0.0:
            continue
        inner = 0.0
        for k in range(last, n_full - 1):
            inner += 2.0 * f_se[k] ** 2 / f[k] ** 2
        total_mse += float(ultimate[i]) * later * inner

    total_process = float(np.sqrt(np.sum(process**2)))
    total_parameter = float(np.sqrt(max(total_mse - total_process**2, 0.0)))

    return MackResult(
        triangle=tri,
        full_triangle=full,
        f=f,
        sigma=sigma,
        f_se=f_se,
        s=s,
        latest=latest,
        ultimate=ultimate,
        ibnr=ibnr,
        process_risk=process,
        parameter_risk=parameter,
        mack_se=mack_se,
        total_ibnr=float(np.nansum(ibnr)),
        total_process_risk=total_process,
        total_parameter_risk=total_parameter,
        total_mack_se=float(np.sqrt(total_mse)),
        est_sigma=est_sigma,
        alpha=alpha,
        tail=tail,
        has_tail=has_tail,
        sigma_source=source,
        n_obs=n_obs,
    )
