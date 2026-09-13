"""Mack's distribution-free chain ladder.

Reference
---------
Mack, T. (1993). Distribution-free calculation of the standard error of chain
ladder reserve estimates. *ASTIN Bulletin* 23(2), 213-225.

Mack, T. (1999). The standard error of chain ladder reserve estimates:
recursive calculation and inclusion of a tail factor. *ASTIN Bulletin* 29(2),
361-366.  The recursion of the 1999 paper is what is implemented here; for the
individual origin periods it is algebraically identical to the closed-form
mean squared error of the 1993 paper (see ``docs`` note in the report).

Model
-----
With C_{i,k} the cumulative claims of origin period i after k development
periods and F_{i,k} = C_{i,k+1} / C_{i,k}:

    (CL1)  E[F_{i,k} | C_{i,1}, ..., C_{i,k}]   = f_k
    (CL2)  Var(F_{i,k} | C_{i,1}, ..., C_{i,k}) = sigma_k^2 / C_{i,k}
    (CL3)  origin periods are independent

Estimators
----------
    f_k       = sum_i C_{i,k+1} / sum_i C_{i,k}                (volume weighted)
    sigma_k^2 = 1/(I_k - 1) * sum_i C_{i,k} (F_{i,k} - f_k)^2
    S_k       = sum_i C_{i,k}                                  (same rows as f_k)

Only the case alpha = 1, w_{i,k} = 1 of Mack's more general variance
assumption is implemented, which is the default of R's
``ChainLadder::MackChainLadder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .triangle import check_triangle, latest_diagonal

__all__ = ["MackResult", "mack_chain_ladder"]


@dataclass
class MackResult:
    """Output of :func:`mack_chain_ladder`."""

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


def _development_factors(tri: np.ndarray):
    """Volume weighted age-to-age factors, column sums and observation counts."""
    m, n = tri.shape
    f = np.full(n - 1, np.nan)
    s = np.full(n - 1, np.nan)
    n_obs = np.zeros(n - 1, dtype=int)
    for k in range(n - 1):
        rows = ~np.isnan(tri[:, k]) & ~np.isnan(tri[:, k + 1])
        n_obs[k] = int(rows.sum())
        if n_obs[k] == 0:
            continue
        s[k] = tri[rows, k].sum()
        f[k] = tri[rows, k + 1].sum() / s[k]
    return f, s, n_obs


def _sigma_from_data(tri: np.ndarray, f: np.ndarray):
    """sigma_k^2 for those k that have at least two observations."""
    n = tri.shape[1]
    sigma2 = np.full(n - 1, np.nan)
    for k in range(n - 1):
        rows = ~np.isnan(tri[:, k]) & ~np.isnan(tri[:, k + 1])
        i_k = int(rows.sum())
        if i_k < 2:
            continue
        c_k = tri[rows, k]
        ratio = tri[rows, k + 1] / c_k
        sigma2[k] = np.sum(c_k * (ratio - f[k]) ** 2) / (i_k - 1)
    return sigma2


def _extrapolate_sigma_mack(sigma2: np.ndarray) -> np.ndarray:
    """Mack's rule for the last development period.

    sigma_{n-1}^2 = min( sigma_{n-2}^4 / sigma_{n-3}^2,
                         min(sigma_{n-3}^2, sigma_{n-2}^2) )

    Applied to every trailing ``NaN``, working from left to right, so that a
    triangle missing more than one tail estimate is still handled.
    """
    out = sigma2.copy()
    for k in range(len(out)):
        if not np.isnan(out[k]):
            continue
        if k < 2 or np.isnan(out[k - 1]) or np.isnan(out[k - 2]):
            out[k] = 0.0
            continue
        a, b = out[k - 2], out[k - 1]  # sigma_{k-2}^2, sigma_{k-1}^2
        if a <= 0:
            out[k] = min(a, b)
        else:
            out[k] = min(b**2 / a, min(a, b))
    return out


def _extrapolate_sigma_loglinear(sigma2: np.ndarray) -> np.ndarray:
    """Log-linear extrapolation of sigma_k.

    Fits ``log(sigma_k) = a + b k`` by ordinary least squares on the periods
    that could be estimated from the data and extrapolates.  This mirrors the
    idea of ``est.sigma = "log-linear"`` in R's ChainLadder but is *not*
    verified against it: that implementation additionally inspects the
    significance of the fit and falls back to Mack's rule.  Use
    ``est_sigma="mack"`` when you need agreement with R.
    """
    out = sigma2.copy()
    known = np.flatnonzero(~np.isnan(out) & (out > 0))
    if known.size < 2:
        return _extrapolate_sigma_mack(sigma2)
    x = known.astype(float)
    y = np.log(np.sqrt(out[known]))
    b, a = np.polyfit(x, y, 1)
    missing = np.flatnonzero(np.isnan(out))
    out[missing] = np.exp(a + b * missing.astype(float)) ** 2
    return out


def mack_chain_ladder(triangle, est_sigma: str = "mack") -> MackResult:
    """Fit the Mack chain ladder to a cumulative run-off triangle.

    Parameters
    ----------
    triangle
        Array-like of shape ``(m, n)`` with ``NaN`` in the unobserved part.
    est_sigma
        How to obtain ``sigma`` for development periods with fewer than two
        observations: ``"mack"`` (Mack's extrapolation rule, the verified path)
        or ``"log-linear"``.

    Returns
    -------
    MackResult
    """
    tri = np.asarray(triangle, dtype=float)
    check_triangle(tri)
    m, n = tri.shape

    f, s, n_obs = _development_factors(tri)
    if np.isnan(f).any():
        raise ValueError("some development period has no observed link ratio")

    sigma2 = _sigma_from_data(tri, f)
    if est_sigma == "mack":
        sigma2 = _extrapolate_sigma_mack(sigma2)
    elif est_sigma in ("log-linear", "loglinear"):
        sigma2 = _extrapolate_sigma_loglinear(sigma2)
    else:
        raise ValueError("est_sigma must be 'mack' or 'log-linear'")
    sigma = np.sqrt(sigma2)

    # standard error of the estimated development factor: sigma_k / sqrt(S_k)
    f_se = sigma / np.sqrt(s)

    # --- projection of the lower right part ------------------------------
    full = tri.copy()
    for i in range(m):
        observed = np.flatnonzero(~np.isnan(tri[i]))
        last = int(observed[-1])
        for k in range(last, n - 1):
            full[i, k + 1] = full[i, k] * f[k]

    latest = latest_diagonal(tri)
    ultimate = full[:, n - 1]
    ibnr = ultimate - latest

    # --- prediction error, recursively (Mack 1999) ------------------------
    process = np.zeros(m)
    parameter = np.zeros(m)
    for i in range(m):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        proc2 = 0.0
        param2 = 0.0
        for k in range(last, n - 1):
            c_ik = full[i, k]
            proc2 = f[k] ** 2 * proc2 + sigma2[k] * c_ik
            param2 = f[k] ** 2 * param2 + f_se[k] ** 2 * c_ik**2
        process[i] = np.sqrt(proc2)
        parameter[i] = np.sqrt(param2)
    mse = process**2 + parameter**2
    mack_se = np.sqrt(mse)

    # --- total over all origin periods ------------------------------------
    # Mack (1993), s.e. of the overall reserve:
    #   mse(total) = sum_i mse_i
    #                + sum_i Chat_{i,n} * (sum_{j>i} Chat_{j,n})
    #                        * sum_k 2 sigma_k^2 / (f_k^2 S_k)
    total_mse = float(np.sum(mse))
    for i in range(m):
        last = int(np.flatnonzero(~np.isnan(tri[i]))[-1])
        if last >= n - 1:
            continue
        tail_sum = float(np.sum(ultimate[i + 1 :]))
        if tail_sum == 0.0:
            continue
        inner = 0.0
        for k in range(last, n - 1):
            inner += 2.0 * sigma2[k] / (f[k] ** 2 * s[k])
        total_mse += float(ultimate[i]) * tail_sum * inner

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
        n_obs=n_obs,
    )
