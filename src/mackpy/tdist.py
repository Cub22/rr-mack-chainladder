"""A small amount of statistics, implemented here to avoid a scipy dependency.

Only one thing is needed: the upper tail probability of a Student t
distribution, used to decide whether the slope of the log-linear sigma
regression is significantly different from zero.  The implementation is the
standard relation between the t distribution and the regularised incomplete
beta function,

    P(T_nu > t) = 0.5 * I_x(nu/2, 1/2),   x = nu / (nu + t^2),  t >= 0,

with ``I_x`` evaluated by the continued fraction of Lentz, as in Numerical
Recipes.  ``tests/test_tdist.py`` checks it against values that do not depend
on this code.
"""

from __future__ import annotations

import math

__all__ = ["t_sf", "betainc"]

_MAXIT = 300
_EPS = 3.0e-16
_FPMIN = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper tail probability P(T > t) of a Student t distribution."""
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(0.5 * df, 0.5, x)
    return tail if t >= 0 else 1.0 - tail
