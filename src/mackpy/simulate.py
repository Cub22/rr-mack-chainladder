"""Simulated run-off triangles.

Property tests need triangles of many shapes, and a reader of the report may
want to see the method on data whose answer is known. Both are served by one
generator here rather than by a helper hidden in the test directory: a test
file importing from a sibling test file only works when the test directory
happens to be on ``sys.path``, which depends on how pytest was invoked.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_triangle", "deterministic_triangle"]


def simulate_triangle(
    n: int,
    seed: int = 0,
    shape: float = 4.0,
    scale: float = 250.0,
    decay: float = 0.05,
) -> np.ndarray:
    """A random but well-behaved cumulative triangle of size ``n``.

    Incremental amounts are drawn from a gamma distribution and damped
    linearly across development periods, so later periods contribute less, as
    in a real run-off. The lower right part is set to ``NaN``.

    Parameters
    ----------
    n
        Number of origin and development periods.
    seed
        Seed for ``numpy.random.default_rng``; the same seed always gives the
        same triangle.
    shape, scale
        Parameters of the gamma distribution of the incremental amounts.
    decay
        Weight of the last development period relative to the first.
    """
    if n < 2:
        raise ValueError("a triangle needs at least two development periods")
    rng = np.random.default_rng(seed)
    incr = rng.gamma(shape=shape, scale=scale, size=(n, n))
    incr *= np.linspace(1.0, decay, n)[None, :]
    cum = np.cumsum(incr, axis=1)
    for i in range(n):
        cum[i, n - i :] = np.nan
    return cum


def deterministic_triangle(factors) -> np.ndarray:
    """A triangle whose link ratios are exactly ``factors`` in every row.

    Useful as a degenerate case: every sigma is zero, so the standard error of
    every reserve must be zero too.
    """
    f = np.asarray(factors, dtype=float)
    n = f.size + 1
    cum = np.zeros((n, n))
    cum[:, 0] = np.arange(1, n + 1) * 100.0
    for k in range(n - 1):
        cum[:, k + 1] = cum[:, k] * f[k]
    for i in range(n):
        cum[i, n - i :] = np.nan
    return cum
