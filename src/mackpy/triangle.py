"""Reading and basic manipulation of run-off triangles.

A triangle is stored as a ``numpy`` array of shape ``(m, n)`` with ``np.nan``
in the unobserved (lower right) part.  Row ``i`` is an origin period, column
``k`` a development period, and the value is the *cumulative* claims amount
C_{i,k}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "read_triangle_csv",
    "cum_to_incr",
    "incr_to_cum",
    "latest_diagonal",
    "check_triangle",
]


def read_triangle_csv(path: str | Path, origin_col: str = "origin") -> pd.DataFrame:
    """Read a triangle stored in wide form.

    The first column holds the origin period labels, the remaining columns the
    development periods.  Empty cells are read as ``NaN``.
    """
    df = pd.read_csv(path)
    if origin_col not in df.columns:
        raise ValueError(f"column {origin_col!r} not found in {path}")
    df = df.set_index(origin_col)
    df.columns = [str(c).strip() for c in df.columns]
    return df.astype(float)


def check_triangle(tri: np.ndarray) -> None:
    """Raise if ``tri`` is not a usable run-off triangle.

    Requirements: at least two development periods, the observed cells form an
    upper-left triangular pattern (no holes inside a row), and all observed
    values are finite.
    """
    arr = np.asarray(tri, dtype=float)
    if arr.ndim != 2:
        raise ValueError("triangle must be two-dimensional")
    m, n = arr.shape
    if n < 2:
        raise ValueError("triangle needs at least two development periods")
    for i in range(m):
        observed = ~np.isnan(arr[i])
        if not observed.any():
            raise ValueError(f"origin row {i} is empty")
        last = int(np.max(np.flatnonzero(observed)))
        if not observed[: last + 1].all():
            raise ValueError(f"origin row {i} has a gap in its development")


def cum_to_incr(tri: np.ndarray) -> np.ndarray:
    """Cumulative to incremental."""
    arr = np.asarray(tri, dtype=float).copy()
    out = arr.copy()
    out[:, 1:] = arr[:, 1:] - arr[:, :-1]
    return out


def incr_to_cum(tri: np.ndarray) -> np.ndarray:
    """Incremental to cumulative, keeping ``NaN`` where the input is missing."""
    arr = np.asarray(tri, dtype=float)
    out = np.nancumsum(arr, axis=1)
    out[np.isnan(arr)] = np.nan
    return out


def latest_diagonal(tri: np.ndarray) -> np.ndarray:
    """Most recent observed cumulative amount for each origin period."""
    arr = np.asarray(tri, dtype=float)
    out = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        observed = np.flatnonzero(~np.isnan(arr[i]))
        if observed.size:
            out[i] = arr[i, observed[-1]]
    return out
