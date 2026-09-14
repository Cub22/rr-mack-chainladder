"""A from-scratch Python implementation of Mack's distribution-free chain ladder."""

from .mack import MackResult, mack_chain_ladder
from .simulate import deterministic_triangle, simulate_triangle
from .triangle import (
    check_triangle,
    cum_to_incr,
    incr_to_cum,
    latest_diagonal,
    read_triangle_csv,
)

__version__ = "0.1.0"

__all__ = [
    "MackResult",
    "mack_chain_ladder",
    "simulate_triangle",
    "deterministic_triangle",
    "read_triangle_csv",
    "cum_to_incr",
    "incr_to_cum",
    "latest_diagonal",
    "check_triangle",
    "__version__",
]
