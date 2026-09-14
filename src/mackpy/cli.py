"""Command line interface.

    python -m mackpy data/raa.csv
    python -m mackpy data/raa.csv --est-sigma log-linear --out out/raa.csv
    python -m mackpy data/raa.csv --alpha 0 --tail 1.05 --tail-se 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .mack import mack_chain_ladder
from .triangle import read_triangle_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mackpy",
        description="Mack chain ladder reserve estimate and standard error.",
    )
    parser.add_argument("triangle", type=Path, help="CSV with a cumulative triangle")
    parser.add_argument(
        "--est-sigma",
        default="mack",
        choices=["mack", "log-linear"],
        help="how to estimate sigma for the last development period",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="exponent of the variance assumption (1 = volume weighted)",
    )
    parser.add_argument(
        "--tail", type=float, default=1.0, help="tail factor beyond the last period"
    )
    parser.add_argument(
        "--tail-se",
        type=float,
        default=0.0,
        help="standard error of the tail factor (0 treats it as known)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write the by-origin table here"
    )
    parser.add_argument(
        "--full-triangle",
        type=Path,
        default=None,
        help="write the completed triangle here",
    )
    parser.add_argument("--digits", type=int, default=2, help="digits when printing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tri = read_triangle_csv(args.triangle)
    res = mack_chain_ladder(
        tri.to_numpy(),
        est_sigma=args.est_sigma,
        alpha=args.alpha,
        tail=args.tail,
        tail_se=args.tail_se,
    )

    byorigin = res.summary()
    byorigin.index = tri.index
    totals = pd.Series(res.totals(), name="Totals")

    pd.set_option("display.width", 160)
    header = f"Mack chain ladder, est.sigma = {args.est_sigma}, alpha = {args.alpha}"
    if res.has_tail:
        header += f", tail = {args.tail}"
        if args.tail_se:
            header += f" (s.e. {args.tail_se})"
    print(header + "\n")
    print(byorigin.round(args.digits).to_string())
    print("\nTotals")
    print(totals.round(args.digits).to_string())

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        byorigin.to_csv(args.out)
        print(f"\nby-origin table written to {args.out}")
    if args.full_triangle is not None:
        args.full_triangle.parent.mkdir(parents=True, exist_ok=True)
        full = pd.DataFrame(
            res.full_triangle, index=tri.index, columns=tri.columns
        )
        full.to_csv(args.full_triangle)
        print(f"completed triangle written to {args.full_triangle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
