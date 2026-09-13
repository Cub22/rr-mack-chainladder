# Reference values

Two kinds of reference material are used in this project, and the distinction
matters for how much weight each one carries.

## 1. Published output (checked into the repository)

`raa_published_byorigin.csv` and `raa_published_totals.csv` were transcribed
by hand from the printed output of

```r
MackChainLadder(RAA, est.sigma = "Mack")
```

as shown in the package vignette *ChainLadder: Claims reserving with R*
(Gesmann, Murphy, Zhang, Carrato, Wüthrich, Concina, Dal Moro), section
"Mack chain-ladder", available at
<https://mages.github.io/ChainLadder/articles/ChainLadder.html>.

The same numbers appear in the CRAN copy of the vignette. The figures are
printed rounded (`Mack.S.E` to one decimal by origin, totals to two), so the
tests that use them apply a tolerance of 0.1 rather than asking for bit-level
agreement. Transcribed values are only as reliable as the transcription, which
is why they are not the primary check.

`data/raa.csv` is the `RAA` cumulative triangle, transcribed from the same
vignette. The triangle is originally from the Reinsurance Association of
America and is discussed in, among others, England & Verrall (2002).

## 2. Locally generated output (not checked in)

The primary check is `R/export_reference.R`, which runs `ChainLadder` in the
current environment and writes its results to `reference/generated/`. The
tests in `tests/test_against_r.py` read those files and compare them to the
Python implementation at full double precision (relative tolerance 1e-8).

`reference/generated/` is git-ignored on purpose: a reference value that is
committed can silently go stale against the package version it came from,
whereas one that is regenerated on every CI run cannot. Run

```sh
make reference
```

or, inside the container,

```sh
Rscript R/export_reference.R
```

If the directory is empty the R-comparison tests skip rather than fail, so the
suite is still usable without R installed. A skipped test is not a passed
test: `make verify` reports how many of each you got.
