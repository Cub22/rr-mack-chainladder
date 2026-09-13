# Mack's chain ladder in Python, checked against R

Reproduction of an existing analysis in a different programming language.

**Source method.** Mack, T. (1993), "Distribution-free calculation of the
standard error of chain ladder reserve estimates", *ASTIN Bulletin* 23(2),
213–225, together with the recursive formulation of Mack (1999), *ASTIN
Bulletin* 29(2), 361–366.

**Reference implementation.** `MackChainLadder` from the R package
`ChainLadder`, on the sample run-off triangles `RAA` and `GenIns`.

**What is here.** The method reimplemented in Python from the formulae
(`src/mackpy/`), an R script that exports the reference numbers
(`R/export_reference.R`), a test suite that compares the two at full double
precision plus property tests that do not depend on R at all (`tests/`), a
container pinning both environments (`Dockerfile`), a CI workflow that runs
the whole comparison on every push (`.github/workflows/ci.yml`), and a Quarto
report with the code visible (`report/report.qmd`).

## Result

On `RAA` with `est.sigma = "Mack"`, the Python implementation reproduces the
published R output exactly at the precision the vignette prints:

| | R (`ChainLadder`) | Python (`mackpy`) |
|---|---|---|
| Total IBNR | 52,135.23 | 52,135.23 |
| Total Mack S.E. | 26,909.01 | 26,909.01 |
| Development factors | 2.999 1.624 1.271 1.172 1.113 1.042 1.033 1.017 1.009 | identical to 4 d.p. |

Every by-origin figure agrees as well; see `report/report.qmd` for the full
table and for the four places where the published method underdetermines the
answer and an implementation has to choose.

## Running it

Without R, the property tests and the comparison against the published
(rounded) output run on their own:

```sh
pip install -r requirements.txt
pip install -e . --no-deps
pytest tests -v
```

With R available, generate the reference and run the full comparison:

```sh
make verify        # Rscript R/export_reference.R && pytest tests
```

Or do both inside the container, which pins the R version and the CRAN
snapshot it installs from:

```sh
make docker-verify
```

A single estimate from the command line:

```sh
python -m mackpy data/raa.csv
python -m mackpy data/raa.csv --est-sigma log-linear --out out/raa.csv
```

## Layout

```
data/raa.csv                 RAA cumulative triangle (provenance in reference/README.md)
src/mackpy/mack.py           the method: factors, sigma, projection, mse
src/mackpy/triangle.py       triangle I/O, cumulative/incremental conversion
src/mackpy/cli.py            command line interface
R/export_reference.R         runs ChainLadder, writes reference/generated/*.csv
R/renv-setup.R               creates or restores a pinned R library
reference/                   published reference values, provenance, and notes
tests/test_raa_published.py  against the vignette output (rounded, tol 0.1)
tests/test_against_r.py      against locally generated R output (rtol 1e-8)
tests/test_properties.py     properties that hold for any triangle, no R needed
report/report.qmd            the report, code visible
Dockerfile, Makefile         pinned environment and entry points
```

## Notes on how reproducible this actually is

Three honest qualifications, since the point of the course is reproducibility
rather than the reserve number.

1. **`reference/generated/` is not committed.** Reference values that sit in a
   repository go stale silently against the package version that produced
   them. Here they are regenerated on every CI run and the R version and
   `ChainLadder` version are written to `SESSION.txt` alongside them. The
   trade-off is that the comparison needs R to be installed; when it is not,
   those tests skip, and the CI workflow has an explicit step that fails if
   the reference files are missing so that a skip cannot pass for a pass.

2. **`renv.lock` is generated, not shipped.** A lockfile has to come from a
   real resolution against CRAN. `R/renv-setup.R` produces one; run it once
   and commit the result. The Docker image pins the R side a different way,
   through the `rocker/r-ver` tag, which fixes both the R version and the CRAN
   snapshot that `install.packages()` reads.

3. **Scope.** Only the case `alpha = 1`, `weights = 1`, no tail factor is
   implemented — the R default. `est_sigma="mack"` is the setting under which
   agreement with R is claimed and tested; `est_sigma="log-linear"` implements
   the plain log-linear extrapolation and not R's significance check and
   fallback, so it is excluded from the comparison tests on purpose rather
   than left to fail quietly.

## Licence

MIT, see `LICENSE`.
