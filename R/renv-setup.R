#!/usr/bin/env Rscript
## Create (or restore) a pinned R library for this project.
##
##   Rscript R/renv-setup.R            # init + snapshot, writes renv.lock
##   Rscript R/renv-setup.R restore    # install exactly what renv.lock says
##
## renv.lock is deliberately NOT committed by whoever wrote the first version
## of this repository, because a lockfile contains package versions and
## repository URLs that must be produced by a real resolution against CRAN -
## writing one by hand would give a file that looks authoritative and is not.
## Generate it once with this script and commit the result; from then on
## `restore` reproduces that library exactly.

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args) >= 1) args[1] else "snapshot"

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

if (mode == "restore") {
  if (!file.exists("renv.lock")) {
    stop("renv.lock not found - run this script without arguments first")
  }
  renv::restore(prompt = FALSE)
} else {
  renv::init(bare = TRUE, restart = FALSE)
  install.packages("ChainLadder")
  renv::snapshot(packages = "ChainLadder", prompt = FALSE)
  cat("\nrenv.lock written. Commit it, and record the version below.\n")
  print(packageVersion("ChainLadder"))
}
