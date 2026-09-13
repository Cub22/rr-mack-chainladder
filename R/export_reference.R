#!/usr/bin/env Rscript
## Export the output of ChainLadder::MackChainLadder so that the Python
## implementation can be compared against it at full precision.
##
## Writes, for each triangle and each est.sigma setting:
##   <name>_triangle.csv         the input triangle (cumulative, wide form)
##   <name>_<sigma>_byorigin.csv Latest / Ultimate / IBNR / Mack.S.E per origin
##   <name>_<sigma>_totals.csv   the same quantities for all origins combined
##   <name>_<sigma>_factors.csv  f, sigma, f.se per development period
##
## Usage:  Rscript R/export_reference.R [output_directory]

suppressPackageStartupMessages(library(ChainLadder))

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "reference/generated"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

write_triangle <- function(tri, name) {
  df <- as.data.frame(unclass(tri))
  colnames(df) <- seq_len(ncol(df))
  df <- cbind(origin = rownames(unclass(tri)), df)
  write.csv(df, file.path(outdir, paste0(name, "_triangle.csv")),
            row.names = FALSE, na = "")
}

export_one <- function(tri, name, est.sigma) {
  fit <- MackChainLadder(tri, est.sigma = est.sigma)
  tag <- paste0(name, "_", gsub("-", "", est.sigma))

  n <- ncol(fit$FullTriangle)
  latest <- getLatestCumulative(tri)
  ultimate <- fit$FullTriangle[, n]
  ibnr <- ultimate - latest
  se <- fit$Mack.S.E[, n]

  byorigin <- data.frame(
    origin = rownames(unclass(tri)),
    Latest = as.numeric(latest),
    Ultimate = as.numeric(ultimate),
    IBNR = as.numeric(ibnr),
    Mack.S.E = as.numeric(se),
    ProcessRisk = as.numeric(fit$Mack.ProcessRisk[, n]),
    ParameterRisk = as.numeric(fit$Mack.ParameterRisk[, n]),
    stringsAsFactors = FALSE
  )
  write.csv(byorigin, file.path(outdir, paste0(tag, "_byorigin.csv")),
            row.names = FALSE)

  totals <- data.frame(
    quantity = c("Latest", "Ultimate", "IBNR", "Mack.S.E",
                 "Total.ProcessRisk", "Total.ParameterRisk"),
    value = c(sum(latest, na.rm = TRUE),
              sum(ultimate, na.rm = TRUE),
              sum(ibnr, na.rm = TRUE),
              as.numeric(fit$Total.Mack.S.E),
              as.numeric(fit$Total.ProcessRisk[length(fit$Total.ProcessRisk)]),
              as.numeric(fit$Total.ParameterRisk[length(fit$Total.ParameterRisk)]))
  )
  write.csv(totals, file.path(outdir, paste0(tag, "_totals.csv")),
            row.names = FALSE)

  ## f and sigma have length n-1 for a square triangle; MackChainLadder
  ## appends a tail factor of 1, which is dropped here.
  keep <- seq_len(ncol(tri) - 1L)
  factors <- data.frame(
    dev = keep,
    f = as.numeric(fit$f)[keep],
    sigma = as.numeric(fit$sigma)[keep],
    f.se = as.numeric(fit$f.se)[keep]
  )
  write.csv(factors, file.path(outdir, paste0(tag, "_factors.csv")),
            row.names = FALSE)

  message(sprintf("%-22s IBNR = %14.4f   Mack.S.E = %14.4f",
                  tag, sum(ibnr, na.rm = TRUE), fit$Total.Mack.S.E))
}

data(RAA, envir = environment())
data(GenIns, envir = environment())

triangles <- list(raa = RAA, genins = GenIns)

for (name in names(triangles)) {
  write_triangle(triangles[[name]], name)
  for (es in c("Mack", "log-linear")) {
    export_one(triangles[[name]], name, es)
  }
}

## Record what produced these numbers, so a later disagreement can be traced.
info <- c(
  paste("generated:", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  paste("ChainLadder:", as.character(packageVersion("ChainLadder"))),
  paste("R:", R.version.string),
  paste("platform:", R.version$platform)
)
writeLines(info, file.path(outdir, "SESSION.txt"))
message("written to ", normalizePath(outdir))
