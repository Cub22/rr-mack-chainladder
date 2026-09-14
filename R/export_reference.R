#!/usr/bin/env Rscript
## Export the output of ChainLadder::MackChainLadder so that the Python
## implementation can be compared against it at full precision.
##
## Every square triangle shipped with ChainLadder that can be fitted is
## exported, for each of the three values of alpha, plus the log-linear sigma
## setting at alpha = 1. A dataset that is missing from the installed version,
## is not square, or that MackChainLadder refuses to fit is skipped with a
## message rather than stopping the run: the point is to compare on as many
## triangles as the installed package happens to provide, not to depend on a
## particular one being there.
##
## Files written per case, with tag = <name>_a<alpha>_<sigma>:
##   <name>_triangle.csv    the input triangle (cumulative, wide form)
##   <tag>_byorigin.csv     Latest / Ultimate / IBNR / Mack.S.E and the split
##   <tag>_totals.csv       the same quantities for all origins combined
##   <tag>_factors.csv      f, sigma, f.se per development period
##
## Usage:  Rscript R/export_reference.R [output_directory]

suppressPackageStartupMessages(library(ChainLadder))

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "reference/generated"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

CANDIDATES <- c("RAA", "GenIns", "ABC", "UKMotor", "MW2008", "MW2014", "Mortgage")
ALPHAS <- c(0, 1, 2)

write_triangle <- function(tri, name) {
  m <- unclass(tri)
  df <- as.data.frame(m)
  colnames(df) <- seq_len(ncol(df))
  df <- cbind(origin = rownames(m), df)
  write.csv(df, file.path(outdir, paste0(name, "_triangle.csv")),
            row.names = FALSE, na = "")
}

export_one <- function(tri, name, alpha, est.sigma) {
  tag <- paste0(name, "_a", alpha, "_", gsub("-", "", est.sigma))
  fit <- MackChainLadder(tri, alpha = alpha, est.sigma = est.sigma)

  n <- ncol(fit$FullTriangle)
  latest <- getLatestCumulative(tri)
  ultimate <- fit$FullTriangle[, n]
  ibnr <- ultimate - latest

  byorigin <- data.frame(
    origin = rownames(unclass(tri)),
    Latest = as.numeric(latest),
    Ultimate = as.numeric(ultimate),
    IBNR = as.numeric(ibnr),
    Mack.S.E = as.numeric(fit$Mack.S.E[, n]),
    ProcessRisk = as.numeric(fit$Mack.ProcessRisk[, n]),
    ParameterRisk = as.numeric(fit$Mack.ParameterRisk[, n]),
    stringsAsFactors = FALSE
  )
  write.csv(byorigin, file.path(outdir, paste0(tag, "_byorigin.csv")),
            row.names = FALSE)

  last <- function(x) as.numeric(x)[length(x)]
  totals <- data.frame(
    quantity = c("Latest", "Ultimate", "IBNR", "Mack.S.E",
                 "Total.ProcessRisk", "Total.ParameterRisk"),
    value = c(sum(latest, na.rm = TRUE),
              sum(ultimate, na.rm = TRUE),
              sum(ibnr, na.rm = TRUE),
              as.numeric(fit$Total.Mack.S.E),
              last(fit$Total.ProcessRisk),
              last(fit$Total.ParameterRisk))
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

  message(sprintf("  %-28s IBNR = %16.4f   Mack.S.E = %14.4f",
                  tag, sum(ibnr, na.rm = TRUE), fit$Total.Mack.S.E))
  invisible(TRUE)
}

usable <- function(tri) {
  m <- unclass(tri)
  is.matrix(m) && nrow(m) == ncol(m) && nrow(m) >= 4 && all(m[!is.na(m)] > 0)
}

exported <- character(0)
for (name in CANDIDATES) {
  tri <- tryCatch({
    data(list = name, package = "ChainLadder", envir = environment())
    get(name, envir = environment())
  }, error = function(e) NULL, warning = function(w) NULL)

  if (is.null(tri)) {
    message(name, ": not available in ChainLadder ",
            as.character(packageVersion("ChainLadder")), " - skipped")
    next
  }
  if (!usable(tri)) {
    message(name, ": not a square, strictly positive triangle - skipped")
    next
  }

  message(name, ":")
  wrote_any <- FALSE
  for (alpha in ALPHAS) {
    ok <- tryCatch(export_one(tri, name, alpha, "Mack"),
                   error = function(e) {
                     message("  alpha=", alpha, " failed: ", conditionMessage(e))
                     FALSE
                   })
    wrote_any <- wrote_any || isTRUE(ok)
  }
  ## Informational only: the Python side does not claim to reproduce R's
  ## log-linear fallback logic, and the test that reads this is allowed to fail.
  tryCatch(export_one(tri, name, 1, "log-linear"),
           error = function(e) message("  log-linear failed: ",
                                       conditionMessage(e)))
  if (wrote_any) {
    write_triangle(tri, name)
    exported <- c(exported, name)
  }
}

if (length(exported) == 0) {
  stop("no triangle could be exported - the comparison would be vacuous")
}

info <- c(
  paste("generated:", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  paste("ChainLadder:", as.character(packageVersion("ChainLadder"))),
  paste("R:", R.version.string),
  paste("platform:", R.version$platform),
  paste("triangles:", paste(exported, collapse = ", ")),
  paste("alphas:", paste(ALPHAS, collapse = ", "))
)
writeLines(info, file.path(outdir, "SESSION.txt"))
message("\nwritten to ", normalizePath(outdir), ": ",
        length(exported), " triangle(s)")
