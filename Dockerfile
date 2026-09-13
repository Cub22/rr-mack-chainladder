# Both environments in one image: R to produce the reference numbers,
# Python to reproduce them.
#
# rocker/r-ver tags pin the R version *and* the CRAN snapshot that
# install.packages() reads from, so the ChainLadder version installed here is
# fixed by the base image tag rather than by the day you happen to build.
# The exact version that ends up in the image is recorded by
# R/export_reference.R in reference/generated/SESSION.txt.
FROM rocker/r-ver:4.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        make \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- R side -----------------------------------------------------------------
# systemfit and Matrix are ChainLadder dependencies for the multivariate part;
# they are not needed for MackChainLadder but keep the install from warning.
RUN Rscript -e 'install.packages("ChainLadder"); \
                if (!requireNamespace("ChainLadder", quietly = TRUE)) quit(status = 1)'

# --- Python side ------------------------------------------------------------
WORKDIR /project
COPY requirements.txt ./
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

COPY . .
RUN pip install -e . --no-deps

# Default: generate the R reference, then check Python against it.
CMD ["make", "verify"]
