PYTHON ?= python3
RSCRIPT ?= Rscript
export PYTHONPATH := src

.PHONY: help install reference test test-informational verify run report docker-build docker-verify clean

help:
	@echo "install        install the package and its dependencies"
	@echo "reference      run ChainLadder in R and write reference/generated/"
	@echo "test           run the Python test suite"
	@echo "test-informational  run the comparisons that are reported, not enforced"
	@echo "verify         reference + test, and report skipped tests"
	@echo "run            run the estimate on data/raa.csv"
	@echo "report         render report/report.qmd"
	@echo "docker-build   build the container"
	@echo "docker-verify  run the full check inside the container"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

reference:
	$(RSCRIPT) R/export_reference.R reference/generated

# Deliberately the bare command rather than `python -m pytest`: the module
# form puts the working directory on sys.path, which hides an import that
# only works locally. CI runs the bare command, so this does too.
test:
	pytest tests -m "not informational"

test-informational:
	pytest tests -m informational || true

verify: reference test
	@echo
	@echo "If any test above was skipped, the R comparison did not run."

run:
	$(PYTHON) -m mackpy data/raa.csv --out out/raa_byorigin.csv \
		--full-triangle out/raa_full_triangle.csv

report:
	quarto render report/report.qmd

docker-build:
	docker build -t mackpy:latest .

docker-verify: docker-build
	docker run --rm mackpy:latest

clean:
	rm -rf out report/*.html report/*.pdf report/*_files reference/generated
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
