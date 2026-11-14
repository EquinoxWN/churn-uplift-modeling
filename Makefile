.PHONY: setup lint test download run bench audit ci

PY ?= python

setup:
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# Tests on small hand-written files and synthetic data (no download needed).
test:
	$(PY) -m pytest -q

# Download the Telco churn file (1 MB) and the Hillstrom campaign file (4 MB), SHA-256 checked.
download:
	cum download telco && cum download hillstrom

# Churn models raw and calibrated on Telco; S/T/X uplift learners on Hillstrom.
run: download
	cum churn && cum uplift

bench:
	@echo "M2/M3: Qini curves, uplift deciles and the campaign simulation are not built yet"

# Known vulnerabilities in the installed Python dependencies.
audit:
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit

ci: setup lint test
