# Nabla's development entry points.
#
# `make ci` runs exactly what .github/workflows/ci.yml runs, in the same order,
# so a green local run means a green pull request. Keeping the two in sync by
# having the workflow call these targets would couple the CI matrix to the
# Makefile's idea of a Python; instead the workflow spells out the same four
# commands and this comment is the contract.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help install test lint fmt typecheck examples bench ci clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and the development extras, editable
	$(PIP) install -e ".[dev]"

test: ## Run the test suite, including the doctests in src/nabla
	$(PYTHON) -m pytest -q

lint: ## Check style and formatting without changing anything
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt: ## Format, and apply the lint fixes that are safe
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck: ## Type-check the package under mypy --strict
	$(PYTHON) -m mypy --strict src

examples: ## Run all three examples end to end (about 95 seconds)
	$(PYTHON) examples/mlp_moons/train.py
	$(PYTHON) examples/mnist_style/train.py
	$(PYTHON) examples/char_transformer/train.py

bench: ## Measure the tape's overhead against hand-written NumPy
	$(PYTHON) benchmarks/overhead.py

ci: lint typecheck test ## Everything CI runs, in the same order

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
