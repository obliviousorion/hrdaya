export UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export PYTHONPATH := $(CURDIR)/src
.PHONY: setup data baseline test lint figures leaderboard hooks
setup:
	uv sync --locked
hooks:
	uvx --from pre-commit==4.3.0 pre-commit install
data:
	uv run --locked python scripts/download_data.py
baseline:
	uv run --locked python scripts/run_experiment.py --config configs/baseline.yaml
lint:
	uvx ruff==0.17.1 check .
test:
	uv run --locked python -m unittest discover -s tests -v
figures:
	uv run --locked python scripts/make_figures.py
leaderboard:
	uv run --locked python scripts/leaderboard.py
