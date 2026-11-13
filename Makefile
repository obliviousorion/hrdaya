export UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export PYTHONPATH := $(CURDIR)/src
.PHONY: setup data baseline test lint figures leaderboard hooks models features ablation demo
setup:
	uv sync --locked
hooks:
	uvx --from pre-commit==4.3.0 pre-commit install
data:
	uv run python scripts/download_data.py
baseline:
	uv run python scripts/run_experiment.py --config configs/baseline.yaml
features:
	uv run python scripts/prepare_features.py
models:
	uv run python scripts/run_all_models.py
ablation:
	uv run python scripts/beat_ablation.py
demo:
	uv run python scripts/demo_replay.py
lint:
	uvx ruff==0.17.1 check .
test:
	uv run python -m unittest discover -s tests -v
figures:
	uv run python scripts/make_figures.py
leaderboard:
	uv run python scripts/leaderboard.py
