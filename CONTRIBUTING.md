# Contributing

Claim an unowned TASKS.md item (and its GitHub issue, if created) before starting.
Leave other tasks alone; agree interface/split changes in the PR before dependent work.
Use a branch per task: `feat/<task>` for implementation/writing, `exp/<experiment>` for experiments.
Keep PRs small, target `main`, and require one teammate review before merge. Configure branch
protection with one approval when a remote exists. This local repo cannot enforce hosted rules.

Run `make setup`, `make hooks`, `make lint`, `make test`. CI uses synthetic ECG only.
Never commit data, secrets, personal planning, model binaries or generated plots. Hook guards
reject large files (>1 MB), strip notebook outputs and lint Python. Review `git diff --cached`
before requesting review. Large legitimate documents need an explicitly agreed policy change.

## Add an experiment

1. Claim a task and use a unique experiment name to avoid output collisions.
2. Copy `configs/baseline.yaml` to `<name>.yaml`. These are YAML 1.2 files written in its JSON
   subset, read with the standard library: keep valid JSON syntax. Set a fixed seed.
3. For a new model/protocol, implement it in `src/mlee/`, register it in `experiments.execute`,
   retain its validation and write config, run metadata, detailed evidence and compact metrics
   with the same schema. Current dispatcher deliberately rejects unsupported methods/settings.
   Frozen AC/DCT/filter/SVM settings remain in baseline.py; changing them needs a new method name.
4. Use `PYTHONPATH=src uv run --locked python scripts/run_experiment.py --config configs/<name>.yaml`.
   Never tune on test results. Use identical split manifests for comparable models; record
   date cohorts, exclusions, threshold selection, counts and all attempted seeds.
5. `make leaderboard` rebuilds the table of all completed runs. Commit only compact metrics,
   config and run metadata. Keep failed runs' metadata for diagnosis. Include baseline comparison,
   protocol, seed, command and limitations in the PR results field.
6. Use `make figures` for shared report/poster assets. Keep notebooks exploratory and outputs stripped.

Maintainer: after selecting a GitHub repo, `scripts/create_issues.sh --dry-run` previews tasks;
`scripts/create_issues.sh --repo OWNER/REPO` creates missing issues by exact task title.
The script is deliberately manual and does not assign owners, push, or create a remote.
