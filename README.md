# Your Heartbeat Is Your Password

Lightweight Deep ECG Biometrics for Edge Authentication — EEE G513.
Report, A3 colour poster and running prototype due **Friday 20 November 2026**;
content freeze **18 November**.

## Quickstart (Python 3.12, uv, Git and Make)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then from the clone:

```sh
make setup
make data
make test
make baseline
make hooks
```

Setup uses `uv sync --locked`. The baseline needs only NumPy, SciPy and scikit-learn.
Download is anonymous (~13 MB), checksum verified, and always kept outside Git.
With a working network and a typical laptop, allow about 10 minutes for setup and baseline;
network speed and first-time Python installation may add time. `make hooks` installs the
pre-commit hook (requires network on first use). `make lint` runs Ruff.

## Current evidence

AC/DCT(32) + scaled RBF SVM, C=10: **79.06% rank-1**, **79.04% macro accuracy**,
**13.11% verification EER**; 89 identities, 5,587 training and 2,288 test beats.
The last usable recording per identity is held out. This is recording-disjoint;
recording dates have not been validated, so **cross-day performance is not established**.
Cosine enrollment templates supply verification scores; test EER is descriptive,
not a validated deployment threshold. `results/baseline.json` is the untouched original evidence.
See [the generated leaderboard](results/LEADERBOARD.md) for all new runs.

## Repository map

| Path | Purpose |
|---|---|
| `src/mlee/` | Frozen pipeline and reproducible run recorder |
| `configs/` | One YAML experiment config per experiment (JSON-compatible YAML) |
| `scripts/` | Download/checksums, run experiments, leaderboard, figures, issue creation |
| `tests/` | Tiny deterministic synthetic ECG integration and integrity checks |
| `notebooks/` | Exploration only; stripped outputs, no benchmark authority |
| `results/` | Small metrics, run metadata/configs and original reference JSON |
| `report/`, `poster/` | Report skeleton, A3 poster scaffold and shared generated figures |
| `sources/`, proposal PDF | Supplied course requirements/templates and project proposal |
| `private/`, `data/` | Ignored local material; never shared |

`make figures` creates ignored SVG figures under `report/figures/` using the latest completed
baseline run (or pass `--run results/<experiment>/<timestamp>` directly to the script).
Detailed confusion/split evidence stays in ignored `details.json`; compact metrics/config/provenance
are tracked. Every run records UTC time, seed, Git HEAD, staged tree and dirty state.
Before the first signed commit HEAD is null and the leaderboard says `uncommitted`.
No model is saved by the classical diagnostic pipeline.

## Pick up work

Choose one unowned item in [TASKS.md](TASKS.md), claim its `Owner:` field via a small PR,
and branch `feat/<task>` or `exp/<task>`. Coordinate shared split manifests before modeling.
Start with reader validation and date split auditing; CNN/LSTM work can proceed on a separate branch.
See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment and review rules.

## Data and code licences

Code is MIT licensed. Supplied course/proposal documents retain their original ownership;
the code licence does not relicense these documents or ECG-ID.
[ECG-ID v1.0.0](https://physionet.org/content/ecgiddb/1.0.0/) (Tatiana Lugovaya,
DOI: 10.13026/C2J01F) is **ODC-By 1.0**, [licence](https://physionet.org/content/ecgiddb/view-license/1.0.0/).
Cite Lugovaya T.S., *Biometric human identification based on electrocardiogram*,
master's thesis, LETI, Saint-Petersburg, June 2005; cite ECG-ID and PhysioNet with results.
PhysioNet's current standard citation: Pollard et al., *PhysioNet as a global platform for
biomedical research*, Nature Health (2026), doi:10.1038/s44360-026-00096-z.
The downloader preserves dataset notices and verifies the pinned official archive and all
999 manifest entries. Never commit ECG data, including generated copies in notebooks.

## Local initialization handoff

This workspace blocked `.git` creation (`Operation not permitted`). Run
`scripts/init_repo.sh` in a local terminal to initialize `main` and stage the files.
It leaves everything uncommitted for the hub's signed commit and adds no remote.
Local setup/data verification and synthetic tests work; fresh dependency downloads and
hook installation could not be exercised here because network access is blocked.
