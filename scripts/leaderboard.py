"""Generate a table from completed run metrics, without selecting a best seed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(root=ROOT):
    lines = ["# Experiment leaderboard", "",
             "Recording-disjoint baseline is not a cross-day result. Test EER is descriptive.", "",
             "| Experiment / UTC run | Rank-1 (%) | Macro (%) | EER (%) | Seed | Git |",
             "|---|---:|---:|---:|---:|---|"]
    for path in sorted((root / "results").glob("*/*/metrics.json")):
        meta = json.loads((path.parent / "run.json").read_text())
        if meta["status"] != "complete":
            continue
        m = json.loads(path.read_text())
        run = path.parent.relative_to(root / "results")
        lines.append(f"| [{run}]({run}/metrics.json) | {m['identification_accuracy']*100:.2f} "
                     f"| {m['macro_identity_accuracy']*100:.2f} | {m['verification_eer']*100:.2f} "
                     f"| {meta['seed']} | {meta['git_hash'] or 'uncommitted'} |")
    (root / "results/LEADERBOARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    generate()
