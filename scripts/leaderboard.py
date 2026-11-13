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
        run_file = path.parent / "run.json"
        if run_file.exists():
            meta = json.loads(run_file.read_text())
            if meta.get("status") != "complete":
                continue
            git_val = meta.get("git_hash") or "uncommitted"
            seed_val = meta.get("seed", 42)
        else:
            git_val = "recorded"
            seed_val = 42

        m = json.loads(path.read_text())
        run = path.parent.relative_to(root / "results")
        acc = m.get("identification_accuracy", 0.0) * 100
        macro = m.get("macro_identity_accuracy", 0.0) * 100
        eer = m.get("verification_eer", 0.0) * 100
        lines.append(f"| [{run}]({run}/metrics.json) | {acc:.2f} | {macro:.2f} | {eer:.2f} | {seed_val} | {git_val} |")
    (root / "results/LEADERBOARD.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    generate()
