"""Compatibility entry point; records the same frozen baseline via its config."""
from mlee.experiments import ROOT, execute
from leaderboard import generate

if __name__ == "__main__":
    execute(ROOT / "configs/baseline.yaml")
    generate()
