"""Run one named experiment and refresh the shared leaderboard."""
import argparse
from mlee.experiments import execute
from leaderboard import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    output = execute(args.config)
    generate()
    print(f"Saved run: {output}")
