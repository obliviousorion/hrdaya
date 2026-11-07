"""Reject unexpectedly large staged files; pre-commit passes file paths."""
import sys
from pathlib import Path


def oversized(paths, limit=1_000_000):
    return [str(p) for p in map(Path, paths) if p.is_file() and p.stat().st_size > limit]


if __name__ == "__main__":
    bad = oversized(sys.argv[1:])
    if bad:
        print("Files exceed 1 MB; keep data and binaries out of Git: " + ", ".join(bad))
        raise SystemExit(1)
