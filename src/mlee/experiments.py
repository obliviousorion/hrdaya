"""Run bookkeeping shared by experiment entry points."""
import contextlib
import io
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlee.baseline import run

ROOT = Path(__file__).resolve().parents[2]


def execute(config_path, root=ROOT):
    config = json.loads(Path(config_path).read_text())
    if set(config) != {"experiment", "seed", "data", "method", "protocol"}:
        raise ValueError("Unexpected config fields; see configs/baseline.yaml")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", config["experiment"]):
        raise ValueError("Invalid experiment name")
    if config["method"] != "ac_dct_svm_frozen" or config["protocol"] != "last_usable_record":
        raise ValueError("Implement and register the requested method/protocol first")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / "results" / config["experiment"] / stamp
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    source_files = sorted((root / "src").rglob("*.py")) + [root / "uv.lock"]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_files if p.is_file()}
    # HEAD is explicitly null before the first signed commit; never fabricate a hash.
    metadata = {"timestamp_utc": stamp, "git_hash": git("rev-parse", "HEAD"),
                "git_tree": git("write-tree"), "dirty": bool(git("status", "--porcelain")),
                "seed": config["seed"], "source_sha256": source_hashes, "status": "running"}
    metadata_file = output / "run.json"
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(root / config["data"], output / "details.json")
        details = json.loads((output / "details.json").read_text())
        metrics = {k: v for k, v in details.items()
                   if k not in {"classes", "confusion_matrix", "split_manifest", "excluded"}}
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return output
