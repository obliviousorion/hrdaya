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
    if config["protocol"] != "last_usable_record":
        raise ValueError("Implement and register the requested method/protocol first")
    if config["method"] not in {"ac_dct_svm_frozen", "resnet_se_hybrid", "resnet_se_dualstream"}:
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
        if config["method"] == "ac_dct_svm_frozen":
            with contextlib.redirect_stdout(io.StringIO()):
                run(root / config["data"], output / "details.json")
            details = json.loads((output / "details.json").read_text())
            metrics = {k: v for k, v in details.items()
                       if k not in {"classes", "confusion_matrix", "split_manifest", "excluded"}}
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        elif config["method"] == "resnet_se_hybrid":
            from mlee.train_hybrid import train_hybrid
            from prepare_features import main as prep_feat
            priv = root / "private"
            if not (priv / "features_train.npz").exists():
                prep_feat(root / config["data"], priv)
            tr = np.load(priv / "features_train.npz", allow_pickle=True)
            te = np.load(priv / "features_test.npz", allow_pickle=True)
            metrics, _, _ = train_hybrid(
                tr["X_raw"], tr["y"], te["X_raw"], te["y"], tr["classes"].tolist(),
                epochs=45, seed=config["seed"], output_dir=output
            )
        elif config["method"] == "resnet_se_dualstream":
            from mlee.train_dualstream import train_dualstream
            from prepare_features import main as prep_feat
            priv = root / "private"
            if not (priv / "features_train.npz").exists():
                prep_feat(root / config["data"], priv)
            tr = np.load(priv / "features_train.npz", allow_pickle=True)
            te = np.load(priv / "features_test.npz", allow_pickle=True)
            metrics, _, _ = train_dualstream(
                tr["X_raw"], tr["y"], te["X_raw"], te["y"], tr["classes"].tolist(),
                epochs=50, seed=config["seed"], output_dir=output
            )
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return output
