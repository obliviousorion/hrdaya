"""Extract AC/DCT(32) features and raw beat windows from ECG-ID.

Saves to private/ (gitignored):
  features_train.npz  - X (N,32) AC/DCT, X_raw (N,300) beats, y (N,) labels, classes
  features_test.npz   - same keys for test split

Uses identical split logic to baseline.run(): last usable record per person → test.
Run after: make data
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlee.baseline import features, read_record, segment


def main(data_dir=None, out_dir=None):
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    out_dir = Path(out_dir) if out_dir else ROOT / "private"
    out_dir.mkdir(exist_ok=True)

    paths = sorted(data_dir.glob("**/Person_*/*.hea"))
    if not paths:
        raise ValueError("No ECG-ID headers found. Run make data first.")

    by_person = {}
    for p in paths:
        by_person.setdefault(p.parent.name, []).append(p)

    train_X, train_X_raw, train_y = [], [], []
    test_X, test_X_raw, test_y = [], [], []
    excluded = []
    classes = []

    for person, records in sorted(by_person.items()):
        records.sort(key=lambda p: int(p.stem.split("_")[-1]))
        accepted = []
        for p in records:
            x, fs, _, _ = read_record(p)
            beats, _ = segment(x, fs)
            if len(beats) < 3:
                excluded.append({"record": str(p), "reason": "< 3 beats"})
                continue
            accepted.append((features(beats), beats))

        if len(accepted) < 2:
            excluded.append({"person": person, "reason": "< 2 usable records"})
            continue

        classes.append(person)
        for idx, (feat, raw_beats) in enumerate(accepted):
            is_test = idx == len(accepted) - 1
            if is_test:
                test_X.extend(feat)
                test_X_raw.extend(raw_beats)
                test_y.extend([person] * len(feat))
            else:
                train_X.extend(feat)
                train_X_raw.extend(raw_beats)
                train_y.extend([person] * len(feat))

    classes = sorted(set(train_y))
    label_map = {c: i for i, c in enumerate(classes)}

    train_y_int = np.array([label_map[y] for y in train_y])
    test_y_int = np.array([label_map[y] for y in test_y])

    np.savez(
        out_dir / "features_train.npz",
        X=np.array(train_X, dtype=np.float32),
        X_raw=np.array(train_X_raw, dtype=np.float32),
        y=train_y_int,
        classes=np.array(classes),
    )
    np.savez(
        out_dir / "features_test.npz",
        X=np.array(test_X, dtype=np.float32),
        X_raw=np.array(test_X_raw, dtype=np.float32),
        y=test_y_int,
        classes=np.array(classes),
    )

    info = {
        "train_beats": len(train_y),
        "test_beats": len(test_y),
        "n_classes": len(classes),
        "excluded": excluded,
    }
    (out_dir / "features_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
