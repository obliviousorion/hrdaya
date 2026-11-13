"""Offline ACCEPT/REJECT demo — the running prototype deliverable.

Loads the INT8 CNN + enrollment templates and replays held-out test recordings.
Threshold: cosine similarity > 0.5 (heuristic, not validation-set tuned — stated explicitly).

Usage:
    PYTHONPATH=src uv run python scripts/demo_replay.py [--person Person_01] [--model cnn|lstm]

Without --person: replays all test recordings.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlee.baseline import features as dct_features
from mlee.baseline import read_record, segment
from mlee.cnn_model import ECG_CNN
from mlee.lstm_model import ECG_LSTM

THRESHOLD = 0.5  # heuristic cosine threshold — not validation-set tuned


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def build_enrollment_templates(model, X_train, y_train, classes, use_raw, scaler, device):
    model.eval()
    if scaler is not None:
        X_train = scaler.transform(X_train)
    with torch.no_grad():
        emb = model.embed(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
    templates = {}
    for i, cls in enumerate(classes):
        mask = y_train == i
        if mask.any():
            templates[cls] = emb[mask].mean(axis=0)
    return templates


def replay(person, test_recording_path, templates, model, use_raw, scaler, device):
    x, fs, _, _ = read_record(test_recording_path)
    beats, _ = segment(x, fs)
    if len(beats) == 0:
        return None, None, 0

    if use_raw:
        inp = torch.tensor(beats.astype(np.float32)).to(device)
    else:
        feat = dct_features(beats)
        if scaler:
            feat = scaler.transform(feat)
        inp = torch.tensor(feat.astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        emb = model.embed(inp).cpu().numpy().mean(axis=0)

    if person not in templates:
        return "REJECT", None, len(beats)

    sim = cosine(emb, templates[person])
    decision = "ACCEPT" if sim >= THRESHOLD else "REJECT"
    return decision, sim, len(beats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default=None, help="e.g. Person_01; omit for all")
    parser.add_argument("--model", default="champion", choices=["champion", "cnn", "lstm"])
    args = parser.parse_args()

    private = ROOT / "private"
    tr = np.load(private / "features_train.npz", allow_pickle=True)
    te = np.load(private / "features_test.npz", allow_pickle=True)
    classes = tr["classes"].tolist()
    n_classes = len(classes)
    device = torch.device("cpu")

    scaler = None

    if args.model == "champion":
        use_raw = True
        model = torch.load(private / "resnet_se_hybrid_int8_model.pt", map_location="cpu", weights_only=False)
        X_train = tr["X_raw"]
    elif args.model == "cnn":
        use_raw = True
        int8_path = private / "cnn_int8_model.pt"
        if int8_path.exists():
            model = torch.load(int8_path, map_location="cpu", weights_only=False)
        else:
            model = ECG_CNN(n_classes=n_classes)
            model.load_state_dict(torch.load(private / "cnn_fp32.pt", map_location="cpu", weights_only=True))
        X_train = tr["X_raw"]
    else:
        use_raw = False
        int8_path = private / "lstm_int8_model.pt"
        if int8_path.exists():
            model = torch.load(int8_path, map_location="cpu", weights_only=False)
        else:
            model = ECG_LSTM(n_classes=n_classes)
            model.load_state_dict(torch.load(private / "lstm_fp32.pt", map_location="cpu", weights_only=True))
        scaler = pickle.loads((private / "lstm_scaler.pkl").read_bytes())
        X_train = tr["X"]

    model.eval()
    y_train = tr["y"]

    print(f"Building enrollment templates ({args.model.upper()})...")
    templates = build_enrollment_templates(model, X_train, y_train, classes, use_raw, scaler, device)

    # Find test recordings (last record per person in ECG-ID)
    data_dir = ROOT / "data"
    paths = sorted(data_dir.glob("**/Person_*/*.hea"))
    by_person = {}
    for p in paths:
        by_person.setdefault(p.parent.name, []).append(p)

    targets = {}
    for person, records in sorted(by_person.items()):
        records.sort(key=lambda p: int(p.stem.split("_")[-1]))
        # Collect accepted recordings
        accepted = []
        for rec in records:
            x, fs, _, _ = read_record(rec)
            beats, _ = segment(x, fs)
            if len(beats) >= 3:
                accepted.append(rec)
        if len(accepted) >= 2:
            targets[person] = accepted[-1]  # last usable = test

    if args.person:
        if args.person not in targets:
            print(f"Person {args.person} not found or has no test recording.")
            sys.exit(1)
        targets = {args.person: targets[args.person]}

    print(f"\nThreshold: cosine ≥ {THRESHOLD} → ACCEPT  (heuristic, not validation-tuned)")
    print(f"{'Person':<15} {'Decision':<10} {'Similarity':>12} {'Beats':>7}")
    print("-" * 50)

    correct = 0
    total = 0
    for person, rec_path in sorted(targets.items()):
        decision, sim, n_beats = replay(person, rec_path, templates, model, use_raw, scaler, device)
        sim_str = f"{sim:.4f}" if sim is not None else "N/A"
        mark = "✓" if decision == "ACCEPT" else "✗"
        print(f"{person:<15} {decision:<10} {sim_str:>12}  {n_beats:>5}   {mark}")
        if decision == "ACCEPT":
            correct += 1
        total += 1

    if total > 0:
        print(f"\nACCEPT rate (genuine): {correct}/{total} = {correct/total*100:.1f}%")
        print(f"(All replays use the person's own claimed identity — genuine trials only)")


if __name__ == "__main__":
    main()
