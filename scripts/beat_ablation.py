"""Beat ablation study: accuracy at 1 and 5 consecutive beats per decision.

Tests SVM baseline, LSTM, and CNN on the same frozen test split.
Beats are taken as the first N consecutive beats from each test recording.
Never mixes enrollment and probe beats.

Usage:
    PYTHONPATH=src uv run python scripts/beat_ablation.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlee.baseline import features as dct_features
from mlee.baseline import read_record, segment
from mlee.cnn_model import ECG_CNN
from mlee.lstm_model import ECG_LSTM


def load_beats_by_recording(data_dir):
    """Return {person: [(record_id, beats_array, feat_array), ...]} sorted by record number."""
    paths = sorted(data_dir.glob("**/Person_*/*.hea"))
    by_person = {}
    for p in paths:
        by_person.setdefault(p.parent.name, []).append(p)

    result = {}
    for person, records in sorted(by_person.items()):
        records.sort(key=lambda p: int(p.stem.split("_")[-1]))
        accepted = []
        for p in records:
            x, fs, _, _ = read_record(p)
            beats, _ = segment(x, fs)
            if len(beats) < 3:
                continue
            accepted.append((p.stem, beats, dct_features(beats)))
        if len(accepted) >= 2:
            result[person] = accepted
    return result


def ablation(n_beats, by_person, svm_model, lstm_model, cnn_model, lstm_scaler, classes, device):
    class_to_idx = {c: i for i, c in enumerate(classes)}

    svm_preds, lstm_preds, cnn_preds, true_labels = [], [], [], []

    for person, recordings in by_person.items():
        # Test recording = last one
        _, test_beats_raw, test_feat = recordings[-1]
        # Training = all but last (for enrollment templates, computed outside this loop)
        true_idx = class_to_idx.get(person)
        if true_idx is None:
            continue

        # Take first min(n_beats, available) beats from test recording
        k = min(n_beats, len(test_beats_raw))
        raw_k = test_beats_raw[:k]        # (k, 300)
        feat_k = test_feat[:k]            # (k, 32)

        # SVM: average feature vector → predict
        avg_feat = feat_k.mean(axis=0, keepdims=True)
        svm_pred = svm_model.predict(avg_feat)[0]
        svm_preds.append(class_to_idx.get(svm_pred, -1))

        # LSTM: scale → average logits
        feat_scaled = lstm_scaler.transform(feat_k)
        with torch.no_grad():
            logits = lstm_model(torch.tensor(feat_scaled, dtype=torch.float32).to(device))
            avg_logit = logits.mean(0)
        lstm_preds.append(avg_logit.argmax().item())

        # CNN: average logits over raw beats
        raw_t = torch.tensor(raw_k, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = cnn_model(raw_t)
            avg_logit = logits.mean(0)
        cnn_preds.append(avg_logit.argmax().item())

        true_labels.append(true_idx)

    true_labels = np.array(true_labels)
    svm_acc = float(np.mean(np.array(svm_preds) == true_labels))
    lstm_acc = float(np.mean(np.array(lstm_preds) == true_labels))
    cnn_acc = float(np.mean(np.array(cnn_preds) == true_labels))
    return {"svm": svm_acc, "lstm": lstm_acc, "cnn": cnn_acc}


def main():
    private = ROOT / "private"
    data_dir = ROOT / "data"

    # Load models
    import pickle as pkl
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # SVM: retrain from cached features (fast)
    tr = np.load(private / "features_train.npz", allow_pickle=True)
    te = np.load(private / "features_test.npz", allow_pickle=True)
    classes = tr["classes"].tolist()
    n_classes = len(classes)
    svm = make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale"))
    svm.fit(tr["X"], tr["y"])

    lstm_scaler = pkl.loads((private / "lstm_scaler.pkl").read_bytes())

    device = torch.device("cpu")
    lstm = ECG_LSTM(n_classes=n_classes, hidden=64)
    lstm.load_state_dict(torch.load(private / "lstm_fp32.pt", map_location="cpu"))
    lstm.eval()

    cnn = ECG_CNN(n_classes=n_classes)
    cnn.load_state_dict(torch.load(private / "cnn_fp32.pt", map_location="cpu"))
    cnn.eval()

    print("Loading ECG-ID recordings for ablation...")
    by_person = load_beats_by_recording(data_dir)

    results = {}
    for n in [1, 5]:
        r = ablation(n, by_person, svm, lstm, cnn, lstm_scaler, classes, device)
        results[n] = r
        print(f"\nN={n} beat(s) per decision:")
        print(f"  SVM:  {r['svm']*100:.2f}%")
        print(f"  LSTM: {r['lstm']*100:.2f}%")
        print(f"  CNN:  {r['cnn']*100:.2f}%")

    out = {
        "protocol": "First N consecutive beats from each test recording. Train beats used for enrollment only.",
        "n_beats_tested": [1, 5],
        "results": {str(k): v for k, v in results.items()},
    }
    (private / "beat_ablation.json").write_text(json.dumps(out, indent=2))
    print("\nSaved to private/beat_ablation.json")
    return out


if __name__ == "__main__":
    main()
