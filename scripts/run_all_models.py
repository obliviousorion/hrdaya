"""Run all three models and produce the full comparison metrics.

Usage:
    PYTHONPATH=src uv run python scripts/run_all_models.py

Outputs (all to private/ — gitignored):
  features_train.npz, features_test.npz   — cached features
  lstm_fp32.pt, lstm_int8.pt              — model weights
  cnn_fp32.pt,  cnn_int8.pt              — model weights
  lstm_scaler.pkl                         — StandardScaler for LSTM

Tracked results (small JSON, no weights):
  results/lstm/TIMESTAMP/metrics.json
  results/cnn/TIMESTAMP/metrics.json
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mlee.cnn_model import ECG_CNN
from mlee.lstm_model import ECG_LSTM
from mlee.train import train
from prepare_features import main as prepare_features


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_or_prepare():
    private = ROOT / "private"
    train_path = private / "features_train.npz"
    test_path = private / "features_test.npz"
    if not train_path.exists() or not test_path.exists():
        print("Preparing features from ECG-ID...")
        prepare_features()
    tr = np.load(train_path, allow_pickle=True)
    te = np.load(test_path, allow_pickle=True)
    return tr, te


def main():
    tr, te = load_or_prepare()
    classes = tr["classes"].tolist()
    n_classes = len(classes)

    X_train_dct = tr["X"]          # (N, 32) AC/DCT features
    X_train_raw = tr["X_raw"]      # (N, 300) raw beat windows
    y_train = tr["y"]

    X_test_dct = te["X"]
    X_test_raw = te["X_raw"]
    y_test = te["y"]

    print(f"Classes: {n_classes}  Train: {len(y_train)}  Test: {len(y_test)}")

    # --- LSTM on AC/DCT(32) features ---
    lstm = ECG_LSTM(n_classes=n_classes, hidden=64)
    lstm_out = ROOT / "results" / "lstm" / stamp()
    lstm_metrics, _, _, _ = train(
        lstm, X_train_dct, y_train, X_test_dct, y_test, classes,
        use_raw=False, scale=True,
        experiment_name="lstm",
        output_dir=lstm_out,
    )

    # --- CNN on raw beat windows ---
    cnn = ECG_CNN(n_classes=n_classes)
    cnn_out = ROOT / "results" / "cnn" / stamp()
    cnn_metrics, _, _, _ = train(
        cnn, X_train_raw, y_train, X_test_raw, y_test, classes,
        use_raw=True, scale=False,
        experiment_name="cnn",
        output_dir=cnn_out,
    )

    print("\n========== SUMMARY ==========")
    print(f"{'Model':<25} {'Rank-1':>8} {'EER':>8} {'FP32 KB':>9} {'INT8 KB':>9}")
    print("-" * 65)
    print(f"{'SVM baseline (ref)':<25} {'79.06%':>8} {'13.11%':>8} {'N/A':>9} {'N/A':>9}")
    for name, m in [("LSTM (AC/DCT feat.)", lstm_metrics), ("CNN FP32 (raw beats)", cnn_metrics)]:
        print(
            f"{name:<25}"
            f" {m['identification_accuracy']*100:>7.2f}%"
            f" {m['verification_eer']*100:>7.2f}%"
            f" {m['model_size_fp32_kb']:>8.1f}"
            f" {m['model_size_int8_kb']:>8.1f}"
        )
    cnn_m = cnn_metrics
    print(
        f"{'CNN INT8 (raw beats)':<25}"
        f" {cnn_m['int8_identification_accuracy']*100:>7.2f}%"
        f" {cnn_m['int8_verification_eer']*100:>7.2f}%"
        f" {'N/A':>8}"
        f" {cnn_m['model_size_int8_kb']:>8.1f}"
    )

    # Write a simple combined summary JSON (leaderboard schema needs run.json; skip for now)
    import json as _json
    summary = {
        "baseline": {"identification_accuracy": 0.7906, "verification_eer": 0.1311,
                     "model_size_fp32_kb": None, "model_size_int8_kb": None},
        "lstm": lstm_metrics,
        "cnn": cnn_metrics,
    }
    (ROOT / "results" / "comparison_summary.json").write_text(_json.dumps(summary, indent=2) + "\n")
    print("\nSummary saved: results/comparison_summary.json")


if __name__ == "__main__":
    main()
