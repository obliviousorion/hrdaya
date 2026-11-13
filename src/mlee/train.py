"""Shared training loop for LSTM and CNN models.

Handles:
- DataLoader construction
- 30-epoch fixed training with Adam
- Rank-1, macro accuracy, cosine EER on the held-out test split
- Saving compact metrics JSON compatible with the existing leaderboard schema
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]


def _eer(labels, scores):
    from sklearn.metrics import roc_curve
    far, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    frr = 1 - tpr
    delta = far - frr
    j = int(np.flatnonzero(delta >= 0)[0])
    i = max(0, j - 1)
    w = 0 if delta[j] == delta[i] else -delta[i] / (delta[j] - delta[i])
    return float(far[i] + w * (far[j] - far[i]))


def _cosine_eer(model, X_test_t, y_test, X_train_t, y_train, classes, device):
    model.eval()
    with torch.no_grad():
        emb_train = model.embed(X_train_t.to(device)).cpu().numpy()
        emb_test = model.embed(X_test_t.to(device)).cpu().numpy()

    # Build enrollment templates: mean embedding per class
    templates = np.stack([
        emb_train[y_train == i].mean(axis=0) for i in range(len(classes))
    ])

    def normalize(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(n, 1e-12)

    scores = normalize(emb_test) @ normalize(templates).T  # (M, C)
    genuine = (y_test[:, None] == np.arange(len(classes))[None, :])
    return _eer(genuine.ravel(), scores.ravel())


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    use_raw: bool = False,         # True → CNN (raw beats); False → LSTM (AC/DCT)
    scale: bool = True,            # StandardScaler on features (not raw beats)
    device_str: str = "cpu",
    experiment_name: str = "model",
    output_dir: Path = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    # Optionally scale features (for LSTM on AC/DCT)
    scaler = None
    if scale and not use_raw:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Training {experiment_name} ---")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}/{epochs}  loss={epoch_loss/len(X_train):.4f}")
    elapsed = time.time() - t0

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device)).cpu()
        pred = logits.argmax(dim=1).numpy()

    rank1 = float(accuracy_score(y_test, pred))
    per_class = [float(np.mean(pred[y_test == i] == i)) for i in range(len(classes))]
    macro = float(np.mean(per_class))

    eer_val = _cosine_eer(model, X_test_t, y_test, X_train_t, y_train, classes, device)

    # Model size
    model_path = Path("/tmp") / f"{experiment_name}_fp32.pt"
    torch.save(model.state_dict(), model_path)
    size_fp32_kb = os.path.getsize(model_path) / 1024

    # INT8 dynamic quantisation
    model_int8 = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    int8_path = Path("/tmp") / f"{experiment_name}_int8.pt"
    torch.save(model_int8.state_dict(), int8_path)
    size_int8_kb = os.path.getsize(int8_path) / 1024

    # INT8 accuracy
    model_int8.eval()
    with torch.no_grad():
        pred_int8 = model_int8(X_test_t).argmax(dim=1).numpy()
    rank1_int8 = float(accuracy_score(y_test, pred_int8))

    # EER for INT8
    eer_int8 = _cosine_eer(model_int8, X_test_t, y_test, X_train_t, y_train, classes, "cpu")

    metrics = {
        "status": "measured",
        "experiment": experiment_name,
        "protocol": "last usable record per person held out; recording-disjoint, NOT proven cross-day",
        "dataset": "ECG-ID 1.0.0",
        "seed": seed,
        "epochs": epochs,
        "identities": len(classes),
        "train_beats": len(X_train),
        "test_beats": len(X_test),
        "identification_accuracy": rank1,
        "macro_identity_accuracy": macro,
        "verification_eer": eer_val,
        "eer_note": "Descriptive empirical test EER; deployment threshold not separately validated.",
        "model_size_fp32_kb": round(size_fp32_kb, 2),
        "model_size_int8_kb": round(size_int8_kb, 2),
        "int8_identification_accuracy": rank1_int8,
        "int8_verification_eer": eer_int8,
        "training_time_s": round(elapsed, 1),
    }

    print(f"\nResults ({experiment_name}):")
    print(f"  Rank-1:       {rank1*100:.2f}%")
    print(f"  Macro:        {macro*100:.2f}%")
    print(f"  EER:          {eer_val*100:.2f}%")
    print(f"  FP32 size:    {size_fp32_kb:.1f} KB")
    print(f"  INT8 size:    {size_int8_kb:.1f} KB")
    print(f"  INT8 Rank-1:  {rank1_int8*100:.2f}%")
    print(f"  INT8 EER:     {eer_int8*100:.2f}%")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        # Copy model weights to private/ for demo use
        private = ROOT / "private"
        private.mkdir(exist_ok=True)
        torch.save(model.state_dict(), private / f"{experiment_name}_fp32.pt")
        # Save full INT8 model object (quantised layers have different schema)
        torch.save(model_int8, private / f"{experiment_name}_int8_model.pt")
        if scaler is not None:
            import pickle
            (private / f"{experiment_name}_scaler.pkl").write_bytes(pickle.dumps(scaler))

    return metrics, model, model_int8, scaler
