"""Experiment runner for advanced lightweight improvements:

1. Baseline CNN with Data Augmentation
2. Lightweight 1D-ResNet
3. 1D-ResNet with ArcFace Loss
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from mlee.arcface import ArcMarginProduct
from mlee.cnn_model import ECG_CNN
from mlee.dataset import ECGDataset
from mlee.resnet_model import ECG_ResNet
from mlee.train import _cosine_eer

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def train_eval(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    classes,
    *,
    head=None,
    augment=False,
    epochs=40,
    lr=1e-3,
    batch_size=64,
    exp_name="improved_model",
):
    device = torch.device("cpu")
    model = model.to(device)
    if head is not None:
        head = head.to(device)

    train_ds = ECGDataset(X_train, y_train, augment=augment)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\n================ Training {exp_name} (Augment={augment}, Epochs={epochs}) ================")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        if head is not None:
            head.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if head is not None:
                emb = model.embed(xb)
                logits = head(emb, yb)
            else:
                logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} - loss: {total_loss/len(X_train):.4f} - lr: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = time.time() - t0

    # Evaluation
    model.eval()
    if head is not None:
        head.eval()

    with torch.no_grad():
        if head is not None:
            emb_test = model.embed(X_test_t)
            logits = head(emb_test)
        else:
            logits = model(X_test_t)
        pred = logits.argmax(dim=1).cpu().numpy()

    rank1 = float(accuracy_score(y_test, pred))
    per_class = [float(np.mean(pred[y_test == i] == i)) for i in range(len(classes))]
    macro = float(np.mean(per_class))

    eer_val = _cosine_eer(
        model,
        X_test_t,
        y_test,
        torch.tensor(X_train, dtype=torch.float32),
        y_train,
        classes,
        device,
    )

    # Measure FP32 size
    tmp_path = Path("/tmp") / f"{exp_name}_fp32.pt"
    torch.save(model.state_dict(), tmp_path)
    fp32_size = os.path.getsize(tmp_path) / 1024

    # INT8 Quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    tmp_int8 = Path("/tmp") / f"{exp_name}_int8.pt"
    torch.save(model_int8.state_dict(), tmp_int8)
    int8_size = os.path.getsize(tmp_int8) / 1024

    model_int8.eval()
    with torch.no_grad():
        if head is not None:
            emb_test_int8 = model_int8.embed(X_test_t.cpu())
            logits_int8 = head(emb_test_int8)
        else:
            logits_int8 = model_int8(X_test_t.cpu())
        pred_int8 = logits_int8.argmax(dim=1).cpu().numpy()
    rank1_int8 = float(accuracy_score(y_test, pred_int8))
    eer_int8 = _cosine_eer(
        model_int8,
        X_test_t.cpu(),
        y_test,
        torch.tensor(X_train, dtype=torch.float32),
        y_train,
        classes,
        "cpu",
    )

    metrics = {
        "experiment": exp_name,
        "augment": augment,
        "epochs": epochs,
        "rank1_fp32": rank1,
        "macro_fp32": macro,
        "eer_fp32": eer_val,
        "rank1_int8": rank1_int8,
        "eer_int8": eer_int8,
        "fp32_kb": round(fp32_size, 2),
        "int8_kb": round(int8_size, 2),
        "time_s": round(elapsed, 1),
    }

    print(f"Results for {exp_name}:")
    print(f"  Rank-1 FP32: {rank1*100:.2f}% | INT8: {rank1_int8*100:.2f}%")
    print(f"  EER    FP32: {eer_val*100:.2f}% | INT8: {eer_int8*100:.2f}%")
    print(f"  Size   FP32: {fp32_size:.1f} KB | INT8: {int8_size:.1f} KB")

    out_dir = ROOT / "results" / exp_name / stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Save to private for replay
    private = ROOT / "private"
    torch.save(model.state_dict(), private / f"{exp_name}_fp32.pt")
    torch.save(model_int8, private / f"{exp_name}_int8_model.pt")

    return metrics, model


def main():
    private = ROOT / "private"
    tr = np.load(private / "features_train.npz", allow_pickle=True)
    te = np.load(private / "features_test.npz", allow_pickle=True)

    X_train = tr["X_raw"]
    y_train = tr["y"]
    X_test = te["X_raw"]
    y_test = te["y"]
    classes = tr["classes"].tolist()
    n_classes = len(classes)

    print(f"Loaded {len(y_train)} train beats, {len(y_test)} test beats, {n_classes} classes.")

    results = []

    # 1. CNN with Data Augmentation
    cnn_aug = ECG_CNN(n_classes=n_classes)
    m1, _ = train_eval(
        cnn_aug,
        X_train,
        y_train,
        X_test,
        y_test,
        classes,
        augment=True,
        epochs=40,
        lr=1e-3,
        exp_name="cnn_augmented",
    )
    results.append(m1)

    # 2. ResNet-1D (with Data Augmentation)
    resnet = ECG_ResNet(n_classes=n_classes, emb_dim=48)
    m2, _ = train_eval(
        resnet,
        X_train,
        y_train,
        X_test,
        y_test,
        classes,
        augment=True,
        epochs=40,
        lr=1e-3,
        exp_name="resnet1d_augmented",
    )
    results.append(m2)

    # 3. ResNet-1D with ArcFace Loss
    resnet_arc = ECG_ResNet(n_classes=n_classes, emb_dim=48)
    arcface_head = ArcMarginProduct(in_features=48, out_features=n_classes, s=30.0, m=0.35)
    m3, best_model = train_eval(
        resnet_arc,
        X_train,
        y_train,
        X_test,
        y_test,
        classes,
        head=arcface_head,
        augment=True,
        epochs=45,
        lr=1e-3,
        exp_name="resnet1d_arcface",
    )
    results.append(m3)

    summary_file = ROOT / "results" / "experiments_comparison.json"
    summary_file.write_text(json.dumps(results, indent=2) + "\n")

    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(f"{'Experiment':<22} {'Rank-1 FP32':>12} {'Rank-1 INT8':>12} {'EER FP32':>10} {'EER INT8':>10} {'Size KB':>9}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['experiment']:<22}"
            f" {r['rank1_fp32']*100:>11.2f}%"
            f" {r['rank1_int8']*100:>11.2f}%"
            f" {r['eer_fp32']*100:>9.2f}%"
            f" {r['eer_int8']*100:>9.2f}%"
            f" {r['int8_kb']:>9.1f}"
        )


if __name__ == "__main__":
    main()
