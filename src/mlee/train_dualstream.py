"""Training script for Dual-Stream ResNet with Label Smoothing and Calibrated ArcFace."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlee.arcface import ArcMarginProduct
from mlee.dataset import ECGDataset
from mlee.resnet_dualstream_model import ECG_ResNet_DualStream
from mlee.train import _eer


def _cosine_eer_dualstream(model, X_test, y_test, X_train, y_train, classes, device):
    model.eval()
    with torch.no_grad():
        emb_train = model.embed(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        emb_test = model.embed(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    templates = np.stack([
        emb_train[y_train == i].mean(axis=0) for i in range(len(classes))
    ])

    def normalize(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(n, 1e-12)

    scores = normalize(emb_test) @ normalize(templates).T
    genuine = (y_test[:, None] == np.arange(len(classes))[None, :])
    return _eer(genuine.ravel(), scores.ravel())


def train_dualstream(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    lambda_arc: float = 0.75,
    label_smoothing: float = 0.05,
    seed: int = 42,
    output_dir: Path = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    n_classes = len(classes)
    model = ECG_ResNet_DualStream(n_classes=n_classes, in_channels=2, emb_dim=48).to(device)
    arcface = ArcMarginProduct(in_features=48, out_features=n_classes, s=32.0, m=0.35).to(device)

    train_ds = ECGDataset(X_train, y_train, augment=True, dual_stream=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = ECGDataset(X_test, y_test, augment=False, dual_stream=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print(f"\n================ Training ECG_ResNet_DualStream (Epochs={epochs}, LabelSmooth={label_smoothing}) ================")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        arcface.train()
        total_loss, ce_loss_sum, arc_loss_sum = 0.0, 0.0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            emb = model.embed(xb)
            logits_ce = model.head(emb)
            logits_arc = arcface(emb, yb)

            loss_ce = criterion_ce(logits_ce, yb)
            loss_arc = criterion_ce(logits_arc, yb)
            loss = loss_ce + (lambda_arc * loss_arc)

            loss.backward()
            optimizer.step()

            bs = len(xb)
            total_loss += loss.item() * bs
            ce_loss_sum += loss_ce.item() * bs
            arc_loss_sum += loss_arc.item() * bs

        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:2d}/{epochs} - loss: {total_loss/len(X_train):.4f} "
                f"(CE: {ce_loss_sum/len(X_train):.4f}, Arc: {arc_loss_sum/len(X_train):.4f}) "
                f"- lr: {scheduler.get_last_lr()[0]:.6f}"
            )

    elapsed = time.time() - t0

    # Evaluation
    model.eval()
    preds_all = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds_all)

    rank1 = float(accuracy_score(y_test, pred))
    per_class = [float(np.mean(pred[y_test == i] == i)) for i in range(len(classes))]
    macro = float(np.mean(per_class))

    eer_val = _cosine_eer_dualstream(model, X_test, y_test, X_train, y_train, classes, device)

    # Model Serialization (FP32)
    tmp_fp32 = Path("/tmp") / "resnet_dualstream_fp32.pt"
    torch.save(model.state_dict(), tmp_fp32)
    fp32_size = os.path.getsize(tmp_fp32) / 1024

    # INT8 Dynamic Quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    tmp_int8 = Path("/tmp") / "resnet_dualstream_int8.pt"
    torch.save(model_int8.state_dict(), tmp_int8)
    int8_size = os.path.getsize(tmp_int8) / 1024

    model_int8.eval()
    preds_int8_all = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits_int8 = model_int8(xb)
            preds_int8_all.append(logits_int8.argmax(dim=1).cpu().numpy())
    pred_int8 = np.concatenate(preds_int8_all)
    rank1_int8 = float(accuracy_score(y_test, pred_int8))
    eer_int8 = _cosine_eer_dualstream(model_int8, X_test, y_test, X_train, y_train, classes, "cpu")

    # Diagnosis: Person_87, 70, 67 accuracy
    idx_87 = classes.index("Person_87") if "Person_87" in classes else -1
    idx_70 = classes.index("Person_70") if "Person_70" in classes else -1
    idx_67 = classes.index("Person_67") if "Person_67" in classes else -1
    acc_87 = float(np.mean(pred_int8[y_test == idx_87] == idx_87)) if idx_87 >= 0 else None
    acc_70 = float(np.mean(pred_int8[y_test == idx_70] == idx_70)) if idx_70 >= 0 else None
    acc_67 = float(np.mean(pred_int8[y_test == idx_67] == idx_67)) if idx_67 >= 0 else None

    metrics = {
        "status": "measured",
        "experiment": "resnet_se_dualstream",
        "model": "ECG_ResNet_DualStream (2-Channel [x, dx] + LabelSmoothing + ArcFace)",
        "protocol": "last usable record per person held out; recording-disjoint",
        "dataset": "ECG-ID 1.0.0",
        "seed": seed,
        "epochs": epochs,
        "identities": len(classes),
        "train_beats": len(X_train),
        "test_beats": len(X_test),
        "identification_accuracy": rank1,
        "macro_identity_accuracy": macro,
        "verification_eer": eer_val,
        "model_size_fp32_kb": round(fp32_size, 2),
        "model_size_int8_kb": round(int8_size, 2),
        "int8_identification_accuracy": rank1_int8,
        "int8_verification_eer": eer_int8,
        "training_time_s": round(elapsed, 1),
        "person_87_accuracy": acc_87,
        "person_70_accuracy": acc_70,
        "person_67_accuracy": acc_67,
    }

    print("\nResults for ECG_ResNet_DualStream:")
    print(f"  Rank-1 FP32: {rank1*100:.2f}% | INT8: {rank1_int8*100:.2f}%")
    print(f"  Macro  FP32: {macro*100:.2f}%")
    print(f"  EER    FP32: {eer_val*100:.2f}% | INT8: {eer_int8*100:.2f}%")
    print(f"  Size   FP32: {fp32_size:.1f} KB | INT8: {int8_size:.1f} KB")
    print(f"  Diagnostics: Person_87={acc_87*100:.1f}%, Person_70={acc_70*100:.1f}%, Person_67={acc_67*100:.1f}%")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    private = ROOT / "private"
    private.mkdir(exist_ok=True)
    torch.save(model.state_dict(), private / "resnet_dualstream_fp32.pt")
    torch.save(model_int8, private / "resnet_dualstream_int8_model.pt")

    return metrics, model, model_int8
