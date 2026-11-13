"""Hybrid training loop combining CrossEntropy and ArcFace margin loss."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from mlee.arcface import ArcMarginProduct
from mlee.dataset import ECGDataset
from mlee.resnet_se_model import ECG_ResNet_SE
from mlee.train import _cosine_eer

ROOT = Path(__file__).resolve().parents[2]


def train_hybrid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list,
    *,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_arc: float = 0.5,
    seed: int = 42,
    output_dir: Path = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    n_classes = len(classes)
    model = ECG_ResNet_SE(n_classes=n_classes, emb_dim=48).to(device)
    arcface = ArcMarginProduct(in_features=48, out_features=n_classes, s=30.0, m=0.30).to(device)

    train_ds = ECGDataset(X_train, y_train, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce = nn.CrossEntropyLoss()

    print(f"\n================ Training ECG_ResNet_SE (Hybrid Loss, Epochs={epochs}) ================")
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
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:2d}/{epochs} - loss: {total_loss/len(X_train):.4f} "
                f"(CE: {ce_loss_sum/len(X_train):.4f}, Arc: {arc_loss_sum/len(X_train):.4f}) "
                f"- lr: {scheduler.get_last_lr()[0]:.6f}"
            )

    elapsed = time.time() - t0

    # Evaluation
    model.eval()
    with torch.no_grad():
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

    # Model Serialization (FP32)
    tmp_fp32 = Path("/tmp") / "resnet_se_hybrid_fp32.pt"
    torch.save(model.state_dict(), tmp_fp32)
    fp32_size = os.path.getsize(tmp_fp32) / 1024

    # INT8 Dynamic Quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )
    tmp_int8 = Path("/tmp") / "resnet_se_hybrid_int8.pt"
    torch.save(model_int8.state_dict(), tmp_int8)
    int8_size = os.path.getsize(tmp_int8) / 1024

    model_int8.eval()
    with torch.no_grad():
        pred_int8 = model_int8(X_test_t.cpu()).argmax(dim=1).cpu().numpy()
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
        "status": "measured",
        "experiment": "resnet_se_hybrid",
        "model": "ECG_ResNet_SE (Depthwise + SE-Block + Hybrid CE/ArcFace)",
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
    }

    print("\nResults for ECG_ResNet_SE:")
    print(f"  Rank-1 FP32: {rank1*100:.2f}% | INT8: {rank1_int8*100:.2f}%")
    print(f"  EER    FP32: {eer_val*100:.2f}% | INT8: {eer_int8*100:.2f}%")
    print(f"  Size   FP32: {fp32_size:.1f} KB | INT8: {int8_size:.1f} KB")

    # Save to results/ and private/
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    private = ROOT / "private"
    private.mkdir(exist_ok=True)
    torch.save(model.state_dict(), private / "resnet_se_hybrid_fp32.pt")
    torch.save(model_int8, private / "resnet_se_hybrid_int8_model.pt")

    return metrics, model, model_int8
