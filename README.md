# Your Heartbeat Is Your Password: Lightweight Deep ECG Biometrics for Edge Authentication

Course Project for **EEE G513: Machine Learning for Electrical Engineers** (Semester I 2026–27)  
Instructor: **Dr. Ritesh Vyas** · Branch: `orion` · Status: **Fully Verified & Documented**

---

## 1. Project Overview

This repository implements an ultra-lightweight, edge-deployable deep learning biometric pipeline operating on single-lead electrocardiogram (ECG) heartbeats from the open **ECG-ID Database v1.0.0** (89 usable subjects, 7,875 segmented beats).

The project benchmark compares:
1. **Classical Baseline:** Autocorrelation + Discrete Cosine Transform (AC/DCT-32) with an RBF Support Vector Machine.
2. **Method A (Recurrent):** A 1-layer LSTM on AC/DCT coefficients.
3. **Proposed Lightweight Architectures:** Compact 1D-ResNet with depthwise-separable convolutions, Squeeze-and-Excitation (SE) channel attention, and a hybrid Cross-Entropy / ArcFace angular margin loss.

All models are evaluated on a **strictly recording-disjoint split** (each subject's chronologically final recording is held out for testing; no adjacent beat leakage).

---

## 2. Key Results Summary

| Model / Architecture | Input Stream | Single-Beat Rank-1 | Macro Identity Acc | Verification EER | Quantized Model Size | CPU Latency (Batch=1) |
|---|---|---:|---:|---:|---:|---:|
| **Baseline: AC/DCT + SVM** | 32-dim AC/DCT | 79.06% | 79.04% | 13.11% | *480 KB* | 1.15 ms |
| **Method A: LSTM (INT8)** | 32-dim AC/DCT seq. | 49.26% | 49.72% | 10.01% | 76.6 KB | 0.28 ms |
| **Simple 1D-CNN (INT8)** | Raw voltage $x[t]$ | 71.37% | 71.31% | 7.32% | **51.9 KB** | **0.15 ms** |
| **Unified 1D-ResNet (`ECG_ResNet_SE`)** | Raw voltage $x[t]$ | 90.87% | 90.41% | **4.50%** | **66.3 KB** | **0.62 ms** |
| **Dual-Stream ResNet (`ECG_ResNet_DualStream`)** | Voltage $x[t]$ + Derivative $\Delta x[t]$ | **91.08%** | **90.94%** | **4.93%** | **67.9 KB** | **0.65 ms** |

### Multi-Beat Decision Window Ablation
In real-world biometric access control, authentication integrates consecutive heartbeats (~75 bpm):
*   **1 Beat (~0.8 s):** 87.64% – 88.76% Rank-1
*   **3 Beats (~2.4 s):** 94.38% – 95.51% Rank-1
*   **5 Beats (~4.0 s):** **95.51% – 96.63% Rank-1**
*   **10 Beats (~8.0 s):** **96.63% – 97.75% Rank-1**

---

## 3. Quickstart & Verification

Ensure [uv](https://docs.astral.sh/uv/) is installed, then run from repository root:

```bash
# 1. Sync dependencies and download official ECG-ID dataset (~13 MB)
make setup
make data

# 2. Run unit and integration tests (deterministic synthetic checks)
make test

# 3. Code formatting and linting (Ruff)
make lint

# 4. Generate report figure assets (pipeline.svg, confusion.svg)
make figures

# 5. Run the offline biometric replay prototype
python scripts/demo_replay.py --model champion
```

---

## 4. Repository Structure & Deliverables Map

```
mlee-ecg-biometrics/
├── configs/                    ← Formal experiment YAML configurations
│   ├── baseline.yaml
│   ├── resnet_se_hybrid.yaml
│   └── resnet_se_dualstream.yaml
├── src/mlee/                   ← Core production package
│   ├── baseline.py             ← Frozen AC/DCT + SVM reference pipeline
│   ├── dataset.py              ← Dual-stream [x, dx] generation & on-the-fly augmentation
│   ├── resnet_se_model.py      ← 1D-ResNet with depthwise convs & SE channel attention
│   ├── resnet_dualstream_model.py ← Dual-stream 2-channel ResNet-SE
│   ├── arcface.py              ← Additive Angular Margin loss module
│   ├── train_hybrid.py         ← Multi-task hybrid Cross-Entropy + ArcFace trainer
│   └── experiments.py          ← Central experiment dispatcher and provenance logger
├── scripts/                    ← Executable tools & demo pipelines
│   ├── demo_replay.py          ← Offline ACCEPT/REJECT prototype demo
│   ├── beat_ablation.py        ← Multi-beat decision evaluation script
│   ├── prepare_features.py     ← ECG-ID beat extraction & feature caching
│   ├── make_figures.py         ← Generates pipeline and confusion SVG assets
│   └── leaderboard.py          ← Rebuilds results/LEADERBOARD.md
├── report/                     ← Project report deliverable
│   ├── report.md               ← Complete written paper (IEEE format)
│   └── figures/                ← pipeline.svg, confusion.svg
├── poster/                     ← Course A3 poster deliverable
│   └── poster.tex              ← Full A3 landscape LaTeX source code
├── results/                    ← Timestamped experiment provenance & metrics JSON
│   ├── LEADERBOARD.md          ← Auto-generated comparison leaderboard
│   └── comparison_summary.json ← Consolidated benchmark summary
├── private/                    ← Local weights & private engineering records (gitignored)
│   ├── JOURNAL.md              ← Comprehensive engineering log & error diagnosis
│   └── *.pt                    ← Quantized INT8 and FP32 checkpoints
├── TASKS.md                    ← Milestone task tracking (all completed)
├── PLAN.md                     ← Original course delivery plan
└── pyproject.toml              ← Python 3.12, PyTorch CPU, locked deps
```

---

## 5. Licensing & Citations

*   **Code:** MIT License.
*   **Dataset:** [ECG-ID Database v1.0.0](https://physionet.org/content/ecgiddb/1.0.0/) by Tatiana Lugovaya, Open Data Commons Attribution License (ODC-By 1.0).
*   **PhysioNet Citation:** Pollard et al., *PhysioNet as a global platform for biomedical research*, Nature Health (2026), doi:10.1038/s44360-026-00096-z.
