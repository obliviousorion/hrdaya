# Team Tasks & Progress Tracking

**Milestone Target:** Friday 20 November 2026 · **Content Freeze:** 18 November  
**Current State:** All technical implementations, model training, evaluation, INT8 quantization, and deliverable documentation are **COMPLETED**.

---

## 8 November — Data and Protocol 2026

### DATA-01: Validate ECG reader and inspect peaks
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Format-16 reader verified against ECG-ID 1.0.0; 4th-order zero-phase bandpass (0.5–40 Hz) and R-peak segmentation (300 samples, 0.6 s window) verified on 310 recordings. Excluded `Person_74` (<2 usable recordings).

### DATA-02: Freeze date-aware split manifests
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Enforced strict recording-disjoint split holding out chronologically final recording per person across 89 identities (5,587 train beats, 2,288 test beats). Audited inter-session dates: 20 subjects have multi-day sessions, 69 subjects single-day.

### TEAM-01: Confirm title team and hardware
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Confirmed course title *"Your Heartbeat Is Your Password: Lightweight Deep ECG Biometrics for Edge Authentication"*; target deployment confirmed for Linux-class (Raspberry Pi) and bare-metal microcontroller (RP2040 Pico / ESP32).

---

## 11 November — Model Comparison 2026

### MODEL-01: Implement compact CNN
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Implemented in `src/mlee/cnn_model.py`. Evaluated on raw 300-sample beats: 71.37% Rank-1, 7.32% EER, 51.9 KB INT8.

### MODEL-02: Implement LSTM comparison
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Implemented in `src/mlee/lstm_model.py` consuming 32-dim AC/DCT coefficients. Evaluated: 49.26% Rank-1, 10.01% EER, 76.6 KB INT8.

### EVAL-01: Verification and error analysis
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Enrolled training cosine templates; evaluated descriptive EER across all models. Analyzed confusion matrix and identified single-session class drift in `Person_87`/`Person_70`.

---

## 14 November — Ablations and Deployment 2026

### EVAL-02: Evaluate day and beat ablations
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Implemented multi-beat evaluation (1, 3, 5, 10 consecutive beats). Champion models achieve 95.51% (5 beats) and 97.75% (10 beats).

### EDGE-01: Export and validate INT8
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Exported dynamic INT8 quantization across all models with zero EER degradation and <0.1% accuracy variation; model size reduced to 66.3 KB (`ECG_ResNet_SE`) and 67.9 KB (`ECG_ResNet_DualStream`).

### EDGE-02: Build offline replay and measure Pi latency
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** `scripts/demo_replay.py` provides offline ACCEPT/REJECT verification (97.8% genuine accept rate). Benchmarked batch-1 CPU latency at 0.62 ms (p95: 1.06 ms).

### EDGE-03: Physical Microcontroller Deployment on Raspberry Pi Pico WH (RP2040)
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Fused BatchNorm into Conv1D weights; engineered zero-dependency ANSI C kernel (`hardware/firmware/nn_inference.h`); built dual-channel firmware (`hardware/build_pico/pico_ecg_biometrics.uf2`, 817 KB); deployed onto physical Pico WH silicon; established Wi-Fi Soft-AP (`SSID: pico`, `picosama`) with dark terminal web dashboard (`http://192.168.4.1`) rendering rolling 5-beat history; verified on physical RP2040 hardware with exact bit-level parity ($\Delta\text{sim} = 0.0000$) and 584.81 ms execution latency (well within the 833.3 ms cardiac cycle).

---

## 16 November — Results and Writing 2026

### WRITE-01: Freeze results and report draft
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** Complete report drafted in `report/report.md` with all sections (Abstract, Intro, Background, Methodology, Results, Conclusion, IEEE References, Annex A). Figures embedded.

### WRITE-02: Draft A3 poster
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** A3 landscape LaTeX scaffold drafted in `poster/poster.tex` matching report numbers, comparative tables, and multi-beat ablation.

---

## 18 November — Content Freeze 2026

### RELEASE-01: Reproduce and proof final package
- **Owner:** Team Orion
- **Status:** **completed**
- **Outcome:** `make test` (8/8 tests) and `make lint` (Ruff 0.17.1) cleanly passing; Git working tree clean on branch `orion`.

---

## 19–20 November — Delivery Buffer 2026

### DELIVER-01: Submit and deliver final artifacts
- **Owner:** Team Orion
- **Status:** **ready**
- **Outcome:** Report, A3 poster LaTeX source, prototype replay scripts, and checkpoints ready for final packaging and submission.
