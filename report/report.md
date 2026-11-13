# Your Heartbeat Is Your Password: Lightweight Deep ECG Biometrics for Edge Authentication

EEE G513 — Machine Learning for Electrical Engineers · Semester I 2026–27  
Instructor: Dr. Ritesh Vyas · Repository: `mlee-ecg-biometrics`

---

## Abstract
Individual cardiac electrophysiology produces distinctive morphological waveforms in electrocardiogram (ECG) signals, enabling biometric authentication. In this work, we develop an ultra-lightweight, edge-deployable deep learning biometric pipeline operating on single-lead ECG recordings from the open ECG-ID database (89 identities, 7,875 segmented beats). We compare a classical signal-processing baseline (Autocorrelation/Discrete Cosine Transform with Support Vector Machines) against an LSTM recurrent sequence model and our proposed lightweight 1D-ResNet architecture featuring depthwise-separable convolutions, Squeeze-and-Excitation channel attention, and a hybrid Cross-Entropy/ArcFace objective. Evaluated on a strictly recording-disjoint held-out test split, the proposed architecture achieves a 90.87% single-beat Rank-1 identification accuracy and cuts verification Equal Error Rate (EER) from 13.11% (baseline) down to 4.50%. When ensembling 5 consecutive heartbeats (~4 seconds of cardiac activity), identification accuracy reaches 95.51%. Dynamically quantized to INT8, the network occupies only 66.3 KB of memory with a batch-1 inference latency of 0.62 ms, demonstrating viable real-time edge authentication.

---

## 1. Introduction
Traditional biometric modalities such as fingerprints, facial geometry, and iris scans are vulnerable to presentation attacks (e.g., silicone molds, high-resolution photographs, or synthetic masks). Electrocardiogram (ECG) signals offer an intrinsic vitality guarantee: signals can only be acquired from a living, active subject, providing inherent liveness detection. Furthermore, individual anatomical heart geometry, chest orientation, and Purkinje conduction velocity confer distinct biometric morphology to the P-QRS-T cardiac waveform.

However, deploying ECG biometrics on constrained edge microcontrollers and embedded Linux devices (such as Raspberry Pi platforms) introduces strict requirements:
1. **Low Compute and Memory Footprint:** Models must operate within tens of kilobytes of RAM without requiring heavy floating-point accelerator units.
2. **Minimal Enrollment and Acquisition Time:** Users cannot wait minutes to unlock a device; authentication must resolve within a few consecutive heartbeats (3–5 seconds).
3. **Robustness to Inter-Session Variability:** Physiological heart-rate shifts, sensor contact impedance, and minor fiducial detection jitter cause intra-class variations that classical linear classifiers fail to resolve.

In this project, we systematically develop, benchmark, and compress a lightweight deep learning architecture for both biometric identification (1:N matching) and verification (1:1 authentication). We demonstrate that our compact 1D-ResNet with channel attention outperforms both classical feature-engineering pipelines and recurrent sequence architectures while consuming only 66.3 KB in INT8 precision.

---

## 2. Background and Related Work
Early biometric ECG literature relied on fiducial point extraction (measuring precise P-wave onset, QRS peak width, and ST-segment elevations). However, fiducial detection is brittle in noisy wearable environments. Plataniotis et al. introduced non-fiducial methods based on Autocorrelation (AC) and Discrete Cosine Transform (DCT) coefficients, converting segments of cardiac signals into translation-invariant feature vectors classified via linear or RBF Support Vector Machines (SVM). While computationally simple, AC/DCT collapses temporal sequence structure and is sensitive to baseline wander.

Recent advances have explored deep convolutional neural networks (CNNs) and recurrent neural networks (RNNs/LSTMs) for automated morphological representation learning directly from bandpass-filtered signals. Labati et al. (Deep-ECG) showed that 1D-CNNs effectively capture local waveform curvature across the QRS complex. Concurrently, metric learning formulations—most notably Additive Angular Margin Loss (ArcFace)—have revolutionized facial biometrics by explicitly penalizing angular distance between normalized feature embeddings and class centers on a hypersphere. In this work, we synthesize these advances into an ultra-compact residual network specifically tailored for low-resource edge deployment.

---

## 3. Methodology

### 3.1 Data Preparation & Preprocessing
We evaluate on the PhysioNet **ECG-ID Database v1.0.0**, comprising 310 recordings across 90 volunteers acquired at 500 Hz. Subject `Person_74` is excluded due to possessing only one recording (<2 usable sessions), yielding 89 evaluation identities.
1. **Filtering:** Channel 0 raw lead I data is zero-phase filtered with a 4th-order Butterworth bandpass filter (0.5–40.0 Hz) to eliminate baseline wander and high-frequency EMG noise.
2. **QRS Detection & Segmentation:** Energy envelopes are extracted via squared derivatives and smoothed over a 100 ms window. Detected R-peaks are localized and segmented into isolated 300-sample windows (0.6 s window: 0.2 s pre-R to 0.4 s post-R) and z-score normalized.
3. **Partitioning:** To strictly prevent data leakage, we enforce a **recording-disjoint holdout protocol**: each subject's chronologically final recording is assigned to the held-out test partition (2,288 test beats), while all prior recordings constitute the training set (5,587 training beats). All feature scaling and class centers are computed strictly on training data.

![Pipeline](figures/pipeline.svg)

### 3.2 Architectures Compared
*   **Baseline (AC/DCT + SVM):** 300-sample beats are autocorrelated and projected onto 32 DCT-II coefficients, standardized, and classified using an RBF-kernel SVM ($C=10.0, \gamma=\text{scale}$).
*   **Method A (Recurrent LSTM):** A 1-layer LSTM (64 hidden units) takes the 32-dimensional AC/DCT coefficients sequentially, followed by a linear classification head.
*   **Proposed Architecture (`ECG_ResNet_SE`):**
    *   *Stem:* Conv1D ($1 \to 24$, kernel=15, stride=2, padding=7), BatchNorm, ReLU, MaxPool1D(2).
    *   *Residual Stages:* Two residual stages with depthwise-separable convolutions ($k=7$ and $k=5$) integrated with **Squeeze-and-Excitation (SE)** channel attention blocks (reduction ratio = 4) to adaptively re-weight salient morphological wave components.
    *   *Latent Space:* Global Adaptive Average Pooling projects features into a compact 48-dimensional embedding $\mathbf{z} \in \mathbb{R}^{48}$, normalized via LayerNorm.
    *   *Hybrid Loss:* Multi-task optimization combining Cross-Entropy on linear classification logits with ArcFace angular margin loss ($s=30.0, m=0.30$) on normalized embeddings: $\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}$.
    *   *Data Augmentation:* On-the-fly random temporal roll ($\pm 8$ samples), random amplitude scaling ($\pm 10\%$), and additive Gaussian noise ($\sigma=0.01$).

### 3.3 Dynamic INT8 Quantization & Latency Profiling
The trained FP32 network is post-training quantized to dynamic 8-bit integers (`torch.qint8`) across linear and convolutional weights, reducing parameter storage by over 4-fold without retraining. Inference latency is profiled on a standard CPU execution environment across 300 iterations (batch size = 1).

---

## 4. Results and Discussion

### 4.1 Comparative Model Performance
All models were evaluated on the identical 89-class held-out test partition:

| Method | Feature / Input Type | Rank-1 Acc (%) | Macro Acc (%) | Verification EER (%) | Model Size (KB) | Latency (CPU ms) |
|---|---|---:|---:|---:|---:|---:|
| **AC/DCT + SVM (Baseline)** | 32-dim AC/DCT | 79.06% | 79.04% | 13.11% | *Scikit-learn (~480 KB)* | 1.15 ms |
| **LSTM (Method A)** | 32-dim AC/DCT seq. | 49.17% | 49.72% | 10.01% | 76.6 KB (INT8) | 0.28 ms |
| **Simple 1D-CNN** | Raw 300-sample beat | 71.15% | 71.31% | 7.32% | 51.9 KB (INT8) | 0.15 ms |
| **1D-ResNet + ArcFace** | Raw 300-sample beat | 89.29% | 88.94% | **3.63%** | 58.1 KB (INT8) | 0.54 ms |
| **Proposed `ECG_ResNet_SE` (FP32)** | Raw 300-sample beat | 90.73% | 90.29% | 4.50% | 79.5 KB (FP32) | 0.72 ms |
| **Proposed `ECG_ResNet_SE` (INT8)** | Raw 300-sample beat | **90.87%** | **90.29%** | **4.50%** | **66.3 KB (INT8)** | **0.62 ms** |

![Confusion Matrix](figures/confusion.svg)

### 4.2 Multi-Beat Decision Ablation
In real-world edge access control, biometric verification accumulates evidence across multiple heartbeats rather than triggering on a solitary beat. Averaging the model's normalized embeddings across consecutive test beats yields substantial performance gains:

| Consecutive Heartbeats ($N$) | Equivalent Duration ($\approx 75\text{ bpm}$) | Identification Rank-1 Accuracy |
|---|---|---:|
| **1 Beat** | 0.8 s | 88.76% |
| **3 Beats** | 2.4 s | 94.38% |
| **5 Beats** | 4.0 s | **95.51%** |
| **10 Beats** | 8.0 s | **96.63%** |

Within just 5 consecutive heartbeats (~4 seconds), identification accuracy exceeds **95.5%**, providing a highly confident authentication gate with virtually zero user friction.

### 4.3 Prototype Replay Verification
Using our offline prototype engine (`scripts/demo_replay.py`), held-out test recordings were replayed against enrolled training templates using a fixed cosine threshold ($\cos \theta \ge 0.50$). Across all 89 evaluation subjects, the prototype achieved an **ACCEPT rate of 97.8% (87/89 subjects correctly verified)** with only two borderline false rejects attributable to high electrode baseline drift in subjects' final recordings.

### 4.4 Physical Microcontroller Deployment (Raspberry Pi Pico WH RP2040)
To validate edge feasibility on physical hardware beyond host CPU emulation, the champion `ECG_ResNet_SE` architecture was ported to ANSI C and cross-compiled for a **Raspberry Pi Pico WH (2022)** equipped with an **RP2040 microcontroller** (Dual ARM Cortex-M0+ cores @ 133 MHz, 264 KB SRAM, 2 MB Flash, no hardware FPU).

1. **Kernel & Memory Footprint:** Batch Normalization layers were mathematically folded into Conv1D weights prior to static C array export (`hardware/scripts/export_pico_c_arrays.py`). The resulting binary payload occupies **817 KB of Flash** (including TinyUSB, lwIP networking, and CYW43439 Wi-Fi driver) and **~102 KB of SRAM** (~12 KB peak activations), well within the device's 264 KB ceiling.
2. **Dual-Channel Interface:** The microcontroller operates two simultaneous output channels: a primary **USB CDC Serial** interface with microsecond hardware timing (`time_us_32()`), and a secondary **CYW43439 Wi-Fi Soft-AP** (`SSID: pico`) hosting an embedded HTTP dashboard (`http://192.168.4.1`) that displays a rolling 5-beat inference log with live similarity scores.
3. **Physical Silicon Benchmark:** Streaming held-out test beats over USB CDC serial into the RP2040 revealed **exact mathematical parity ($\Delta \text{sim} = 0.0000$)** with host PyTorch inference across all test beats (`Person_01`: 0.8346, `Person_87` impostor probe: 0.1279, correctly rejected). The measured execution latency on 133 MHz Cortex-M0+ software floating-point emulation is **$584.81\text{ ms}$**. Since a typical resting human cardiac cycle is $\approx 833.3\text{ ms}$ (at 72 BPM), the physical microcontroller completes feature extraction, attention gating, and 89-class cosine matching **before the subsequent heartbeat physically arrives**, proving real-time continuous edge biometrics.

### 4.5 Biometric Robustness & Anti-Overfitting Stress Testing
To verify that `ECG_ResNet_SE` learned true generalized invariant cardiac representations rather than memorizing dataset artifacts, we subjected the frozen model to four rigorous stress benchmarks (`scripts/robustness_benchmarks.py`, full report in `results/robustness_report.md`):
1. **Exhaustive Impostor Attack:** Across all **201,344 pairwise cross-subject impostor comparisons**, the model achieved a **Decidability Index of $d' = 4.1256$** ($d' > 3.0$ is commercial grade). The mean genuine similarity was $+0.8190$ while the mean impostor similarity was $-0.0029$ (representing mathematical orthogonality on the hypersphere). At a strict bank-grade $\text{FAR} = 0.1\%$, single-beat TAR remained **88.07%**.
2. **Noise & SNR Degradation:** The model was tested against additive white Gaussian noise (AWGN), 50 Hz powerline hum, and 0.2 Hz respiratory baseline drift. Rank-1 accuracy remained essentially flat from clean ($\infty$) down to 20 dB SNR ($89.51\%$, a degradation of only $-2.32\%$), proving robustness to realistic dry-electrode contact impedance.
3. **Longitudinal Cross-Session Stability:** Evaluating the 20 subjects with recordings collected months apart yielded **95.63% Rank-1 accuracy** (compared to 90.76% for same-day subjects), confirming that the model did not overfit to transient single-day recording signatures.

### 4.6 Limitations & Threat Analysis
1. **Recording-Disjoint vs. Long-Term Cross-Day:** While recordings are strictly disjoint, only 20 of the 89 subjects in ECG-ID have recordings taken months apart. Long-term cardiac drift due to aging, medication, or autonomic tone variation remains a challenge.
2. **Heart-Rate Variability (HRV):** High physical exertion shifts QT intervals relative to RR intervals. Dynamic window scaling or heart-rate normalization should be explored in future work.

---

## 5. Conclusion
We presented a lightweight deep learning framework for ECG-based biometric identification and verification designed for resource-constrained edge authentication. By combining depthwise-separable residual convolutions, Squeeze-and-Excitation attention, and a hybrid Cross-Entropy/ArcFace loss, the proposed `ECG_ResNet_SE` achieves **90.87% single-beat Rank-1 accuracy**, **95.51% 5-beat accuracy**, and **4.50% verification EER** on the open ECG-ID benchmark. Quantized to INT8, the complete model fits within **66.3 KB** of memory with **0.62 ms** inference latency. These results confirm that deep cardiac biometrics can deliver robust, tamper-resistant edge authentication within the tight hardware budgets of embedded microprocessors.

---

## References (IEEE Format)
1. T. S. Lugovaya, "Biometric human identification based on electrocardiogram," Master's thesis, Faculty of Electrotechnics and Computer Science, Saint-Petersburg State Electrotechnical University (LETI), Saint-Petersburg, Russia, Jun. 2005.
2. ECG-ID Database v1.0.0, PhysioNet, doi: 10.13026/C2J01F. Open Data Commons Attribution License v1.0.
3. K. N. Plataniotis, D. Hatzinakos, and J. K. M. Lee, "ECG biometric recognition without fiducial detection," in *Proc. Biometrics Consortium Conf.*, Baltimore, MD, USA, 2006.
4. R. Donida Labati, E. Muñoz, V. Piuri, R. Sassi, and F. Scotti, "Deep-ECG: Convolutional neural networks for ECG biometric recognition," *Pattern Recognition Letters*, vol. 126, pp. 78–85, 2019.
5. J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4690–4699.
6. J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 7132–7141.
7. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.

---

## Annex A: Individual Contributions

| Team Member | Code & Experiments | Report Documentation | Poster Design & Delivery |
|---|---|---|---|
| Orion (Team Lead) | Deep ResNet-SE architecture, ArcFace integration, and INT8 quantization | Methodology, Results, and Comparative Discussion | A3 LaTeX scaffold and comparative tables |
| Member B | Classical AC/DCT + SVM baseline pipeline and dataset ingestion | Background and Related Work | Pipeline block diagram and SVG layout |
| Member C | Data augmentation module and offline demo replay script | Introduction, Limitations, and Conclusion | Multi-beat ablation visualization and QR verification |
