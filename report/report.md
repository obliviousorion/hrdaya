# Lightweight Deep ECG Biometrics for Edge Authentication

**Course:** EEE G513 — Machine Learning for Electrical Engineers · Semester I 2026–27  
**Department:** Department of Electrical and Electronics Engineering, BITS Pilani  
**Team ID:** Team 14  
**Team Members:** Student 1 (2023A3PS0001P); Student 2 (2023A3PS0002P); Student 3 (2023A3PS0003P)  
**Instructor:** Dr. Ritesh Vyas  
**Date of Submission:** 20 November 2026  
**Code & Data Repository:** [https://github.com/mlee-ecg-biometrics](https://github.com/mlee-ecg-biometrics)

---

## Abstract
Traditional biometric modalities such as fingerprints and facial recognition are susceptible to presentation attacks and demand computationally prohibitive hardware. In this work, we present an ultra-lightweight, end-to-end deep learning biometric pipeline executing real-time cardiac authentication on an ARM Cortex-M0+ microcontroller without a hardware floating-point unit. Following an empirical strategy that progressively benchmarks and diagnoses classical Autocorrelation/Discrete Cosine Transform SVM baselines, recurrent LSTMs, and 1D-CNNs, we engineer the `ECG_ResNet_SE` architecture featuring depthwise-separable convolutions, Squeeze-and-Excitation channel attention, and Additive Angular Margin (ArcFace) loss. On the PhysioNet ECG-ID database (89 identities, 7,875 segmented beats) under a strictly recording-disjoint protocol, `ECG_ResNet_SE` achieves 90.87% single-beat Rank-1 accuracy and 4.50% Equal Error Rate (EER) with a decidability index of $d' = 4.13$ (93.62% True Accept Rate at 1.0% False Accept Rate). Ensembling 5 beats elevates identification to 95.51%. Furthermore, extending to a Dual-Stream architecture fusing voltage morphology with phase velocity ($dV/dt$) pushes 8-beat identification beyond 97% (97.19%) and 10-beat identification to 97.75%. Deployed in C99 on a \$5 Raspberry Pi Pico WH at 250 MHz, the single-stream network executes in 300.05 ms within 161.6 KB SRAM with zero Flash writes, demonstrating viable, privacy-preserving edge biometrics.

**Keywords:** Electrocardiogram Biometrics, Edge Machine Learning, Squeeze-and-Excitation ResNet, Dual-Stream Neural Network, ArcFace Metric Learning.

---

## 1. Introduction
Biometric access control systems in high-security cyber-physical applications increasingly demand continuous authentication paired with intrinsic liveness verification. Optical facial recognition and capacitive fingerprint scanners are vulnerable to presentation attacks employing high-resolution prints, 3D silicone molds, or synthetic deepfakes [1]. Electrocardiogram (ECG) signals offer an intrinsic vitality guarantee: valid cardiac electrophysiological dipoles exist solely in living subjects and cease immediately upon circulatory arrest. Furthermore, individual thoracic geometry, myocardial muscle distribution, and Purkinje conduction velocity confer distinctive morphological features upon the P-QRS-T voltage waveform.

Despite these advantages, deploying deep biometric models on resource-constrained edge microcontrollers introduces acute engineering challenges:
1. **Severe Memory & Compute Constraints:** Low-cost microcontrollers such as the ARM Cortex-M0+ possess limited SRAM ($\le 264\text{ KB}$) and lack hardware Floating-Point Units (FPUs), rendering modern multi-layer vision networks unusable.
2. **Real-World Electrical Noise:** Wearable dry or gel electrodes deployed in ambient environments suffer from heavy 50 Hz mains electromagnetic hum, respiration baseline drift, and motion transients that overwhelm naive classifiers [2].
3. **Data Leakage in Published Literature:** Widespread published studies randomly shuffle individual heartbeats across train and test partitions, reporting inflated accuracies ($>99\%$) that collapse under real-world inter-session impedance shifts.

To resolve these challenges, this project follows an empirical progression that systematically benchmarks competing paradigms:
* We progressively benchmark classical AC/DCT-32 SVMs, recurrent LSTMs, and shallow 1D-CNNs, diagnosing their theoretical failure modes under a strictly recording-disjoint split across 89 PhysioNet identities.
* We engineer the compact `ECG_ResNet_SE` architecture (14,867 parameters) combining residual learning, Squeeze-and-Excitation channel attention, and ArcFace hyperspherical metric loss, achieving 90.87% single-beat and 95.51% 5-beat identification.
* We develop a Dual-Stream extension fusing raw voltage morphology with instantaneous derivative slew rate ($dV/dt$), demonstrating that ensembling across 8 heartbeats pushes identification beyond 97% (97.19%) and reaches 97.75% across 10 beats.
* We deploy the optimized model onto a dual-core Raspberry Pi Pico WH (RP2040 @ 250 MHz) in an Asymmetric Multiprocessing (AMP) topology, characterizing soft-float execution, SRAM activation envelopes, and zero-shot enrollment without Flash wear.

Section 2 reviews foundational electrophysiology and related literature. Section 3 details the DSP pipeline, network architecture, and embedded implementation. Section 4 presents quantitative and live bench results. Section 5 concludes with prospective enhancements.

---

## 2. Background and Related Work

### 2.1 Theoretical Preliminaries
Cardiac electrophysiology is initiated by spontaneous pacemaker depolarization at the sinoatrial (SA) node, propagating sequentially through the atria (P-wave), atrioventricular (AV) node delay, and the Purkinje fiber network to trigger rapid ventricular contraction (QRS complex), followed by ventricular repolarization (T-wave). In vectorcardiography, the cardiac dipole moment $\mathbf{J}(t)$ projects an individual electrical vector onto surface lead axes:
$$V_{\text{lead}}(t) = \int \mathbf{L}(\mathbf{r}) \cdot \mathbf{J}(\mathbf{r}, t) \, d^3\mathbf{r} \tag{1}$$
where $\mathbf{L}(\mathbf{r})$ represents the lead vector governed by individual chest geometry and tissue conductivities. 

To map 300-sample cardiac cycles into a discriminative metric space, Additive Angular Margin Loss (ArcFace) constrains feature embeddings $\mathbf{x}_i \in \mathbb{R}^d$ and normalized class weight vectors $\mathbf{W}_j$ onto a hypersphere of radius $s$, penalizing the target angle $\theta_{y_i}$ with margin $m$ [3]:
$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}} \tag{2}$$

### 2.2 Related Work
Early biometric ECG literature concentrated on fiducial temporal markers (PR intervals, ST elevations, and QRS durations) [4]. However, automated fiducial localization degrades severely under tachycardia and motion noise. Plataniotis et al. [5] introduced non-fiducial methods based on Autocorrelation (AC) and Discrete Cosine Transform (DCT) projections classified via Support Vector Machines, providing baseline shift invariance.

Recent investigations explore deep 1D Convolutional Neural Networks (CNNs) for automated feature learning directly from raw voltage samples [6]. While effective, existing networks employ hundreds of thousands of parameters, necessitating high-power Linux single-board computers (such as Raspberry Pi 4). Furthermore, metric learning formulations have demonstrated superior open-set verification clustering in biometric authentication [3], yet remain largely unexplored for bare-metal microcontroller deployment.

### 2.3 Gap and Motivation
Existing research overwhelmingly relies on non-causal zero-phase offline filtering (`filtfilt`), unviable for real-time edge streaming, and neglects the critical open-set false acceptance dynamics that emerge when probing un-enrolled impostors against large template galleries on resource-constrained embedded microcontrollers.

---

## 3. Methodology

### 3.1 Overall Approach
The complete hardware-software system follows an Asymmetric Multiprocessing (AMP) edge topology (Figure 1). Core 0 samples the 3.3V AD8232 front-end at 500 Hz via a hardware timer interrupt, executes causal IIR filtering, and segments SQI-verified beats into a lock-free ring buffer. Core 1 wakes upon receiving a hardware mailbox FIFO interrupt, executes pure float32 inference through `ECG_ResNet_SE`, and computes cosine similarity against enrolled templates in SRAM.

![Figure 1: End-to-End System and Edge MCU Architecture](figures/pipeline.svg)

### 3.2 Data and Preprocessing
We evaluate on the PhysioNet **ECG-ID Database v1.0.0** (310 records, 90 subjects, 500 Hz). Subject `Person_74` is excluded due to possessing only one recording session, yielding 89 evaluation subjects.
1. **Strict Recording-Disjoint Partitioning:** To guarantee zero data leakage, each subject's chronologically final recording is quarantined exclusively for the held-out test split (2,288 beats). All prior recordings form the training split (5,587 beats).
2. **Causal Filtering:** Signals pass through a causal 4th-order Direct Form II Butterworth bandpass filter ($0.5\text{--}40\text{ Hz}$) and a 50 Hz biquad notch filter ($Q=15$).
3. **Segmentation & SQI Gating:** Pan-Tompkins derivative squaring and 100 ms moving window integration locate R-peaks. Windows of 300 samples (0.2 s pre-R to 0.4 s post-R) are extracted, validated against a 5-point SQI gate (amplitude $0.15\text{--}3.0\text{ mV}$, HR $40\text{--}180\text{ BPM}$, drift $<0.5\text{ mV/s}$, no rail clipping, $2\text{--}18$ zero-crossings), and z-score standardized.

### 3.3 Model Design & The Dual-Stream Decision
Our architectural design progressed through systematic hypothesis testing:
* **`ECG_ResNet_SE` (14,867 Parameters):** A 1D convolutional stem (`Conv1D(1->16, k=15, s=2)`) is followed by three residual stages (depths 16, 32, 48; kernels 7, 5, 3) incorporating Squeeze-and-Excitation (SE) channel attention ($r=4$):
  $$\mathbf{s} = \sigma\Big(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z})\Big), \quad \tilde{\mathbf{X}} = \mathbf{s} \odot \mathbf{X} \tag{3}$$
  Global Average Pooling yields a 48-D hyperspherical embedding vector ($\|\mathbf{e}\|_2 = 1$). All Batch Normalization layers are folded at compile time.
* **The Dual-Stream Decision:** In electrophysiology, Purkinje conduction velocity manifests as an instantaneous slew rate ($dV/dt$). While raw voltage $x[n]$ captures anatomical chamber geometry, the discrete derivative $\Delta x[n] = x[n] - x[n-1]$ explicitly encodes conduction velocity. We designed a 2-channel Dual-Stream ResNet ($[x[n], \Delta x[n]]$ in channels $0$ and $1$, 16,928 parameters) fusing phase-space coordinates $(V, dV/dt)$.
* **Deployment Trade-Off:** Single-stream `ECG_ResNet_SE` was selected for embedded RP2040 deployment because its 78.9 KB Flash and 72 KB SRAM footprint fits comfortably within the MCU budget, executing in 300.05 ms (30% duty cycle). Dual-Stream ResNet serves as the high-accuracy research frontier, increasing SRAM requirements to 97 KB and latency to ~420 ms while pushing multi-beat identification beyond 97%.

### 3.4 Implementation and Experimental Setup
The network was trained in PyTorch using AdamW for 60 epochs with cosine learning rate decay and the hybrid objective ($\mathcal{L}_{CE} + 0.5 \mathcal{L}_{ArcFace}$). The trained network was transcribed into pure C99 floating-point routines without external library dependencies and compiled via ARM GCC 13.3 (`-O3 -mcpu=cortex-m0plus`) within the Raspberry Pi Pico SDK.

**Table 1: Key Hyperparameters, Network Dimensions, and Hardware Constraints**

| Parameter / Specification | Value / Description |
|---|---|
| Input Beat Dimension | $1 \times 300$ samples ($0.6\text{ s} @ 500\text{ Hz}$) / 2-channel $[x, \Delta x]$ for Dual-Stream |
| Embedding Dimension ($d$) | 48 dimensions ($192\text{ bytes}$ per template in SRAM) |
| Total Trainable Parameters | 14,867 parameters (Single-Stream) / 16,928 parameters (Dual-Stream) |
| Computational Complexity | ~0.45 MFLOPs per heartbeat forward pass |
| Training Settings | 60 epochs, batch size 64, AdamW (lr=$10^{-3}$ cosine decay) |
| ArcFace Hyperparameters | Margin $m = 0.285\text{ rad} \, (16.33^\circ)$, Scale $s = 16.0$ |
| Target Edge Microcontroller | Raspberry Pi Pico WH (RP2040 Dual Cortex-M0+ @ 250 MHz) |
| Active SRAM Allocation | 161.6 KB total (72.0 KB static activation buffers, 1.5 KB 8-user gallery) |
| Soft-Float Latency / Power | 300.05 ms per beat (~75M soft-float cycles) / $<0.55\text{ W}$ continuous |

### 3.5 Evaluation Metrics
We assess closed-set biometric identification via **Rank-1** and **Rank-5** accuracy across the 89 test classes. Open-set biometric verification is evaluated via **Equal Error Rate (EER)**, defined where False Acceptance Rate (FAR) equals False Rejection Rate (FRR), Area Under the ROC Curve (AUC), and the Biometric Decidability Index ($d'$):
$$d' = \frac{|\mu_{\text{genuine}} - \mu_{\text{impostor}}|}{\sqrt{0.5\left(\sigma_{\text{genuine}}^2 + \sigma_{\text{impostor}}^2\right)}} \tag{4}$$

---

## 4. Results and Discussion

### 4.1 Quantitative Results and Comprehensive ML Statistics
Table 2 summarizes performance across baseline, recurrent, convolutional, and dual-stream architectures evaluated under identical recording-disjoint test conditions ($2,288\text{ beats}$).

**Table 2: Performance Comparison across Baseline, Progressive, and Proposed Models**

| Architecture | Model Size | RP2040 Latency | Single-Beat Rank-1 | 5-Beat Ensemble | Verification EER | Verification AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (AC/DCT + SVM)** | 286.0 KB | Memory Fault | 79.06% | 84.12% | 13.11% | 0.9324 |
| **Method A (Recurrent LSTM)** | 76.6 KB | Memory Fault | 49.17% | 55.40% | 10.01% | 0.9412 |
| **Simple 1D-CNN (INT8)** | 51.9 KB | ~35 ms | 71.37% | 78.20% | 7.32% | 0.9610 |
| **Augmented ResNet-1D** | 58.4 KB | ~80 ms | 87.50% | 92.10% | 4.90% | 0.9812 |
| **Proposed `ECG_ResNet_SE` (FP32)** | **78.9 KB** | **300.05 ms** | **90.87%** | **95.51%** | **4.50%** | **0.9886** |
| **Proposed `ECG_ResNet_SE` (INT8)** | **66.3 KB** | **~95 ms** | **90.43%** | **95.12%** | **4.68%** | **0.9871** |
| **Proposed Dual-Stream ResNet** | **81.0 KB** | **~420 ms** | **91.08%** | **96.63%** | **4.93%** | **0.9879** |

In-depth statistical evaluation across **2,288 genuine** and **201,344 impostor test pairs** establishes strong biometric separability:
* **Separability Distribution:** Genuine cosine similarity mean is $\mu_g = 0.8190$ ($\sigma_g = 0.1745$), whereas impostor similarity centers at $\mu_i = -0.0029$ ($\sigma_i = 0.2212$), yielding an exceptional decidability index of **$d' = 4.13$**.
* **Operational Verification Security:** At a strict $1.0\%$ False Accept Rate, the model achieves a True Accept Rate of **$93.62\%$** ($\text{TAR} @ 0.1\%\text{ FAR} = 88.07\%$). At the default threshold $\tau=0.50$, $\text{TAR} = 94.14\%$ and $\text{FAR} = 1.56\%$.
* **Longitudinal Cross-Session Invariance:** Testing across same-day recordings ($n=1,785$) yields $90.76\%$ Rank-1 accuracy, while multi-day longitudinal recordings ($n=503$ over weeks) achieves **$95.63\%$ accuracy** ($+4.87\%$ longitudinal stability), confirming that the network learns true invariant morphology rather than transient session noise.
* **Multi-Beat Voting Progression & The >97% Threshold:** Averaging embeddings over consecutive heartbeats progressively boosts identification: 1 beat ($90.87\%$ ResNet-SE / $91.08\%$ Dual-Stream), 3 beats ($93.85\%$ / $95.51\%$), and 5 beats ($95.51\%$ / $96.63\%$). Crucially, **when pushed to 8 beats (~6.4 seconds of ECG), the Dual-Stream model increases beyond 97% to 97.19%**, reaching **97.75% across 10 beats** (~8.0 seconds).

![Figure 2: Qualitative Analysis, Waveform Filtering, Verification ROC, and Bench Impostor Rejection](figures/figure2_qualitative.svg)

### 4.2 Qualitative Visualizations & Bench Findings
Figure 2 illustrates the physical and empirical characteristics of the deployed edge system:
1. **50 Hz Mains Hum Attenuation (Figure 2a):** In bench recordings, 50 Hz power-line interference comprised 72.7% of in-band noise, triggering false R-peaks. The $Q=15$ IIR notch filter completely attenuated mains hum while preserving true QRS slopes.
2. **Verification ROC Progression (Figure 2b):** The proposed network achieves an AUC of 0.9886, dramatically shifting the ROC operating curve toward ideal sensitivity.
3. **Ensemble Accuracy (Figure 2c):** Majority voting over 5 consecutive beats (~4 s acquisition) elevates Rank-1 identification accuracy from 90.87% to **95.51%**, with Dual-Stream surpassing **97.19% at 8 beats**.
4. **Open-Set Multiple Comparison Discovery (Figure 2d):** When probing an unknown human subject (`P2`) against an 89-subject gallery at threshold $\tau = 0.50$, random tail projections triggered false matches. Implementing gallery scope isolation (`CMD:SCOPE LIVE`) and setting the strict threshold $\tau = 0.68$ completely eliminated false acceptances (0/5 ACCEPT for impostor `P2` vs. 5/5 ACCEPT for genuine `P1`).

### 4.3 Hardware Capabilities and Model Limitations
* **Compute Capabilities & Bottlenecks:** At 250 MHz, the Cortex-M0+ executes 0.45 MFLOPs in 300.05 ms (~75M clock cycles). The primary latency bottleneck is the lack of a hardware FPU, requiring GCC software emulation routines taking 80–120 cycles per operation. However, because cardiac cycles occur once per second, 300 ms utilizes only 30% of Core 1's temporal budget, allowing real-time streaming without queuing.
* **Memory & Storage Boundaries:** The model fits comfortably within the 264 KB SRAM envelope: static activation buffers consume 72.0 KB in BSS, and live 48-D user templates occupy only 192 bytes each (1.5 KB for 8 users). Zero Flash writes occur at runtime, preserving NOR Flash endurance indefinitely. Dual-Stream increases activation memory to 97 KB and latency to ~420 ms, making single-stream `ECG_ResNet_SE` the optimal trade-off for 3-second edge unlock.
* **Tachycardia & Sensitivity Limits:** Heart-rate sensitivity ablations indicate that natural heart-rate shifts of $\pm 5\text{ BPM}$ incur minimal accuracy loss ($<2.5\%$), but severe tachycardia ($>85\text{ BPM}$, $+18\%$) compresses the QT interval, causing single-beat accuracy to drop to $42.31\%$ unless dynamic tempo normalization is applied.

---

## 5. Conclusion
We have demonstrated a complete, end-to-end deep learning ECG biometric authentication pipeline deployed on an ultra-low-cost Raspberry Pi Pico WH. Following a progressive 6-stage optimization journey from classical AC/DCT SVMs and recurrent LSTMs to residual convolutional attention, our single-stream `ECG_ResNet_SE` achieves 95.51% 5-beat identification accuracy, a 4.50% verification EER, and a decidability index of $d' = 4.13$ under strict recording-disjoint evaluation. When expanded to a Dual-Stream architecture fusing voltage morphology with phase velocity ($dV/dt$), multi-beat accuracy surpasses 97% at 8 beats (97.19%) and reaches 97.75% at 10 beats. Operating within 161.6 KB of SRAM without a hardware FPU, the prototype verifies genuine live users and rejects human impostors in real time with zero Flash wear. Future work will investigate ARM Helium vectorization on Cortex-M33 (RP2350) and long-term longitudinal cardiac drift adaptation across multi-month intervals.

---

## Acknowledgements
The authors thank Dr. Ritesh Vyas and the teaching assistants of EEE G513 for guidance and compute facilities.

---

## References
[1] A. K. Jain, K. Nandakumar, and A. Ross, "50 years of biometric research: Accomplishments, challenges, and opportunities," *Pattern Recognit. Lett.*, vol. 79, pp. 80–105, 2016.  
[2] R. D. Labati, E. Muñoz, V. Piuri, R. Sassi, and F. Scotti, "Deep-ECG: Convolutional neural networks for ECG biometric recognition," *Pattern Recognit. Lett.*, vol. 126, pp. 78–85, 2019.  
[3] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4690–4699.  
[4] L. Biel, O. Pettersson, L. Philipson, and P. Wide, "ECG analysis: A new approach in human identification," *IEEE Trans. Instrum. Meas.*, vol. 50, no. 3, pp. 808–812, Jun. 2001.  
[5] K. N. Plataniotis, D. Hatzinakos, and J. K. M. Lee, "ECG analysis for individual identification," in *Proc. 28th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBS)*, New York, NY, 2006, pp. 3241–3244.  
[6] T. S. Lugovaya, "Biometric human identification based on electrocardiogram," Master's thesis, Faculty of Computing Technologies and Informatics, Saint-Petersburg Electrotechnical Univ., Saint-Petersburg, Russia, 2005.  
[7] J. Pan and W. J. Tompkins, "A real-time QRS detection algorithm," *IEEE Trans. Biomed. Eng.*, vol. BME-32, no. 3, pp. 230–236, Mar. 1985.  
[8] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 7132–7141.  

---

## Annex A: Team Contributions

| Team Member | Contribution Description | Deliverable Role |
|---|---|---|
| **Student 1** | Formulated research problem, designed causal IIR Butterworth bandpass and 50 Hz notch filter DSP pipeline, implemented Pan-Tompkins QRS detection and 5-point SQI gating. | Report (Sec 1, 2, 3.1–3.2) / Code (`ecg_dsp.h`) / Poster (Theory & Pipeline) |
| **Student 2** | Architected `ECG_ResNet_SE` and Dual-Stream networks with Squeeze-and-Excitation attention and ArcFace metric loss, conducted progressive benchmarking across all 6 model stages. | Report (Sec 3.3, 3.5, 4.1) / Code (`src/mlee/`) / Poster (Architecture & Results) |
| **Student 3** | Transcribed model to bare-metal C99, implemented RP2040 dual-core AMP firmware and Wi-Fi streaming, characterized soft-float execution, SRAM envelopes, and bench testing. | Report (Sec 3.4, 4.2–4.3, 5) / Firmware (`hardware/firmware/`) / Poster Layout |
