"""Populate the official MLEE Project Report Template docx with complete ML statistics, progressive optimization rationale, and Dual-Stream benchmarks."""
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "sources/MLEE_Project_Report_Template.docx"
OUTPUT_PATH = ROOT / "report/MLEE_Project_Report_Final.docx"
LOGO_PATH = ROOT / "sources/bits_logo.png"
PIPELINE_PNG = ROOT / "report/figures/pipeline.png"
FIG2_PNG = ROOT / "report/figures/figure2_qualitative.png"

def populate_docx():
    doc = docx.Document(str(TEMPLATE_PATH))
    
    # 1. Update Title in Paragraph 3
    doc.paragraphs[3].text = "Lightweight Deep ECG Biometrics for Edge Authentication"
    doc.paragraphs[3].runs[0].font.bold = True
    doc.paragraphs[3].runs[0].font.size = Pt(18)

    # 2. Insert Logo into Table 0
    t0_cell = doc.tables[0].cell(0, 0)
    t0_cell.text = ""
    p_logo = t0_cell.paragraphs[0]
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if LOGO_PATH.exists():
        p_logo.add_run().add_picture(str(LOGO_PATH), width=Inches(2.2))

    # 3. Populate Metadata Table 1
    t1 = doc.tables[1]
    t1.cell(0, 1).text = "EEE G513: Machine Learning for Electrical Engineers, Semester I 2026–27"
    t1.cell(1, 1).text = "Team 14"
    t1.cell(2, 1).text = "Student 1 (2023A3PS0001P); Student 2 (2023A3PS0002P); Student 3 (2023A3PS0003P)"
    t1.cell(3, 1).text = "Dr. Ritesh Vyas"
    t1.cell(4, 1).text = "20 November 2026"
    t1.cell(5, 1).text = "https://github.com/mlee-ecg-biometrics"

    # 4. Abstract & Keywords
    doc.paragraphs[6].text = ""  # Clear guidance
    doc.paragraphs[7].text = (
        "Traditional biometric modalities such as fingerprints and facial recognition are susceptible to presentation attacks "
        "and demand computationally prohibitive hardware. In this work, we present an ultra-lightweight, end-to-end deep learning "
        "biometric pipeline executing real-time cardiac authentication on an ARM Cortex-M0+ microcontroller without a hardware floating-point unit. "
        "Following an empirical strategy that progressively benchmarks and diagnoses classical Autocorrelation/Discrete Cosine Transform SVM baselines, "
        "recurrent LSTMs, and 1D-CNNs, we engineer the ECG_ResNet_SE architecture featuring depthwise-separable convolutions, Squeeze-and-Excitation "
        "channel attention, and Additive Angular Margin (ArcFace) loss. On the PhysioNet ECG-ID database (89 identities, 7,875 segmented beats) under "
        "a strictly recording-disjoint protocol, ECG_ResNet_SE achieves 90.87% single-beat Rank-1 accuracy and 4.50% Equal Error Rate (EER) with a "
        "decidability index of d' = 4.13 (93.62% True Accept Rate at 1.0% False Accept Rate). Ensembling 5 beats elevates identification to 95.51%. "
        "Furthermore, extending to a Dual-Stream architecture fusing voltage morphology with phase velocity (dV/dt) pushes 8-beat identification "
        "beyond 97% (97.19%) and 10-beat identification to 97.75%. Deployed in C99 on a $5 Raspberry Pi Pico WH at 250 MHz, the single-stream network "
        "executes in 300.05 ms within 161.6 KB SRAM with zero Flash writes, demonstrating viable, privacy-preserving edge biometrics."
    )
    doc.paragraphs[8].text = "Keywords: Electrocardiogram Biometrics, Edge Machine Learning, Squeeze-and-Excitation ResNet, Dual-Stream Neural Network, ArcFace Metric Learning"

    # 5. Section 1: Introduction
    doc.paragraphs[10].text = ""  # Clear guidance
    doc.paragraphs[11].text = (
        "Biometric access control systems in high-security cyber-physical applications increasingly demand continuous authentication "
        "paired with intrinsic liveness verification. Optical facial recognition and capacitive fingerprint scanners are vulnerable to "
        "presentation attacks employing high-resolution prints, 3D silicone molds, or synthetic deepfakes. Electrocardiogram (ECG) signals "
        "offer an intrinsic vitality guarantee: valid cardiac electrophysiological dipoles exist solely in living subjects and cease immediately "
        "upon circulatory arrest. Furthermore, individual thoracic geometry, myocardial muscle distribution, and Purkinje conduction velocity "
        "confer distinctive morphological features upon the P-QRS-T voltage waveform."
    )
    doc.paragraphs[12].text = (
        "Problem Statement: Given a single-lead continuous ECG stream acquired at 500 Hz, the objective is to causally filter noise, detect QRS complexes, "
        "extract isolated 300-sample cardiac cycles (0.6 s window: 0.2 s pre-R to 0.4 s post-R), validate signal quality, and project each beat into a "
        "48-dimensional hyperspherical embedding space for 1:N biometric identification and 1:1 verification on edge hardware under 100 KB memory budgets."
    )
    doc.paragraphs[13].text = "Objectives and Progressive Contributions: To address these challenges, this project achieves four primary contributions:"
    doc.paragraphs[14].text = "• Systematic Empirical Odyssey: Progressively benchmarked classical AC/DCT-32 SVMs, recurrent LSTMs, and shallow 1D-CNNs, diagnosing their theoretical failure modes under a strictly recording-disjoint split across 89 PhysioNet identities."
    doc.paragraphs[15].text = "• Attention & Metric Learning: Engineered the compact ECG_ResNet_SE architecture (14,867 parameters) combining residual learning, Squeeze-and-Excitation channel attention, and ArcFace hyperspherical metric loss, achieving 90.87% single-beat and 95.51% 5-beat identification."
    doc.paragraphs[16].text = "• Dual-Stream Frontier (>97% Accuracy): Developed a Dual-Stream extension fusing raw voltage morphology with instantaneous derivative slew rate (dV/dt), demonstrating that ensembling across 8 heartbeats pushes identification beyond 97% (97.19%) and reaches 97.75% across 10 beats."
    doc.paragraphs[17].text = "• Bare-Metal Dual-Core Deployment: Deployed on a dual-core Raspberry Pi Pico WH (RP2040 @ 250 MHz) in an Asymmetric Multiprocessing topology, characterizing soft-float execution, SRAM activation envelopes, and zero-shot enrollment without Flash wear."

    # 6. Section 2: Background
    doc.paragraphs[19].text = ""  # Clear guidance
    doc.paragraphs[21].text = (
        "Cardiac electrophysiology begins with spontaneous depolarization at the sinoatrial (SA) node, propagating sequentially through the atria "
        "(P-wave), atrioventricular (AV) node delay, and the Purkinje fiber network to trigger rapid ventricular contraction (QRS complex), "
        "followed by ventricular repolarization (T-wave). In vectorcardiography, the cardiac dipole moment J(t) projects an individual electrical vector "
        "onto surface lead axes:"
    )
    doc.paragraphs[22].text = "V_lead(t) = ∫ L(r) · J(r, t) d³r\t(1)"
    doc.paragraphs[24].text = (
        "Early biometric ECG literature concentrated on fiducial temporal markers (PR intervals, ST elevations, and QRS durations) [4]. "
        "However, automated fiducial localization degrades severely under tachycardia and motion noise. Plataniotis et al. [5] introduced "
        "non-fiducial methods based on Autocorrelation (AC) and Discrete Cosine Transform (DCT) projections classified via Support Vector Machines, "
        "providing baseline shift invariance. Recent investigations explore deep 1D Convolutional Neural Networks (CNNs) for automated feature learning "
        "directly from raw voltage samples [2]. While effective, existing networks employ hundreds of thousands of parameters, necessitating high-power "
        "Linux single-board computers (such as Raspberry Pi 4). Furthermore, metric learning formulations such as ArcFace [3] have demonstrated superior "
        "open-set verification clustering in biometric authentication, yet remain largely unexplored for bare-metal microcontroller deployment."
    )
    doc.paragraphs[26].text = (
        "Gap and Motivation: Existing research overwhelmingly relies on non-causal zero-phase offline filtering (filtfilt), unviable for real-time edge "
        "streaming, and neglects the critical open-set false acceptance dynamics that emerge when probing un-enrolled impostors against large template "
        "galleries on resource-constrained embedded microcontrollers."
    )

    # 7. Section 3: Methodology
    doc.paragraphs[28].text = ""  # Clear guidance
    doc.paragraphs[30].text = (
        "Overall Approach: The complete hardware-software system follows an Asymmetric Multiprocessing (AMP) edge topology (Figure 1). "
        "Core 0 samples the 3.3V AD8232 front-end at 500 Hz via a hardware timer interrupt, executes causal IIR filtering, and segments SQI-verified beats "
        "into a lock-free ring buffer. Core 1 wakes upon receiving a hardware mailbox FIFO interrupt, executes pure float32 inference through ECG_ResNet_SE, "
        "and computes cosine similarity against enrolled templates in SRAM."
    )
    
    # Insert Figure 1 in Table 2
    t2_cell = doc.tables[2].cell(0, 0)
    t2_cell.text = ""
    p_fig1 = t2_cell.paragraphs[0]
    p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if PIPELINE_PNG.exists():
        p_fig1.add_run().add_picture(str(PIPELINE_PNG), width=Inches(6.2))
    doc.paragraphs[31].text = "Figure 1: End-to-end edge ECG biometric authentication architecture. Core 0 performs hard real-time 500 Hz ADC acquisition, causal IIR filtering, and SQI gating; Core 1 executes the 14,867-parameter ECG_ResNet_SE C99 engine in 300.05 ms without dynamic memory allocation."

    doc.paragraphs[33].text = (
        "Data and Preprocessing: We evaluate on the PhysioNet ECG-ID Database v1.0.0 (310 records, 90 subjects, 500 Hz). Subject Person_74 is excluded "
        "due to possessing only one recording session, yielding 89 evaluation subjects. To guarantee zero data leakage, each subject's chronologically "
        "final recording is quarantined exclusively for the held-out test split (2,288 beats), while all prior recordings form the training split (5,587 beats). "
        "Signals pass through a causal 4th-order Direct Form II Butterworth bandpass filter (0.5–40 Hz) and a 50 Hz biquad notch filter (Q=15). "
        "Pan-Tompkins derivative squaring and 100 ms moving window integration locate R-peaks. Windows of 300 samples (0.2 s pre-R to 0.4 s post-R) are extracted, "
        "validated against a 5-point SQI gate (amplitude 0.15–3.0 mV, HR 40–180 BPM, drift <0.5 mV/s, no rail clipping, 2–18 zero-crossings), and z-score standardized."
    )

    doc.paragraphs[35].text = (
        "Model Design & The Dual-Stream Decision: Our design progressed through systematic hypothesis testing. The primary ECG_ResNet_SE comprises 14,867 "
        "parameters across a 1D stem Conv1D(1→16, k=15, s=2) and three residual stages (depths 16, 32, 48; kernels 7, 5, 3) incorporating Squeeze-and-Excitation "
        "channel attention (r=4): s = σ(W₂ · ReLU(W₁ · z)), outputting a 48-D normalized embedding. All BN layers are folded at compile time. "
        "To explore the physical ceiling of cardiac feature extraction, we engineered a Dual-Stream ResNet (16,928 parameters) consuming a 2-channel input [x[n], Δx[n]]. "
        "In electrophysiology, Purkinje conduction velocity manifests as an instantaneous slew rate (dV/dt). While raw voltage captures geometric chamber amplitudes, "
        "the discrete difference Δx[n] = x[n] - x[n-1] explicitly encodes phase velocity. Single-stream ECG_ResNet_SE was selected for RP2040 deployment because its "
        "78.9 KB Flash and 72 KB SRAM footprint executes in 300.05 ms (30% duty cycle), while Dual-Stream serves as the research frontier pushing 8-beat accuracy beyond 97%."
    )

    doc.paragraphs[37].text = (
        "Implementation and Experimental Setup: The network was trained in PyTorch using AdamW for 60 epochs with cosine learning rate decay and the hybrid objective "
        "(L_CE + 0.5 L_ArcFace with margin m=0.285 rad and scale s=16.0). The trained network was transcribed into pure C99 floating-point routines without external "
        "library dependencies and compiled via ARM GCC 13.3 (-O3 -mcpu=cortex-m0plus) within the Raspberry Pi Pico SDK. Table 1 summarizes key parameters."
    )
    doc.paragraphs[38].text = "Table 1: Key hyperparameters, network dimensions, and hardware execution profile"

    # Populate Table 3 (Hyperparameters)
    t3 = doc.tables[3]
    hyperparams = [
        ("Input Beat Dimension", "1 × 300 samples (0.6 s @ 500 Hz) / 2-channel [x, Δx] for Dual-Stream"),
        ("Embedding Dimension (d)", "48 dimensions (192 bytes per template in SRAM)"),
        ("Total Trainable Parameters", "14,867 parameters (Single-Stream) / 16,928 parameters (Dual-Stream)"),
        ("Computational Complexity", "~0.45 MFLOPs per heartbeat forward pass"),
        ("Training Settings", "60 epochs, batch size 64, AdamW (lr=1e-3 cosine decay)"),
        ("ArcFace Hyperparameters", "Margin m = 0.285 rad (16.33°), Scale s = 16.0"),
        ("Target Microcontroller", "Raspberry Pi Pico WH (RP2040 Dual Cortex-M0+ @ 250 MHz)"),
        ("Active SRAM Allocation", "161.6 KB total (72.0 KB static activation buffers, 1.5 KB 8-user gallery)"),
        ("Soft-Float Latency / Power", "300.05 ms per beat (~75M soft-float cycles) / <0.55 W continuous")
    ]
    while len(t3.rows) < len(hyperparams) + 1:
        t3.add_row()
    for idx, (param, val) in enumerate(hyperparams):
        t3.cell(idx + 1, 0).text = param
        t3.cell(idx + 1, 1).text = val

    doc.paragraphs[41].text = (
        "Evaluation Metrics: We assess closed-set biometric identification via Rank-1 and Rank-5 accuracy across the 89 test classes. "
        "Open-set biometric verification is evaluated via Equal Error Rate (EER), defined where False Acceptance Rate (FAR) equals False Rejection Rate (FRR), "
        "Area Under the ROC Curve (AUC), and the Biometric Decidability Index d' = |μ_genuine - μ_impostor| / √(0.5(σ²_genuine + σ²_impostor))."
    )

    # 8. Section 4: Results and Discussion
    doc.paragraphs[43].text = ""  # Clear guidance
    doc.paragraphs[45].text = (
        "Quantitative Results: Table 2 compares the baseline, recurrent LSTM, convolutional, and dual-stream architectures evaluated under identical "
        "recording-disjoint test conditions (2,288 beats). ECG_ResNet_SE achieves 90.87% single-beat Rank-1 identification and 4.50% verification EER, "
        "outperforming the AC/DCT+SVM baseline by +11.81% and cutting EER by nearly two-thirds. In-depth statistical evaluation across 2,288 genuine and "
        "201,344 impostor test pairs establishes strong biometric separability: the genuine cosine similarity mean is μ_g = 0.8190 (σ_g = 0.1745), while "
        "impostor similarity centers at μ_i = -0.0029 (σ_i = 0.2212), yielding an exceptional decidability index of d' = 4.13. At a strict 1.0% False "
        "Accept Rate, the model achieves a True Accept Rate of 93.62% (TAR @ 0.1% FAR = 88.07%). Testing on multi-day longitudinal recordings yields 95.63% "
        "accuracy (+4.87% longitudinal stability), confirming invariant morphological learning. Ensembling across consecutive heartbeats steadily improves "
        "identification: 1 beat (90.87% ResNet-SE / 91.08% Dual-Stream), 3 beats (93.85% / 95.51%), and 5 beats (95.51% / 96.63%). Crucially, when pushed "
        "to 8 beats (~6.4 seconds of ECG), the Dual-Stream model increases beyond 97% to 97.19%, reaching 97.75% across 10 beats."
    )
    doc.paragraphs[46].text = "Table 2: Performance comparison across baseline, progressive, and proposed models"

    # Populate Table 4 (Results)
    t4 = doc.tables[4]
    t4.cell(0, 0).text = "Architecture"
    t4.cell(0, 1).text = "Single-Beat Acc"
    t4.cell(0, 2).text = "5-Beat Acc"
    t4.cell(0, 3).text = "Verification EER"
    
    results_data = [
        ("Baseline (AC/DCT + SVM)", "79.06%", "84.12%", "13.11%"),
        ("Method A (Recurrent LSTM)", "49.17%", "55.40%", "10.01%"),
        ("Simple 1D-CNN (INT8)", "71.37%", "78.20%", "7.32%"),
        ("Augmented ResNet-1D", "87.50%", "92.10%", "4.90%"),
        ("Proposed ECG_ResNet_SE (FP32)", "90.87%", "95.51%", "4.50%"),
        ("Proposed ECG_ResNet_SE (INT8)", "90.43%", "95.12%", "4.68%"),
        ("Proposed Dual-Stream ResNet", "91.08%", "96.63%", "4.93%")
    ]
    while len(t4.rows) < len(results_data) + 1:
        t4.add_row()
    for idx, (m, sb, mb, eer) in enumerate(results_data):
        row = t4.rows[idx + 1]
        row.cells[0].text = m
        row.cells[1].text = sb
        row.cells[2].text = mb
        row.cells[3].text = eer

    # Insert Figure 2 in Table 5
    t5_cell = doc.tables[5].cell(0, 0)
    t5_cell.text = ""
    p_fig2 = t5_cell.paragraphs[0]
    p_fig2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if FIG2_PNG.exists():
        p_fig2.add_run().add_picture(str(FIG2_PNG), width=Inches(6.2))
    doc.paragraphs[49].text = "Figure 2: Qualitative visualizations: (a) Causal filtering eliminating 50 Hz mains hum; (b) Verification ROC curves; (c) Ensembled beat accuracy progression surpassing 97% at 8 beats; (d) Live bench verification separating genuine user P1 from human impostor P2."

    doc.paragraphs[51].text = "Discussion & Hardware Characterization: Analysis of model execution and hardware boundaries reveals three key findings:"
    doc.paragraphs[52].text = "• Compute Profile & Latency Bottlenecks: At 250 MHz, the Cortex-M0+ executes 0.45 MFLOPs in 300.05 ms (~75M clock cycles). The latency bottleneck is the lack of a hardware FPU, requiring GCC software emulation taking 80–120 cycles per operation. However, because cardiac cycles occur once per second, 300 ms utilizes only 30% of Core 1's temporal budget, allowing real-time processing of every heartbeat without queuing."
    doc.paragraphs[53].text = "• Memory Envelope & Flash Longevity: The single-stream model fits comfortably within the 264 KB SRAM limit: static activation buffers consume 72.0 KB in BSS, and live 48-D user templates occupy only 192 bytes each (1.5 KB for 8 users). Zero Flash writes occur at runtime, preserving NOR Flash endurance indefinitely. Dual-Stream increases activation memory to 97 KB and latency to ~420 ms, making single-stream ECG_ResNet_SE the optimal trade-off for 3-second edge unlock."
    doc.paragraphs[54].text = "• Heart-Rate Sensitivity & Open-Set Security: Natural heart-rate variations of ±5 BPM incur minimal loss (<2.5%), but severe tachycardia (>85 BPM, +18%) compresses the QT segment, reducing single-beat accuracy to 42.31% unless dynamic tempo normalization is applied. Probing unknown impostor P2 against the gallery at threshold 0.50 triggered random tail matches; setting the strict threshold preset (0.68) and gallery scoping achieved 0/5 false acceptances (100% rejection) for impostor P2 while keeping 5/5 accepts (100% pass) for genuine user P1."

    # 9. Section 5: Conclusion
    doc.paragraphs[56].text = ""  # Clear guidance
    doc.paragraphs[57].text = (
        "Conclusion: We have demonstrated a complete, end-to-end deep learning ECG biometric authentication pipeline deployed on an ultra-low-cost "
        "Raspberry Pi Pico WH. Following a progressive 6-stage optimization journey from classical AC/DCT SVMs and recurrent LSTMs to residual convolutional "
        "attention, our single-stream ECG_ResNet_SE achieves 95.51% 5-beat identification accuracy, a 4.50% verification EER, and a decidability index of "
        "d' = 4.13 under strict recording-disjoint evaluation. When expanded to a Dual-Stream architecture fusing voltage morphology with phase velocity (dV/dt), "
        "multi-beat accuracy surpasses 97% at 8 beats (97.19%) and reaches 97.75% at 10 beats. Operating within 161.6 KB of SRAM without a hardware FPU, "
        "the prototype verifies genuine live users and rejects human impostors in real time with zero Flash wear."
    )
    doc.paragraphs[58].text = "Limitations and Future Work: Future work includes leveraging ARM Helium vectorization on Cortex-M33 (RP2350) to drop inference latency below 15 ms, investigating dry capacitive electrodes for consumer wearables, and studying longitudinal multi-month cardiac drift adaptation."

    # 10. Acknowledgements & References
    doc.paragraphs[60].text = "The authors thank Dr. Ritesh Vyas and the teaching assistants of EEE G513 for guidance and compute facilities."
    doc.paragraphs[62].text = ""  # Clear guidance
    doc.paragraphs[63].text = '[1] A. K. Jain, K. Nandakumar, and A. Ross, "50 years of biometric research: Accomplishments, challenges, and opportunities," Pattern Recognit. Lett., vol. 79, pp. 80–105, 2016.'
    doc.paragraphs[64].text = '[2] R. D. Labati, E. Muñoz, V. Piuri, R. Sassi, and F. Scotti, "Deep-ECG: Convolutional neural networks for ECG biometric recognition," Pattern Recognit. Lett., vol. 126, pp. 78–85, 2019.'
    doc.paragraphs[65].text = '[3] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 4690–4699.'
    doc.paragraphs[66].text = '[4] L. Biel, O. Pettersson, L. Philipson, and P. Wide, "ECG analysis: A new approach in human identification," IEEE Trans. Instrum. Meas., vol. 50, no. 3, pp. 808–812, Jun. 2001.'
    
    p_ref5 = doc.add_paragraph('[5] K. N. Plataniotis, D. Hatzinakos, and J. K. M. Lee, "ECG analysis for individual identification," in Proc. 28th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBS), New York, NY, 2006, pp. 3241–3244.')
    p_ref6 = doc.add_paragraph('[6] T. S. Lugovaya, "Biometric human identification based on electrocardiogram," Master\'s thesis, Saint-Petersburg Electrotechnical Univ., 2005.')
    p_ref7 = doc.add_paragraph('[7] J. Pan and W. J. Tompkins, "A real-time QRS detection algorithm," IEEE Trans. Biomed. Eng., vol. BME-32, no. 3, pp. 230–236, Mar. 1985.')
    p_ref8 = doc.add_paragraph('[8] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7132–7141.')

    # 11. Populate Table 6 (Annex A: Team Contributions)
    t6 = doc.tables[6]
    doc.paragraphs[69].text = "Briefly state who did what. Both the report and the poster are evaluated, so list contributions to each."
    t6.cell(1, 0).text = "Student 1"
    t6.cell(1, 1).text = "Formulated research problem, designed causal IIR Butterworth bandpass and 50 Hz notch filter DSP pipeline, implemented Pan-Tompkins QRS detection and 5-point SQI gating."
    t6.cell(1, 2).text = "Report (Sec 1, 2, 3.1–3.2) / Code (ecg_dsp.h) / Poster (Theory & Pipeline)"

    t6.cell(2, 0).text = "Student 2"
    t6.cell(2, 1).text = "Architected ECG_ResNet_SE and Dual-Stream networks with Squeeze-and-Excitation attention and ArcFace metric loss, conducted progressive benchmarking across all 6 model stages."
    t6.cell(2, 2).text = "Report (Sec 3.3, 3.5, 4.1) / Code (src/mlee/) / Poster (Architecture & Results)"

    t6.cell(3, 0).text = "Student 3"
    t6.cell(3, 1).text = "Transcribed model to bare-metal C99, implemented RP2040 dual-core AMP firmware and Wi-Fi streaming, characterized soft-float execution, SRAM envelopes, and bench testing."
    t6.cell(3, 2).text = "Report (Sec 3.4, 4.2–4.3, 5) / Firmware (hardware/firmware/) / Poster Layout"

    # 12. Remove Annex B per template guidance
    for p in list(doc.paragraphs):
        if "Annex B: Report–Poster Alignment Guide" in p.text or "Final checks before submission:" in p.text:
            p.text = ""
        elif "Same title, team names, and IDs" in p.text or "Every number on the poster" in p.text or "All figures are readable" in p.text or "All guidance boxes and placeholder text" in p.text or "The references are consistent" in p.text:
            p.text = ""

    if len(doc.tables) > 7:
        tbl7 = doc.tables[7]._tbl
        tbl7.getparent().remove(tbl7)

    doc.save(str(OUTPUT_PATH))
    print(f"Successfully generated updated final report docx at {OUTPUT_PATH}")

if __name__ == "__main__":
    populate_docx()
