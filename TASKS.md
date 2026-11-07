# Team tasks

Claim an item by filling Owner in a small PR before starting. Dependencies: DATA-02 before
final model scoring; EVAL-01 before EDGE-02; freeze results before poster proof.

## 8 November — data and protocol 2026

### DATA-01: Validate ECG reader and inspect peaks
Owner:
Status: open
Acceptance: Compare format-16 reader with wfdb.rdrecord; inspect five varied raw/filtered records and peaks; record discrepancies and exclusions.

### DATA-02: Freeze date-aware split manifests
Owner:
Status: open
Acceptance: Inventory record dates/hashes/counts, verify dates manually, publish true different-day and same-day cohorts with exclusions; prevent recording/date leakage.

### TEAM-01: Confirm title team and hardware
Owner:
Status: open
Acceptance: Recover submitted form title and team IDs (max three); confirm Pi model/access, upload format and deadline hour; keep shared docs project-only.

## 11 November — model comparison 2026

### MODEL-01: Implement compact CNN
Owner:
Status: open
Acceptance: Seed 42, compact Conv1D embedding, batch 64, at most 30 epochs, development-only early stop and train-only normalization; record frozen split/config/metrics.

### MODEL-02: Implement LSTM comparison
Owner:
Status: open
Acceptance: LSTM(32) with same windows/split/budget; compare rank-1 and macro accuracy to frozen classical baseline.

### EVAL-01: Verification and error analysis
Owner:
Status: open
Acceptance: Enrollment-only templates, ROC/EER and trial counts; development-selected threshold with held-out FAR/FRR; inspect confusion and failure cases. Start report sections 1–3.

## 14 November — ablations and deployment 2026

### EVAL-02: Evaluate day and beat ablations
Owner:
Status: open
Acceptance: Report true different-day versus same-day cohort/confounds; contiguous groups of 1/3/5/10 beats within records; never mix enrollment and probe beats.

### EDGE-01: Export and validate INT8
Owner:
Status: open
Acceptance: Training-only representative calibration; FP32/INT8 parity, accuracy/EER and serialized KB on same split; log conversion failures.

### EDGE-02: Build offline replay and measure Pi latency
Owner:
Status: open
Acceptance: Claimed-ID ACCEPT/REJECT using fixed development threshold; 20 warmups and at least 100 batch-1 timings, median/p95, hardware/threads/software and preprocessing scope; laptop fallback labelled honestly.

## 16 November — results and writing 2026

### WRITE-01: Freeze results and report draft
Owner:
Status: open
Acceptance: By 15 Nov export pipeline/ROC/confusion/ablation evidence; populate baseline/LSTM/CNN FP32/INT8 table without invented values; complete draft and limitations/IEEE references/Annex A by 16 Nov.

### WRITE-02: Draft A3 poster
Owner:
Status: open
Acceptance: Use same shared figures and metrics as report; show comparison, ablation/errors, limitations, 3–5 references and code QR; render A3 proof.

## 18 November — content freeze 2026

### RELEASE-01: Reproduce and proof final package
Owner:
Status: open
Acceptance: By 17 Nov clean-clone run, offline replay rehearsal and metric/title/team alignment; by 18 Nov finalize DOCX/review PDF, A3 PDF/source and running code; arrange colour print and verify QR.

## 19–20 November — delivery buffer 2026

### DELIVER-01: Submit and deliver final artifacts
Owner:
Status: open
Acceptance: Only reproduction/format/print/demo repairs after freeze; team confirms official cutoff and submits report, A3 poster and prototype by Fri 20 Nov; verify receipts.
