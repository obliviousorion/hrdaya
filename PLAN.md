# Shared delivery plan

Final report + A3 colour poster + running prototype: **20 November 2026, firm**.
Freeze substantive work on **18 November**, reserve 19–20 November for repairs and delivery.
[Task ownership and acceptance checks](TASKS.md) govern the shared work.

Committed scope: ECG-ID only; AC/DCT+SVM, LSTM and compact 1D-CNN comparison;
rank-1/macro accuracy, cosine verification ROC/EER, genuine different-day versus same-day
cohorts, 1/3/5/10-beat decisions, FP32/INT8 size and measured Raspberry Pi latency,
offline ACCEPT/REJECT replay. Live acquisition, external pretraining and ESP32 work are optional.

Use raw channel 0, 0.5–40 Hz filtering and 0.6 s windows. Validate the reader against wfdb,
inspect five records' R-peaks, inventory dates/exclusions and freeze splits before tuning.
Every recording/date stays within one partition; identity enrollment must be present for
closed-set testing. Train preprocessing on training only. Thresholds come from development data.
Do not label recording-disjoint results as cross-day, or laptop timing as Pi timing.
Use seed 42 initially; report all seeds if additional runs fit. CNN/LSTM: batch 64,
maximum 30 epochs, development-only early stopping. No test-based selection.

## Report requirements (located templates, no numeric marking grid found)

Title/team/IDs/instructor/code link; abstract 120–150 words; introduction 250–350;
background 300–400 with 4–8 sources; methodology 400–500 with pipeline figure;
results/discussion 500–600 with 2–4 figures/tables; conclusion 150–200;
IEEE references and Annex A individual contributions. About 1,720–2,200 main-text words
are guidance, not a hard page limit. Remove instructional Annex B.
Report and poster carry equal weight according to the supplied template.
Final report follows the supplied DOCX template; export a review PDF and confirm upload format.

A3 colour poster must match report/JSON numbers, include Baseline/Method A/Proposed,
error analysis/ablation, limitations, 3–5 references and tested code QR.
The supplied LaTeX template is A0; the scaffold here targets A3 and needs a legibility proof.
Confirm exact submitted title/team (maximum three), current course notices, deadline hour,
Pi availability and printing arrangements. Keep an offline laptop replay fallback and
report any missing Pi measurement explicitly. No aspirational proposal metrics are results.
