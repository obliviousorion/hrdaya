# Your Heartbeat Is Your Password: Lightweight Deep ECG Biometrics for Edge Authentication

EEE G513 — team names/IDs: TODO — instructor: TODO — code URL: TODO

## Abstract (120–150 words)
TODO: problem, protocol, models, measured results and limitations.

## 1. Introduction (250–350 words)
TODO: motivation, research question, committed scope and contributions.

## 2. Background and related work (300–400 words)
TODO: AC/DCT, CNN, LSTM, verification and 4–8 relevant IEEE-cited sources.

## 3. Methodology (400–500 words)
TODO: ECG-ID attribution, raw channel/filter, validated reader and R-peaks,
exclusions/date-aware cohorts, frozen splits, seeds, train-only preprocessing,
architectures, configuration, development-only thresholds, Pi measurement protocol.

![Pipeline](figures/pipeline.svg)

## 4. Results and discussion (500–600 words)

| Method | Rank-1 (%) | EER (%) | Model KB | Pi median/p95 ms |
|---|---:|---:|---:|---|
| AC/DCT + SVM (recording-disjoint diagnostic) | 79.06 | 13.11 | not measured | not measured |
| LSTM | pending | pending | pending | pending |
| CNN FP32 | pending | pending | pending | pending |
| CNN INT8 | pending | pending | pending | pending |

TODO: match JSON and leaderboard; compare identical cohorts; distinguish same-day,
true different-day and current recording-disjoint diagnostic. Include 2–4 figures/tables,
ROC/EER trial counts, 1/3/5/10-beat ablation, error analysis and limitations.
Test EER is descriptive; validation-selected threshold required for deployment.

![Confusion matrix](figures/confusion.svg)

## 5. Conclusion (150–200 words)
TODO: supported findings, limits, future work; no aspirational proposal numbers.

## References (IEEE numbered)
[1] T. S. Lugovaya, “Biometric human identification based on electrocardiogram,” master's thesis, LETI, 2005.
[2] ECG-ID Database v1.0.0, PhysioNet, doi:10.13026/C2J01F. ODC-By 1.0.
TODO: PhysioNet standard citation and method sources, verify bibliographic details.

## Annex A: Individual contributions

| Team member / ID | Code and experiments | Report | Poster |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

Use the supplied DOCX template for final formatting, export review PDF, remove all TODOs
and instructional content; no Annex B. Confirm course upload format before delivery.
