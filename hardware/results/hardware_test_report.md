# Hardware Test Report — Raspberry Pi Pico WH

**Timestamp:** 2026-11-13T17:37:24Z  
**Port:** /dev/ttyACM0  
**Beats tested:** 5  
**Model:** `ECG_ResNet_SE` (14,867 params, BN-fused float32)  

---

## Per-Beat Results

| # | Identity | HW Sim | Ref Sim | Δsim | Latency (ms) | Decision |
|---|---|---|---|---|---|---|
| 1 | `Person_01` | 0.8346 | 0.8346 | 0.0000 | 584.66 | ACCEPT |
| 2 | `Person_01` | 0.8217 | 0.8217 | 0.0000 | 584.63 | ACCEPT |
| 3 | `Person_01` | 0.8340 | 0.8340 | 0.0000 | 584.67 | ACCEPT |
| 4 | `Person_01` | 0.8271 | 0.8271 | 0.0000 | 584.62 | ACCEPT |
| 5 | `Person_01` | 0.7764 | 0.7764 | 0.0000 | 585.45 | ACCEPT |

---

## Summary

| Metric | Value | Target | Status |
|---|---|---|---|
| Mean latency | 584.81 ms | ≤ 60 ms | FAIL |
| Max latency  | 585.45 ms  | ≤ 60 ms | FAIL |
| Sim accuracy (Δ < 0.005) | 5/5 | 5/5 | PASS |
| ACCEPT rate | 5/5 | — | — |
