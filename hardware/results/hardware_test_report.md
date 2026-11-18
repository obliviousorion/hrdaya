# Hardware Test Report — Raspberry Pi Pico WH

**Timestamp:** 2026-11-18T09:17:31Z  
**Port:** /dev/ttyACM0  
**Beats tested:** 1  
**Model:** `ECG_ResNet_SE` (14,867 params, BN-fused float32)  

---

## Per-Beat Results

| # | Identity | HW Sim | Ref Sim | Δsim | Latency (ms) | Decision |
|---|---|---|---|---|---|---|
| 1 | `Impostor` | 0.3609 | 0.3609 | 0.0000 | 300.10 | REJECT |

---

## Summary

| Metric | Value | Target | Status |
|---|---|---|---|
| Mean latency | 300.10 ms | ≤ 60 ms | FAIL |
| Max latency  | 300.10 ms  | ≤ 60 ms | FAIL |
| Sim accuracy (Δ < 0.005) | 1/1 | 1/1 | PASS |
| ACCEPT rate | 1/1 | — | — |
