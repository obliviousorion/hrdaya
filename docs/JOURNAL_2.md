# Experimental & Architectural Research Journal — Part 2: Hardware Triage & Live AD8232 Edge Integration

**Project:** Lightweight Deep ECG Biometrics for Edge Authentication (`mlee-ecg-biometrics`)  
**Lead Engineer / Author:** Orion  
**Active Hardware Platform:** Raspberry Pi Pico WH (RP2040 Dual Cortex-M0+ @ 250 MHz AMP) + AD8232 Single-Lead ECG Front-End  
**Date:** 18 November 2026  
**Continuity:** Builds directly on [`private/JOURNAL.md`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/private/JOURNAL.md) (v1 Replay-Only Pico WH firmware)  
**Confidentiality:** Private Engineering & Research Journal (`private/JOURNAL_2.md`)

---

## 1. Pre-Modification Backup & Reversibility Guarantee

Before modifying any firmware from `JOURNAL.md` (v1 Dual-Core AMP USB-Replay build), a complete snapshot of the verified working binary and sources was archived to [`hardware/backups/v1_replay_only/`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/backups/v1_replay_only/):

| Archived File | Size | Purpose |
|---|---:|---|
| `pico_ecg_biometrics.uf2` | 824 KB | Verified v1 flashable binary (1-step rollback via BOOTSEL) |
| `main.c` | 9.8 KB | v1 Dual-Core AMP USB replay firmware |
| `wifi_server.c`, `wifi_server.h` | 14.8 KB | v1 read-only Wi-Fi AP dashboard |
| `CMakeLists.txt` | 1.5 KB | v1 Pico SDK build configuration |

**Hardware Safety / Non-Permanence Guarantees:**
* RP2040 USB BOOTSEL bootloader resides in permanent read-only silicon ROM (impossible to brick via firmware).
* Zero Flash writes occur at runtime — all live zero-shot user templates live in volatile SRAM and clear cleanly on reboot.
* Clock and VREG remain unchanged from verified v1 settings (`250 MHz @ 1.20V`).
* AD8232 draws ~170 µA from `3V3(OUT)` (well within the 300 mA regulator limit), and `GP26`/`GP16`/`GP17` are configured strictly as 3.3V inputs.

---

## 2. Hardware Inventory & Empirical Triage Audit

Five hardware components available on hand were evaluated against `ECG_ResNet_SE`'s exact deployment footprint (**78.9 KB** BN-fused weights + 89 templates in Flash/RODATA, **72 KB** static float32 activation buffers in SRAM, **500 Hz** 3.3V analog ECG input) and subjected to live bench health checks:

| Hardware | Specs vs. Model Fit | Empirical Triage Test & Observations | Verdict |
|---|---|---|---|
| **Raspberry Pi Pico WH (2022)** | RP2040 Dual Cortex-M0+ @ 250 MHz, 264 KB SRAM, 2 MB Flash, 12-bit 3.3V ADC, CYW43439 Wi-Fi | Previously flashed & verified (300.05 ms inference, $\Delta\text{sim}=0.0000$, active Wi-Fi AP). Fits breadboard cleanly. | **PASS — Primary Edge MCU** |
| **AD8232 ECG Front-End + 3-Lead Cable** | 3.3V analog instrumentation amp + `LO+`/`LO-` leads-off detection | Confirmed operational; native 3.3V logic match for Pico WH; fits breadboard cleanly. | **PASS — Live Biometric Sensor** |
| **ESP32 DevKit** | Dual Xtensa LX6 @ 240 MHz + HW FPU, 520 KB SRAM, 4+ MB Flash | Enumerated `CP2102` (`10c4:ea60`) on `/dev/ttyUSB0`, but MCU returned **0 bytes** across normal boot, download-mode reset, and SLIP SYNC probes. Board **heated up and emitted a burnt smell** while plugged in → shorted 3.3V regulator / dead silicon. Unplugged immediately. | **FAIL — Retired** |
| **Arduino Nano** | ATmega328P @ 16 MHz, **2 KB SRAM / 32 KB Flash**, 10-bit **5V** ADC | Did **not** enumerate in `lsusb`; onboard `PWR` LED blinked a few times upon connecting and then went completely dead → tripping regulator/polyfuse or short. Also incapable of holding 79 KB weights / 72 KB activations and unsafe to wire directly to 3.3V AD8232. | **FAIL — Retired** |
| **Raspberry Pi 4 Model B** | Quad Cortex-A72 @ 1.5 GHz, Linux SBC, **no built-in ADC** | Powered via USB-C with no SD card: **both red PWR and green ACT LEDs stayed solid ON** (healthy Pi 4B without SD card blinks a repeating 4-flash green pattern). Indicates corrupted SPI EEPROM or damaged 3.3V PMIC/SoC. No microSD card or HDMI display available. | **FAIL / BLOCKED — Retired** |

**Outcome:** Hardware triage prevented wasted firmware effort on damaged boards and protected the host USB ports. The verified, breadboard-compatible hardware pair is **Raspberry Pi Pico WH + AD8232**, which also maximally reinforces the project's core academic thesis: *complete edge biometric privacy on a \$5 microcontroller without an FPU*.

---

## 3. Architecture Selection: Option 3 — Dual-Mode Hybrid Edge System

### 3.1 Design Decisions & Rationale
1. **Dual-Mode Hybrid Firmware (Live AD8232 + Backward-Compatible USB Replay):**
   * **Mode 1 — Live AD8232 Biometric Mode:** Works completely standalone on a USB power bank (monitored and controlled from any smartphone connected to Wi-Fi SSID `pico` → `http://192.168.4.1`) **or** connected to a battery-powered laptop running a real-time 60 FPS ECG oscilloscope at `http://localhost:8080`.
   * **Mode 2 — PhysioNet USB Replay Mode:** Preserves 100% compatibility with `pico_deploy.py stream` and `run_hardware_test.py` ($\Delta\text{sim} = 0.0000$ across all 89 PhysioNet subjects), serving as both the scientific parity benchmark and a zero-risk demo fallback.
2. **Standby + Manual Trigger Flow (Eliminating Electrode-Application Noise):**
   * Applying wet electrodes to skin creates a 2–3 second DC settling transient and arm-movement spikes.
   * Instead of auto-firing `REJECT` decisions the instant skin is touched, the Pico defaults to **`STANDBY / PREVIEW`**: it monitors `LO+`/`LO-`, settles for 3 seconds, displays `SIGNAL READY · <BPM>`, and streams the live waveform so the operator can verify signal quality *before* triggering:
     * **`[ ENROLL 5B ]`:** Captures 5 SQI-verified beats (~4–5 s), computes the L2-normalized 48-D centroid (`192 bytes`), and appends it to the in-RAM live database (up to 8 live users alongside the 89 PhysioNet templates, zero retraining).
     * **`[ AUTH 5B ]`:** Captures 5 SQI-verified beats, runs Core 1 inference on each, computes the 5-beat majority verdict (`>= 3/5 ACCEPT`), blinks the LED (`2×` solid ACCEPT / `5×` rapid REJECT), and returns cleanly to Standby.
     * **`[ CONTINUOUS ]`:** Optional toggle demonstrating continuous passive beat-by-beat authentication with instant `LEADS OFF` lockout upon electrode removal.
3. **5-Point Automatic Signal Quality Index (SQI) Gate (`ecg_dsp.h`):**
   * Every live beat must pass all 5 checks before ever reaching `ECG_ResNet_SE`:
     1. **3.0 s Settling Hold-Off** (`1,500` samples after any `LEADS OFF → LEADS ON` edge).
     2. **Zero Leads-Off Inside Window** (`LO+ == 0 && LO- == 0` for all 300 samples).
     3. **Anti-Clipping Guard** (raw 12-bit ADC strictly inside `[50, 4045]`).
     4. **Physiological RR-Interval Sanity** (`200..667` samples = **45–150 BPM**).
     5. **QRS Variance & Crest-Factor Gate** (`std >= 0.004`, `|R - mean| / std >= 2.20`, rejecting flatlines, pure 50 Hz sine hum, and slow baseline swells).
4. **50 Hz Mains Notch Filter (India Power Grid):**
   * Real-time Direct Form II Transposed float32 biquad cascade at $f_s = 500\text{ Hz}$ (`scipy.signal` exact bilinear coefficients):
     * Stage 1: 2nd-order Butterworth HPF @ `0.5 Hz` (`b = [0.995567, -1.991134, 0.995567]`, `a = [1, -1.991114, 0.991154]`)
     * Stage 2: 2nd-order Butterworth LPF @ `40.0 Hz` (`b = [0.046132, 0.092264, 0.046132]`, `a = [1, -1.307285, 0.491812]`)
     * Stage 3: 2nd-order IIR Notch @ `50.0 Hz`, $Q = 15$ (`b = [0.979483, -1.584836, 0.979483]`, `a = [1, -1.584836, 0.958966]`)
5. **Controls & Breadboard Pinout:**
   * Strictly software UI controls (phone + laptop + USB serial) and Pico onboard LED (no external LEDs or push-buttons required).
   * Wiring: `AD8232 3.3V → Pin 36 (3V3 OUT)`, `GND → Pin 38 (GND)`, `OUTPUT → Pin 31 (GP26 / ADC0)`, `LO+ → Pin 21 (GP16, internal pull-up)`, `LO- → Pin 22 (GP17, internal pull-up)`.
   * **Electrical Safety:** Live human electrodes are only ever attached when powered by a **USB power bank** or a **laptop unplugged from AC mains (battery only)**.

---

## 4. Implemented Files & Firmware Architecture

| File | Change | Role |
|---|---|---|
| [`hardware/firmware/ecg_dsp.h`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/firmware/ecg_dsp.h) | **[NEW]** | Portable 500 Hz causal biquad cascade (0.5–40 Hz + 50 Hz notch), adaptive QRS R-peak extractor, 300-sample z-score window slicer, BPM estimator, and 5-point SQI gate. |
| [`hardware/firmware/host_dsp_test.c`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/firmware/host_dsp_test.c) | **[NEW]** | Host x86 C test suite verifying `ecg_dsp.h` + `nn_inference.h` on continuous 500 Hz PhysioNet recordings (with injected 50 Hz hum & 0.2 Hz wander) plus 3 SQI artifact stress tests. |
| [`hardware/firmware/CMakeLists.txt`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/firmware/CMakeLists.txt) | **[MODIFIED]** | Linked `hardware_adc`. |
| [`hardware/firmware/main.c`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/firmware/main.c) | **[MODIFIED]** | Added 500 Hz timer ISR (`512`-sample lock-free ring buffer = 1.024 s slack), non-blocking `getchar_timeout_us(0)` serial command + CSV parser, Standby/Enroll/Auth/Continuous state machine, Core 1 8-user live enrollment & 5-beat voting, and 100 Hz toggleable USB telemetry (`CMD:TELEM ON/OFF`). |
| [`hardware/firmware/wifi_server.h`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/firmware/wifi_server.h) / [`wifi_server.c`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/firmware/wifi_server.c) | **[MODIFIED]** | Added live sensor/BPM/mode/vote telemetry and static-memory HTTP `303` action routes (`/enroll`, `/auth`, `/cont`, `/standby`, `/clear`) with zero dynamic allocation and 0-byte large stack buffers. |
| [`hardware/scripts/live_ecg_monitor.py`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/scripts/live_ecg_monitor.py) | **[NEW]** | Zero-dependency (`pyserial` + Python stdlib) dark-terminal 60 FPS HTML5 `<canvas>` live ECG oscilloscope & control panel at `http://localhost:8080` (includes both Live AD8232 controls and Stage-1 PhysioNet Replay / 5-beat enroll dry-run buttons). |
| [`hardware/scripts/pico_deploy.py`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/scripts/pico_deploy.py) | **[MODIFIED]** | Added `monitor` subcommand (`uv run python hardware/scripts/pico_deploy.py monitor`). |

---

## 5. Stage 1 Automated Verification Results (Pre-Electrode)

All automated host, DSP, firmware cross-compilation, and Python pipeline tests passed on the first build:

### 5.1 Host C Inference Parity (`hardware/build/host_test`)
* Compiled with `gcc -O2 -Wall -Wextra` — **0 warnings**.
* 5/5 embedded test beats matched PyTorch reference bit-for-bit:
  * `Person_01`: `0.8346` (`ACCEPT`), `Person_02`: `0.7368` (`ACCEPT`), `Person_03`: `0.9247` (`ACCEPT`), `Person_88`: `0.8062` (`ACCEPT`), `Person_87`: `0.1279` (`REJECT`).

### 5.2 Host C Continuous Streaming DSP + 5-Point SQI + Inference (`hardware/build/host_dsp_test`)
* Fed continuous raw 500 Hz PhysioNet recordings corrupted with **+0.15 mV 50 Hz mains hum** and **+0.25 mV 0.2 Hz respiratory baseline wander**, preceded by a `LEADS OFF → SETTLING (3s) → READY` transition:
  * **`Person_01/rec_2.dat`:** **18 / 18 exact Rank-1 ACCEPT**, mean cosine similarity = **`0.8011`**, max = **`0.8630`**, `sqi_rejects = 0`.
  * **`Person_03/rec_2.dat`:** **20 / 20 exact Rank-1 ACCEPT**, mean cosine similarity = **`0.7902`**, max = **`0.8804`**, `sqi_rejects = 0`.
* Artifact Rejection Stress Tests (8 seconds each):
  * Pure 50 Hz mains hum → **0 emitted beats** (PASS)
  * Flatline (`ADC = 2048`) → **0 emitted beats** (PASS)
  * ADC rail clipping spikes (`0 / 4095`) → **0 emitted beats** (PASS)

### 5.3 Pico WH Cross-Compilation & Memory Footprint (`ninja -C hardware/build_pico`)
* **Build:** `197/197` targets compiled & linked cleanly (`Release`, `PICO_BOARD=pico_w`).
* **Flash (`.bin` / `.uf2`):** `435,244` bytes raw text (`422 KB / 2,048 KB` = **20.6% Flash**); `.uf2` payload = **843 KB** (+19 KB over v1's 824 KB).
* **SRAM (`.bss`):** `162,428` bytes (**158.6 KB / 264 KB** = **60.1% SRAM**, leaving **>105 KB free** for stacks and heap).

### 5.4 Python Pipeline & Syntax Checks
* `uv run python -m py_compile hardware/scripts/live_ecg_monitor.py hardware/scripts/pico_deploy.py` → **PASS**.
* `make test` (`unittest discover -s tests -v`) → **8 / 8 tests PASSED** (0.112 s).

---

## 6. Pending Hardware Bench Validation Logs

### 6.1 Stage 1B — Pico WH Flash, Replay Parity & Live-Command Verification *(Electrodes Unplugged)* — **PASSED**
* [x] **Flashed `pico_ecg_biometrics.uf2` (843 KB):** BOOTSEL enumerated on `/dev/sdc1` (updated `pico_deploy.py` `find_pico_mount()` to dynamically discover `RPI-RP2` via `lsblk` instead of hardcoding `/dev/sda1`). Re-enumerated cleanly at `/dev/ttyACM0`.
* [x] **Backward-Compatible USB Replay Parity (with 500 Hz ADC ISR active):**
  * `Person_01`: `HW Sim = 0.8346`, `Ref Sim = 0.8346`, **`Δ = 0.0000`**, **`300.21 ms`**, **`ACCEPT`**
  * `Impostor`: `HW Sim = 0.3609`, `Ref Sim = 0.3609`, **`Δ = 0.0000`**, **`300.10 ms`**, **`REJECT`**
  * 500 Hz hardware timer ISR overhead on Core 1 neural inference: **+0.16 ms (~0.05%)**.
* [x] **AD8232 Wiring & Telemetry Verification (cable unplugged):**
  * `CMD:STATUS` → `mode=STANDBY, sensor=LEADS OFF, bpm=0.0, live=0, sqi_rej=0`.
  * `CMD:TELEM ON` → **46 lines / 0.45 s** (exact **100 Hz** stream); raw ADC0 counts `1986–2121` centered cleanly at **~2048 (1.65 V AD8232 mid-rail)**, proving 3.3V power, GND, `OUTPUT → GP26`, and `LO+/LO- → GP16/GP17` are wired accurately.
* [x] **On-Device Zero-Shot 5-Beat Enrollment Verification:**
  * Enrolled 5 beats of `Person_87` (historically misclassified as `Person_89` due to single-session dataset drift) as `Live_NewUser` (`300.2–300.5 ms`/beat, `EVENT:ENROLLED,Live_NewUser,1`).
  * Held-out beat #5 immediately matched **`Live_NewUser -> ACCEPT (sim = 0.8943, 300.60 ms)`**.
  * `CMD:CLEAR` cleanly cleared SRAM live slots (`live=0`, 89 PhysioNet templates intact).
* [ ] **Interactive UI Dry Run (User Browser & Phone):**
  * Run `uv run python hardware/scripts/pico_deploy.py monitor` → open `http://localhost:8080` and connect smartphone to `pico` (`picosama`) → `http://192.168.4.1`, verify all buttons and live status updates before Stage 2.

## 7. Real AD8232 Archive Validation (`MLEE-live-recordings-2026-11-16.zip`) & Approach A Fallback System

### 7.1 Archive Inspection & Hardware Signal Findings
An existing archive of 500 Hz Pico WH + AD8232 live bench recordings (`data/MLEE-live-recordings-2026-11-16.zip`, 2 subjects `P1` and `P2`, Lead I wrist placement, format `ECG1,seq,us,adc,leads_off,dropped,12`, 0 dropped samples) was inspected and validated against our Option 3 pipeline:

1. **50 Hz Mains Hum Validation:** `P1-s1-enroll-174755.json` measured `mains_50hz_to_ecg_band_power = 0.7268` (**72.7% of ECG band power** even on battery!). Adding our 50 Hz IIR notch ($Q = 15$) directly suppressed this interference and lifted usable clean beat yield on noisy `P2_enroll` from 8 beats to **19 SQI-verified beats** (`P1_enroll`: 74 clean beats, `P1_r1`: 26, `P1_r2`: 27, `P1_s2_retake1`: 46, `P2_test1`: 27).
2. **AD8232 Physical Rail Bounds:** Measured AD8232 output rails were ADC `~11` and `~4064` (not `0`/`4095`). Our `[50, 4045]` anti-clipping SQI guard in `ecg_dsp.h` already rejects railed segments with comfortable margin.
3. **Morphological Fidelity:** Real live AD8232 beats achieved mean R-peak crest factor `4.15` (clearing `ECG_MIN_CREST = 2.20`) and `0.891` correlation with PhysioNet ECG-ID Lead I.

### 7.2 Physical Pico WH Silicon Results on Real AD8232 Data (`P1` vs. `P2`)
Five beats from `P1-s1-enroll-174755.frames` were zero-shot enrolled directly on the physical RP2040 (`CMD:ENROLL P1_AD8232`, `300.0 ms`/beat, 192 B SRAM, zero retraining), and 25 held-out AD8232 beats across 5 separate recordings were evaluated on Pico WH Core 1:

| Real AD8232 Capture | Role | Pico WH Core-1 Matches (5 Beats) | Cosine Similarity Range | Mean Sim | Latency |
|---|---|---|---|---:|---:|
| **`P1-config-174605-r1`** | Genuine (same placement) | **5 / 5 `ACCEPT [P1_AD8232]`** | `0.6518 – 0.9162` | **0.7591** | 300.3 ms |
| **`P1-config-174605-r2`** | Genuine (same placement) | **5 / 5 `ACCEPT [P1_AD8232]`** | `0.8676 – 0.9709` | **0.9217** | 300.3 ms |
| **`P1-s2-test1-retake`** | Genuine (peel & restick, HR 95→106) | 1 / 5 `P1_AD8232` (`0.6109`); 4 / 5 matched random PhysioNet strangers (`0.54–0.70`) | `0.5403 – 0.6994` | 0.6326 | 300.4 ms |
| **`P2-s1-enroll`** | Unknown Human Impostor | 1 `REJECT` (`0.3956`), 3 matched random PhysioNet strangers (`0.56–0.62`), 1 matched `P1_AD8232` (`0.6307`) | `0.3956 – 0.6307` | 0.5527 | 300.5 ms |
| **`P2-s1-test1`** | Unknown Human Impostor (reused pads) | 2 matched PhysioNet strangers (`0.64–0.69`), 3 matched `P1_AD8232` (`0.66–0.75`) | `0.6413 – 0.7521` | 0.6834 | 300.3 ms |

> **Key Engineering Finding (Open-Set Multiple-Comparisons Effect):**
> Across same-placement real AD8232 captures (`P1_r1` + `P1_r2`), the frozen model achieved **10 / 10 (100%) exact recognition** of `P1_AD8232` with cosine similarities up to **`0.9709`**. On noisy unknown probes (`P2`) and re-placed electrodes (`P1_s2`), searching `max()` across **89 unrelated PhysioNet templates + 1 live user** at `0.50` introduces spurious false matches to random PhysioNet identities (`Person_77`, `Person_72`). For live-only deployments, restricting matching to the live-enrolled gallery (or raising the threshold slightly alongside fresh gel pads + alcohol skin prep) mitigates this multiple-comparisons effect.

---

### 7.3 Approach A: 1-Click Live Recorder & Pre-Recorded AD8232 Fallback (`live_ecg_monitor.py`)
To give the presenter complete control and a rock-solid demo fallback **without straining or modifying the Pico WH firmware**, **Approach A** was implemented directly in [`hardware/scripts/live_ecg_monitor.py`](file:///home/orion/projects/mlee/mlee-ecg-biometrics/hardware/scripts/live_ecg_monitor.py):

1. **1-Click Live Recorder (`[ RECORD 20s ]` / `[ STOP & SAVE ]`):**
   * Records 20 seconds (`2,000` samples @ 100 Hz) of the Pico's already-bandlimited (`< 40 Hz` LPF + 50 Hz notch) filtered telemetry stream.
   * **Nyquist-Preserving 5× Resampling:** Because the filtered ECG is strictly bandlimited below `40 Hz` (Nyquist = `80 Hz < 100 Hz`), `scipy.signal.resample_poly(wave_100, 5, 1)` reconstructs the 500 Hz waveform with **`0.99982` mean beat correlation** against native 500 Hz extraction (verified on `P1-config-174605-r2.frames`, 27/27 beats).
   * Saves `wave_100hz`, `beats_300`, and `rpeaks_100` cleanly to gitignored `private/recordings/<label>_<HHMMSS>.npz`.
2. **1-Click Pre-Recorded AD8232 Playback (`[ PLAYBACK: ENROLL 5B ]` / `[ PLAYBACK: AUTH 5B ]`):**
   * Automatically populates a dropdown with all clean sessions from `data/MLEE-live-recordings-2026-11-16.zip` **and** any user-recorded files in `private/recordings/`.
   * Animates the real AD8232 waveform smoothly on the `<canvas>` oscilloscope at 100 Hz while streaming 5 SQI-verified 300-sample beats at realistic heartbeat timing (~1 s/beat) to **Pico WH Core 1** — so every embedding, enrollment centroid, cosine match, LED blink, and phone dashboard (`192.168.4.1`) update runs on real RP2040 silicon.

### 7.4 Option 3+ Workflow Upgrades: Selective Unenroll, Gallery Scope, Threshold Presets & Unified 5-Beat Voting
To address the open-set multiple-comparisons finding from §7.2 and provide full biometric lifecycle management (`Enroll → Authenticate → Revoke`) directly on the Pico WH, four zero-strain upgrades were implemented across `main.c`, `wifi_server.c/h`, and `live_ecg_monitor.py`:

1. **Selective Per-User Unenroll & Live Gallery Inspector:**
   * Core 1 (`CORE1_OP_UNENROLL_ONE`) revokes any single slot (`0..7`) and compacts the SRAM table cleanly (`CMD:LIST`, `CMD:UNENROLL <slot_or_name>`, phone route `GET /unenroll?id=<k>`).
   * Laptop monitor displays an **Enrolled Live Users (Pico SRAM)** panel with per-user **`[ UNENROLL ]`** buttons, plus **`[ DELETE SAVED ]`** for user-recorded `.npz` takes (read-only archive `.zip` protected).
2. **Gallery Scope Toggle (`LIVE ONLY` vs. `ALL 89+N`):**
   * `CMD:SCOPE LIVE` / `CMD:SCOPE ALL` and phone route `GET /scope` toggle whether open-set matching searches strictly the enrolled live users (`s_live_users[]`) or all 89 PhysioNet templates + live users.
3. **Runtime Security Threshold Presets (`0.50` / `0.60` / `0.68`):**
   * `CMD:THRESH <val>` and phone route `GET /thresh` switch live between Standard (`0.50`), Strict (`0.60`), and Bank-Grade `0.1% FAR` (`0.68`) operating points.
4. **Unified On-Chip 5-Beat Voting Across Tier 1 & Tier 2:**
   * `run_and_report_replay()` now participates in `CAP_AUTH_5`, so Tier 2 `[ PLAYBACK: AUTH 5B ]` arms `CMD:AUTH` on the Pico WH and produces the exact same on-chip `EVENT:VOTE` verdict, LED pattern, and phone dashboard banner as Tier 1 live electrodes.

**Compiled Footprint (Option 3+):**
* **Flash:** `439,068` bytes raw text (**428.8 KB / 2,048 KB = 20.9%**; `.uf2` = **850 KB**, +7 KB).
* **SRAM:** `165,480` bytes (**161.6 KB / 264 KB = 61.2%**, **>102 KB free**).
* **Latency:** `300.2–300.5 ms` (0.00 ms regression).

**Physical Pico WH Verification (Option 3+):**

| Test Step | Configuration | Physical RP2040 Result |
|---|---|---|
| Multi-user enroll (`P1_AD8232` + `User_Two`) | 5 beats each (192 B SRAM each) | `EVENT:USERS,0:P1_AD8232,1:User_Two` |
| Genuine AD8232 `P1_r2` (5-beat unified `CMD:AUTH`) | `LIVE ONLY` @ `0.68` | **`EVENT:VOTE,P1_AD8232,ACCEPT,5/5,0.9217`** (`0.8676–0.9709`, `300.2 ms`) |
| Human Impostor AD8232 `P2_enroll` (5-beat `CMD:AUTH`) | `LIVE ONLY` @ `0.68` | **`EVENT:VOTE,IMPOSTOR,REJECT,0/5,0.3047`** (`0.0098–0.6307`, eliminated 4/5 spurious PhysioNet matches!) |
| Selective revocation (`CMD:UNENROLL P1_AD8232`) | `User_Two` kept intact | `P1_r2` → **`REJECT (-0.4285)`**; `User_Two` → **`ACCEPT (0.9714)`** |

---

## 8. Three-Tier Live Demo Runbook

Launch the monitor once:
```bash
uv run python hardware/scripts/pico_deploy.py monitor
# Open http://localhost:8080 (laptop) + http://192.168.4.1 (phone on Wi-Fi 'pico' / 'picosama')
```

| Tier | When to Use | Exact Demo Steps |
|---|---|---|
| **Tier 1: Live AD8232 Electrodes** | Primary demo (battery-only laptop or USB power bank, fresh gel pads) | Apply R/L/COM pads → wait for `SIGNAL READY · BPM` → `[ AUTH 5 BEATS ]` (shows `REJECT` before enrollment) → `[ ENROLL 5 BEATS ]` → `[ AUTH 5 BEATS ]` (`ACCEPT`) → swap to friend (`REJECT`) |
| **Tier 2: Pre-Recorded AD8232 Fallback** | Tight time slot, noisy room, or no volunteer for electrodes | Select `[Archive] P1 Enroll (60s, AD8232)` → `[ PLAYBACK: ENROLL 5B ]` → select `[Archive] P1 Test r2 (20s)` → `[ PLAYBACK: AUTH 5B ]` (5/5 `ACCEPT`, sim ~`0.92`) → `[ CLEAR LIVE USERS ]` |
| **Tier 3: PhysioNet 89-Subject Benchmark** | Scientific parity defense (`Δsim = 0.0000`) | Click `[ REPLAY Person_01 ]` (`0.8346 ACCEPT`), `[ REPLAY Person_03 ]` (`0.9247 ACCEPT`), and `[ REPLAY IMPOSTOR ]` (`0.3609 REJECT`) |

---

## 9. Pending Live Electrode Check (Stage 2)
* [ ] Unplug laptop AC charger (or move Pico WH to USB power bank), attach 3-lead cable & fresh electrodes.
* [ ] Verify `LEADS OFF → SETTLING (3s) → SIGNAL READY · <BPM>` and clean P-QRS-T waveform on `localhost:8080`.
* [ ] Click `[ RECORD 20s ]` to archive clean live backup sessions of yourself (and a friend) into `private/recordings/`.
* [ ] Run live `[ ENROLL 5 BEATS ]` → `[ AUTH 5 BEATS ]` → `[ CONTINUOUS ]` → standalone power-bank phone test (`192.168.4.1`).
