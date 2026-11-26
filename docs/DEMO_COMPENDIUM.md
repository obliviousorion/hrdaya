# Biometric Hardware & Presentation Compendium
**Course Project:** EEE G513 Machine Learning for Electrical Engineers  
**System:** Raspberry Pi Pico WH RP2040 Dual-Core AMP @ 250 MHz  
**Location:** `private/DEMO_COMPENDIUM.md` (Confidential / Quick Reference)

---

## 1. Quick-Glance Presentation Cheat Sheet

| Parameter | Value | Notes |
|---|---|---|
| **Serial Port** | `/dev/ttyACM0` @ `115200` baud | USB CDC interface |
| **Wi-Fi SSID** | `pico` | WPA2 autonomous Access Point |
| **Wi-Fi Password** | `picosama` | Auto-DHCP gateway: `192.168.4.1` |
| **Web Dashboard** | `http://192.168.4.1` | Terminal-dark theme, 2s auto-refresh |
| **Physical Board** | Raspberry Pi Pico WH | RP2040 Dual Cortex-M0+ @ 250 MHz (1.20V) |
| **Core 0 Role** | USB CDC, Wi-Fi AP, HTTP server, LED signaling | Non-blocking networking (< 1 ms ping) |
| **Core 1 Role** | Dedicated 1D ResNet-SE neural inference | Inter-core 32-bit hardware SIO FIFO |
| **Measured Latency** | **300.05 ms** | Software float32 emulation (zero FPU) |
| **Mathematical Parity** | **$\Delta\text{sim} = 0.0000$** | 100% exact match with host PyTorch |
| **Biometric Threshold**| $\cos \theta \ge 0.50$ | ACCEPT $\ge 0.50$, REJECT $< 0.50$ |

---

## 2. Live Demo Command Sequence (Step-by-Step)

### Step 0: Pre-Flight Check & Phone Connection
1. Plug the Pico WH into the laptop USB port. Verify device:
   ```bash
   ls -l /dev/ttyACM0
   ```
2. Connect your smartphone to Wi-Fi SSID **`pico`** (Password: **`picosama`**).
   *(Tip: On Android/iOS, if prompted "Wi-Fi has no internet access", select "Stay connected".)*
3. Open mobile browser to **`http://192.168.4.1`**.
4. Confirm the web page renders the rolling 5-beat table and system banner.

---

### Step 1: Authentic User Authentication Demo (The "Unlock")
Demonstrates successful identification of an enrolled individual.

```bash
uv run python hardware/scripts/pico_deploy.py stream --subject Person_01
```

* **What Happens:**
  * **Terminal Output:** Shows `Person_01`, `HW Sim = 0.8346`, `Ref Sim = 0.8346`, `Delta = 0.0000`, `Latency = 300.05 ms`, `Decision = ACCEPT`.
  * **Physical Actuator (LED):** Pico onboard LED emits **2 solid pulses** (120 ms ON / 100 ms OFF) representing door unlock / access granted.
  * **Phone Dashboard:** Automatically updates to display a green row: `Person_01`, `ACCEPT`, `0.8346`, `300.05 ms`.

---

### Step 2: Impostor Intrusion & Rejection Demo (The "Alarm")
Demonstrates active defense against an out-of-distribution / unauthorized waveform.

```bash
uv run python hardware/scripts/pico_deploy.py stream --impostor
```

* **What Happens:**
  * **Terminal Output:** Shows `Impostor`, `HW Sim = 0.3609`, `Ref Sim = 0.3609`, `Delta = 0.0000`, `Latency = 299.98 ms`, `Decision = REJECT`.
  * **Physical Actuator (LED):** Pico onboard LED emits **5 rapid alert pulses** (60 ms ON / 60 ms OFF) signaling security lockout.
  * **Phone Dashboard:** Automatically logs a red row: `Impostor`, `REJECT`, `0.3609`, `299.98 ms`.

---

### Step 3: Multi-Beat Stream & Concurrency Stress Test
Demonstrates dual-core independence: Core 1 computes neural activations while Core 0 handles web traffic simultaneously.

```bash
uv run python hardware/scripts/pico_deploy.py stream --beats 5
```

* **Live Action:** While the terminal is streaming the 5 beats, **repeatedly refresh the webpage on your smartphone**.
* **Talking Point:** *"Notice there is zero lag or dropped packets on the phone. Core 0 continuously services lwIP HTTP polling in a non-blocking loop, while Core 1 executes the 300 ms neural network entirely in parallel."*

---

### Step 4: Arbitrary Subject Evaluation (Audience Choice)
Allow the professor or evaluator to pick any subject from 1 to 90:

```bash
uv run python hardware/scripts/pico_deploy.py stream --subject Person_05
uv run python hardware/scripts/pico_deploy.py stream --subject Person_12
uv run python hardware/scripts/pico_deploy.py stream --beat-idx 150
```

---

## 3. Host Machine Learning & Benchmark Workflows

If asked to train, benchmark, or verify models live on the laptop:

| Workflow Goal | Terminal Command | Duration / Details |
|---|---|---|
| **Full Reproducibility Pipeline** | `make setup && make data && make features && make champion` | Full end-to-end retraining from scratch |
| **Classical SVM Baseline** | `make baseline` | AC/DCT(32) + SVM (79.06% Rank-1, 13.11% EER) |
| **Train Champion Model** | `make champion` | Trains `ECG_ResNet_SE` + ArcFace for 45 epochs (seed: 42) |
| **Run 4-Part Robustness Suite** | `make robustness` | 201k impostor attacks, noise sweep, HRV warping, cross-month |
| **Multi-Beat Window Ablation** | `make ablation` | 1-beat (90.87%), 3-beats (94.38%), 5-beats (95.51%), 10-beats (96.63%) |
| **Offline Prototype Demo Replay** | `make demo` | Replays full test sessions across all 89 subjects (97.8% ACCEPT) |
| **Automated Test Suite** | `make test` | Runs unit/integration tests |
| **Code Style & Linting** | `make lint` | Ruff linter (0 errors) |
| **Re-Generate Figures** | `make figures` | Re-renders `pipeline.svg` and `confusion.svg` |
| **Rebuild Leaderboard** | `make leaderboard` | Updates `results/LEADERBOARD.md` |

---

## 4. Hardware Firmware Build & Flash Commands

If you modify firmware code (`hardware/firmware/main.c` or `wifi_server.c`) during the session:

```bash
# 1. One-command build, flash, and test:
uv run python hardware/scripts/pico_deploy.py all

# 2. Build only (Ninja):
uv run python hardware/scripts/pico_deploy.py build
# or: ninja -C hardware/build_pico

# 3. Flash only (via 1200-baud reset & mass-storage copy):
uv run python hardware/scripts/pico_deploy.py flash

# 4. Live serial monitor:
uv run python hardware/scripts/pico_deploy.py listen

# 5. Re-export weights to model_data.h after model retraining:
uv run python hardware/scripts/export_pico_c_arrays.py
```

---

## 5. Key Metrics Defense Card (Memorize for Q&A)

### Machine Learning & Mathematical Separation
* **Architecture:** `ECG_ResNet_SE` (Depthwise Separable 1D Conv + Squeeze-and-Excitation attention + LayerNorm).
* **Parameters:** **14,867 parameters** ($59.5\text{ KB}$ raw FP32 weights).
* **Loss Function:** ArcFace ($s = 16.0$, angular margin penalty $m = 0.50\text{ rad} \approx 28.6^\circ$).
* **Embedding Manifold:** 48-dimensional unit hypersphere $\mathbb{S}^{47}$ ($192\text{ bytes}$ per template vector).
* **Decidability Index ($d'$ / D-Prime):** **$\mathbf{4.1256}$** across **201,344 empirical impostor comparisons**. ($d' > 3.0$ is commercial/military grade).
* **Mean Genuine Similarity:** $\mu_{\text{gen}} = \mathbf{+0.8190}$ ($\sigma = 0.1745$).
* **Mean Impostor Similarity:** $\mu_{\text{imp}} = \mathbf{-0.0029}$ ($\sigma = 0.2212$) — **True mathematical orthogonality**.
* **False Acceptance Rate (FAR):** Only **1.56%** at deployed threshold ($0.50$); **88.07% TAR** at bank-grade $\text{FAR} = 0.1\%$ ($1 \text{ in } 1,000$).
* **Noise Tolerance:** Flat down to **20 dB SNR ($89.51\%$, only $-2.32\%$ drop)**; ready for dry consumer electrodes.
* **Longitudinal Invariance:** Cohort of 20 subjects recorded months apart achieved **95.63% Rank-1 accuracy** ($+4.87\%$ over same-day recordings), proving zero temporal degradation.
* **Multi-Beat Decision Window:** 1 beat = $90.87\%$; 5 beats = **$95.51\%$** ($\approx 4\text{ seconds}$).

### Embedded Hardware & Silicon Constraints
* **Silicon:** Raspberry Pi Pico WH (RP2040 Dual Cortex-M0+, no hardware FPU).
* **Clock Frequency:** Overclocked to **250 MHz** at **1.20V core VREG** (48 MHz USB PLL isolated).
* **Inference Latency:** **300.05 ms** (down from $584.05\text{ ms}$ at stock 133 MHz, a **48.6% speedup**).
* **Cardiac Cycle Fraction:** At 72 BPM ($833\text{ ms}$ cycle), inference takes **$36.0\%$ of the heartbeat**, leaving $64\%$ CPU idle headroom.
* **RAM Footprint:** **104 KB / 264 KB** ($39.4\%$ utilized, peak activation buffer $\sim 12\text{ KB}$).
* **Flash Footprint:** **824 KB / 2 MB** ($40.2\%$ utilized, includes lwIP TCP/IP stack and CYW43439 driver).
* **Quantization Loss:** **$\Delta\text{sim} = 0.0000$** (bit-for-bit mathematical parity between C kernel and PyTorch).

---

## 6. Zero-Shot Enrollment Explanation (Common Panel Question)

**Question:** *"How does your system enroll a new user? Do you have to retrain the neural network on a server?"*

**Answer:**
> *"No, our system uses **Deep Metric Learning**, which means you **never retrain the neural network to enroll new people**.*  
> *When a new user registers:*
> 1. *They touch an ECG sensor for 5 heartbeats (~4 seconds).*
> 2. *The pre-trained frozen feature extractor on Core 1 maps each beat to a 48-dimensional unit vector $\mathbf{z} \in \mathbb{S}^{47}$.*
> 3. *The 5 vectors are averaged into a centroid template ($48 \times 4 = 192\text{ bytes}$).*
> 4. *This 192-byte vector is appended to non-volatile NOR Flash memory.*
> 5. *From that moment on, authentication is a simple 48-multiply dot product $\mathbf{z} \cdot \mathbf{t}_{\text{new}} \ge 0.50$. Zero GPU retraining, zero backpropagation, 192 bytes of storage."*

---

## 7. Emergency Troubleshooting Playbook

| Issue Encountered | Root Cause | Instant Fix Command |
|---|---|---|
| **`/dev/ttyACM0` busy or permission denied** | Another process is reading serial or permissions reset | `sudo fuser -k /dev/ttyACM0` then `sudo chmod 666 /dev/ttyACM0` |
| **Phone connects to Wi-Fi but webpage won't open** | Mobile data interference or phone cached old IP | Disable "Mobile Data" temporarily on phone; browse explicitly to `http://192.168.4.1` |
| **`pico_deploy.py flash` says drive not mounted** | Linux udisks daemon delay after 1200-baud reset | `udisksctl mount -b /dev/sda1` (script retries 4 times automatically) |
| **Pico halts or LED doesn't blink** | Power transient or USB brownout | Unplug USB cable, hold BOOTSEL, replug, and run `uv run python hardware/scripts/pico_deploy.py flash` |
| **Re-running tests fails with Python environment error** | Missing dependencies in local shell | Run all commands prefixed with `uv run python ...` |
