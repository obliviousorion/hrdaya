#!/usr/bin/env python3
"""
run_hardware_test.py — Unified hardware verification script.

Streams held-out ECG test beats to the Pico WH over USB serial,
parses responses, computes PyTorch reference similarities, and writes
a hardware test report to hardware/results/hardware_test_report.md.

Usage:
    python hardware/scripts/run_hardware_test.py --port /dev/ttyACM0 --beats 20
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def pytorch_reference_sims(X_raw, y, classes):
    """Compute cosine similarities using the PyTorch champion model (reference)."""
    try:
        import torch
        from mlee.resnet_se_model import ECG_ResNet_SE
    except ImportError:
        return None, None

    private = ROOT / "private"
    fp32_path = private / "resnet_se_hybrid_fp32.pt"
    if not fp32_path.exists():
        return None, None

    n_classes = len(classes)
    model = ECG_ResNet_SE(n_classes=n_classes, emb_dim=48)
    model.load_state_dict(
        torch.load(fp32_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        tr = np.load(private / "features_train.npz", allow_pickle=True)
        emb_train = model.embed(
            torch.tensor(tr["X_raw"], dtype=torch.float32)
        ).numpy()
        templates = np.stack([
            emb_train[tr["y"] == i].mean(axis=0) for i in range(n_classes)
        ])
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        templates = templates / np.maximum(norms, 1e-12)

        X_t = torch.tensor(X_raw, dtype=torch.float32)
        embs = model.embed(X_t).numpy()
        embs_n = embs / np.maximum(
            np.linalg.norm(embs, axis=1, keepdims=True), 1e-12
        )

    sims = []
    for i, (emb, true_cls) in enumerate(zip(embs_n, y)):
        s = float(np.dot(emb, templates[true_cls]))
        sims.append(s)

    return sims, templates


def parse_pico_line(line: str):
    """
    Parse a line like:
      [3] STREAM        -> ACCEPT  sim=0.8346  t=41230 us (41.23 ms)
    Returns (sim, latency_us, decision) or None on parse failure.
    """
    try:
        parts = line.strip()
        sim_part = [p for p in parts.split() if p.startswith("sim=")]
        lat_part = [p for p in parts.split() if p.startswith("t=")]
        if not sim_part or not lat_part:
            return None
        sim = float(sim_part[0].split("=")[1])
        lat_us = int(lat_part[0].split("=")[1].replace("us", ""))
        decision = "ACCEPT" if "ACCEPT" in parts else "REJECT"
        return sim, lat_us, decision
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Pico WH Hardware Verification Test")
    parser.add_argument(
        "--port", default="/dev/ttyACM0",
        help="USB serial port (default: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--beats", type=int, default=20,
        help="Number of test beats to stream (default: 20)"
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Serial read timeout per beat in seconds (default: 5.0)"
    )
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed.")
        print("  Run: uv add pyserial  (or pip install pyserial)")
        sys.exit(1)

    private = ROOT / "private"
    te_path = private / "features_test.npz"
    if not te_path.exists():
        print(f"ERROR: {te_path} not found. Run `make data` first.")
        sys.exit(1)

    te = np.load(te_path, allow_pickle=True)
    X_test = te["X_raw"]
    y_test  = te["y"]
    classes = te["classes"].tolist()
    n_beats = min(args.beats, len(y_test))

    print("Computing PyTorch reference similarities...")
    ref_sims, _ = pytorch_reference_sims(
        X_test[:n_beats], y_test[:n_beats], classes
    )
    has_ref = ref_sims is not None
    if not has_ref:
        print("  (PyTorch model not available — skipping reference comparison)")

    print(f"Connecting to Pico WH on {args.port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
        time.sleep(2.0)  # Wait for Pico DTR reset
    except Exception as e:
        print(f"ERROR: Cannot open {args.port}: {e}")
        print("  Is the Pico flashed? Check: ls -la /dev/ttyACM*")
        sys.exit(1)

    # Drain boot messages
    ser.reset_input_buffer()
    time.sleep(0.5)

    # Flush remaining boot output (self-test)
    print("Waiting for Pico self-test to complete (~5 s)...")
    deadline = time.time() + 8.0
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  [boot] {line}")
        if "Listening for live beats" in line:
            break

    print(f"\nStreaming {n_beats} held-out test beats to Pico...\n")
    header = f"{'#':>3}  {'True Identity':<14}  {'HW Sim':>8}  {'Ref Sim':>8}  {'Delta':>7}  {'Latency ms':>10}  {'Decision':<8}  {'Match?'}"
    print(header)
    print("-" * len(header))

    rows = []
    n_pass = 0
    n_match_ref = 0
    latencies = []

    for i in range(n_beats):
        beat      = X_test[i]
        true_cls  = int(y_test[i])
        true_name = classes[true_cls]
        ref_sim   = ref_sims[i] if has_ref else float("nan")

        # Stream beat to Pico (300 comma-separated floats + newline)
        payload = ",".join(f"{v:.6f}" for v in beat) + "\n"
        ser.write(payload.encode("utf-8"))

        # Read Pico response
        resp_line = ""
        for _ in range(3):
            resp_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if resp_line:
                break

        parsed = parse_pico_line(resp_line)
        if parsed is None:
            print(f"{i+1:>3}  {true_name:<14}  [NO RESPONSE: {resp_line!r}]")
            rows.append((true_name, None, ref_sim, None, None, None))
            continue

        hw_sim, lat_us, decision = parsed
        lat_ms  = lat_us / 1000.0
        delta   = abs(hw_sim - ref_sim) if has_ref else float("nan")
        passed  = decision == "ACCEPT"
        matches = delta < 0.005 if has_ref else True

        status = "OK" if (passed and matches) else ("SIM-DRIFT" if not matches else "REJECT")
        if passed:
            n_pass += 1
        if matches:
            n_match_ref += 1
        latencies.append(lat_ms)
        rows.append((true_name, hw_sim, ref_sim, delta, lat_ms, decision))

        ref_str = f"{ref_sim:8.4f}" if has_ref else "    N/A "
        dlt_str = f"{delta:7.4f}"   if has_ref else "    N/A"
        print(f"{i+1:>3}  {true_name:<14}  {hw_sim:8.4f}  {ref_str}  {dlt_str}  {lat_ms:10.2f}  {decision:<8}  {status}")

        time.sleep(0.15)

    ser.close()

    # ---- Summary -----------------------------------------------------------
    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        max_lat  = max(latencies)
    else:
        mean_lat = max_lat = 0.0

    print("\n" + "=" * 70)
    print(f"  HARDWARE TEST SUMMARY  ({n_beats} beats)")
    print("=" * 70)
    print(f"  ACCEPT decisions  : {n_pass}/{n_beats}")
    print(f"  Sim within 0.005  : {n_match_ref}/{n_beats}" if has_ref else "  Reference N/A")
    print(f"  Mean latency      : {mean_lat:.2f} ms")
    print(f"  Max latency       : {max_lat:.2f} ms")
    overall = "PASS" if (mean_lat <= 60.0) else "FAIL (latency)"
    print(f"  Latency budget    : {overall}  (target <= 60 ms)")
    print("=" * 70)

    # ---- Write report -------------------------------------------------------
    results_dir = ROOT / "hardware" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "hardware_test_report.md"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(report_path, "w") as f:
        f.write("# Hardware Test Report — Raspberry Pi Pico WH\n\n")
        f.write(f"**Timestamp:** {ts}  \n")
        f.write(f"**Port:** {args.port}  \n")
        f.write(f"**Beats tested:** {n_beats}  \n")
        f.write("**Model:** `ECG_ResNet_SE` (14,867 params, BN-fused float32)  \n\n")
        f.write("---\n\n")
        f.write("## Per-Beat Results\n\n")
        f.write("| # | Identity | HW Sim | Ref Sim | Δsim | Latency (ms) | Decision |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for idx, row in enumerate(rows):
            name, hw, ref, delta, lat, dec = row
            hw_s  = f"{hw:.4f}"   if hw    is not None else "N/A"
            ref_s = f"{ref:.4f}"  if has_ref and not np.isnan(ref) else "N/A"
            d_s   = f"{delta:.4f}" if delta is not None and not np.isnan(delta) else "N/A"
            l_s   = f"{lat:.2f}"  if lat   is not None else "N/A"
            dec_s = dec if dec else "N/A"
            f.write(f"| {idx+1} | `{name}` | {hw_s} | {ref_s} | {d_s} | {l_s} | {dec_s} |\n")
        f.write("\n---\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | Value | Target | Status |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Mean latency | {mean_lat:.2f} ms | ≤ 60 ms | {'PASS' if mean_lat <= 60 else 'FAIL'} |\n")
        f.write(f"| Max latency  | {max_lat:.2f} ms  | ≤ 60 ms | {'PASS' if max_lat <= 60 else 'FAIL'} |\n")
        if has_ref:
            f.write(f"| Sim accuracy (Δ < 0.005) | {n_match_ref}/{n_beats} | {n_beats}/{n_beats} | {'PASS' if n_match_ref == n_beats else 'PARTIAL'} |\n")
        f.write(f"| ACCEPT rate | {n_pass}/{n_beats} | — | — |\n")

    print(f"\nReport saved to: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
