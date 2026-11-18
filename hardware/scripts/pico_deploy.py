#!/usr/bin/env python3
"""
pico_deploy.py — All-in-One Automated Pico WH Build, Flash & Test Harness

Usage:
  uv run python hardware/scripts/pico_deploy.py build             # Recompile firmware with ninja
  uv run python hardware/scripts/pico_deploy.py flash             # Soft-reset, mount, and flash UF2
  uv run python hardware/scripts/pico_deploy.py all               # Build + Flash + Run verification
  uv run python hardware/scripts/pico_deploy.py stream --subject Person_01  # Stream 1 beat
  uv run python hardware/scripts/pico_deploy.py stream --impostor           # Test impostor rejection
  uv run python hardware/scripts/pico_deploy.py listen            # Monitor live serial logs
  uv run python hardware/scripts/pico_deploy.py monitor           # Launch live ECG oscilloscope UI
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "hardware" / "build_pico"
UF2_PATH = BUILD_DIR / "pico_ecg_biometrics.uf2"
SERIAL_PORT = "/dev/ttyACM0"


def log(msg: str):
    print(f"\033[1;34m[pico-deploy]\033[0m {msg}")


def log_ok(msg: str):
    print(f"\033[1;32m[pico-deploy] [OK]\033[0m {msg}")


def log_err(msg: str):
    print(f"\033[1;31m[pico-deploy] [ERROR]\033[0m {msg}")


def run_cmd(cmd, check=True):
    log(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if check and res.returncode != 0:
        log_err(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    return res.returncode


def cmd_build():
    """Build the Pico firmware using Ninja."""
    log("Building firmware with Ninja...")
    ret = run_cmd(f"ninja -C {BUILD_DIR}")
    if ret == 0 and UF2_PATH.exists():
        size_kb = UF2_PATH.stat().st_size / 1024.0
        log_ok(f"Build complete: {UF2_PATH.name} ({size_kb:.1f} KB)")
    else:
        log_err("Build failed or UF2 file not generated.")
        sys.exit(1)


def find_pico_mount():
    """Find the mount directory for the Pico BOOTSEL drive (RPI-RP2)."""
    candidates = [
        Path("/run/media") / os.getenv("USER", "orion") / "RPI-RP2",
        Path("/media") / os.getenv("USER", "orion") / "RPI-RP2",
        Path("/run/media/orion/RPI-RP2"),
    ]
    for c in candidates:
        if c.exists() and (c / "INFO_UF2.TXT").exists():
            return c

    # Dynamically discover RPI-RP2 partition via lsblk (handles sda1, sdb1, sdc1, etc.)
    for _ in range(4):
        res = subprocess.run(
            "lsblk -rnpo NAME,LABEL", shell=True, capture_output=True, text=True
        )
        dev = None
        for line in res.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip() == "RPI-RP2":
                dev = parts[0].strip()
                break
        if dev is None and Path("/dev/sda1").exists():
            dev = "/dev/sda1"
        if dev:
            subprocess.run(f"udisksctl mount -b {dev}", shell=True, capture_output=True)
            time.sleep(1.0)
            for c in candidates:
                if c.exists() and (c / "INFO_UF2.TXT").exists():
                    return c
        time.sleep(1.0)
    return None


def cmd_flash():
    """Soft-reset Pico, mount bootloader, and copy UF2."""
    if not UF2_PATH.exists():
        log_err(f"Binary not found: {UF2_PATH}. Run build first.")
        sys.exit(1)

    mount_dir = find_pico_mount()

    if mount_dir is None:
        # If not mounted, try triggering 1200-baud reset on CDC serial
        if Path(SERIAL_PORT).exists():
            log(f"Triggering 1200-baud soft-reset on {SERIAL_PORT}...")
            run_cmd(f"stty -F {SERIAL_PORT} 1200", check=False)
            time.sleep(3.0)
            mount_dir = find_pico_mount()

    if mount_dir is None:
        log_err("Could not find mounted Pico (RPI-RP2). Is the Pico plugged in?")
        log_err("Try manually holding the BOOTSEL button while plugging in USB.")
        sys.exit(1)

    log_ok(f"Found Pico bootloader drive at: {mount_dir}")
    log(f"Copying {UF2_PATH.name} to {mount_dir}...")
    run_cmd(f"cp {UF2_PATH} {mount_dir}/ && sync")
    log_ok("UF2 flashed. Pico is rebooting into runtime firmware...")

    # Wait for /dev/ttyACM0 to re-enumerate
    log("Waiting for USB CDC serial (/dev/ttyACM0) to re-appear...")
    for _ in range(20):
        time.sleep(0.5)
        if Path(SERIAL_PORT).exists():
            break

    if Path(SERIAL_PORT).exists():
        log_ok(f"Device reconnected at {SERIAL_PORT}!")
    else:
        log("Pico rebooted (serial port re-enumeration may take a few seconds).")


def cmd_stream(extra_args):
    """Run hardware verification test stream."""
    script_path = ROOT / "hardware" / "scripts" / "run_hardware_test.py"
    cmd = f"uv run python {script_path} --port {SERIAL_PORT} {' '.join(extra_args)}"
    run_cmd(cmd)


def cmd_listen(duration_s=30):
    """Monitor real-time serial output from Pico."""
    log(f"Listening to {SERIAL_PORT} for {duration_s}s (press Ctrl+C to stop)...")
    try:
        import serial
        s = serial.Serial(SERIAL_PORT, 115200, timeout=1.0)
        t0 = time.time()
        while time.time() - t0 < duration_s:
            line = s.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"\033[36m[PICO]\033[0m {line}")
        s.close()
    except KeyboardInterrupt:
        log("Stopped listening.")
    except Exception as e:
        log_err(f"Serial read error: {e}")


def cmd_monitor(http_port=8080):
    """Launch the zero-dependency live ECG oscilloscope & control UI."""
    script_path = ROOT / "hardware" / "scripts" / "live_ecg_monitor.py"
    cmd = f"uv run python {script_path} --port {SERIAL_PORT} --http-port {http_port}"
    run_cmd(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Pico WH Automated Build, Flash & Test Utility"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("build", help="Compile firmware using ninja")
    subparsers.add_parser("flash", help="Flash latest UF2 binary to Pico")
    subparsers.add_parser("all", help="Build, flash, and run test stream")

    stream_parser = subparsers.add_parser("stream", help="Stream test beats to Pico")
    stream_parser.add_argument("--subject", type=str, help="Subject name (e.g. Person_01)")
    stream_parser.add_argument("--beat-idx", type=int, help="Beat index in test set")
    stream_parser.add_argument("--impostor", action="store_true", help="Send impostor probe")
    stream_parser.add_argument("--beats", type=int, default=5, help="Number of beats to stream")

    listen_parser = subparsers.add_parser("listen", help="Monitor live serial logs")
    listen_parser.add_argument("--time", type=int, default=30, help="Listen duration in seconds")

    mon_parser = subparsers.add_parser("monitor", help="Launch live ECG oscilloscope web UI")
    mon_parser.add_argument("--http-port", type=int, default=8080, help="Local UI port")

    args, unknown = parser.parse_known_args()

    if args.command == "build":
        cmd_build()
    elif args.command == "flash":
        cmd_flash()
    elif args.command == "all":
        cmd_build()
        cmd_flash()
        time.sleep(2.0)
        cmd_stream(["--beats", "5"])
    elif args.command == "stream":
        stream_args = []
        if getattr(args, "subject", None):
            stream_args.extend(["--subject", args.subject])
        if getattr(args, "beat_idx", None) is not None:
            stream_args.extend(["--beat-idx", str(args.beat_idx)])
        if getattr(args, "impostor", False):
            stream_args.append("--impostor")
        if not getattr(args, "subject", None) and getattr(args, "beat_idx", None) is None and not getattr(args, "impostor", False):
            stream_args.extend(["--beats", str(args.beats)])
        stream_args.extend(unknown)
        cmd_stream(stream_args)
    elif args.command == "listen":
        cmd_listen(duration_s=args.time)
    elif args.command == "monitor":
        cmd_monitor(http_port=args.http_port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
