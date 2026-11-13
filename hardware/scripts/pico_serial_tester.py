"""Streaming tester: Streams real held-out ECG test beats to Raspberry Pi Pico over USB Serial."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Pico Serial ECG Tester")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (e.g., /dev/ttyACM0 on Linux, COM3 on Windows)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--num_beats", type=int, default=10, help="Number of test beats to stream")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("Please install pyserial first: uv add pyserial")
        sys.exit(1)

    private = ROOT / "private"
    te_path = private / "features_test.npz"
    if not te_path.exists():
        print("features_test.npz not found! Please run make data && make features first.")
        sys.exit(1)

    te = np.load(te_path, allow_pickle=True)
    X_test = te["X_raw"]
    y_test = te["y"]
    classes = te["classes"].tolist()

    print(f"Connecting to Raspberry Pi Pico WH on {args.port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2.0)
        time.sleep(2.0)  # Wait for DTR reset
    except Exception as e:
        print(f"Error opening serial port {args.port}: {e}")
        print("Tip: Make sure the Pico is flashed and running, and check port with 'ls -la /dev/ttyACM*'")
        sys.exit(1)

    print(f"Streaming {args.num_beats} test heartbeats to Pico...\n")
    print(f"{'Beat #':<8} {'True Identity':<15} {'Pico Serial Output'}")
    print("-" * 65)

    for i in range(min(args.num_beats, len(y_test))):
        beat = X_test[i]
        true_name = classes[y_test[i]]

        # Send 300 comma-separated floats
        payload = ",".join(f"{v:.5f}" for v in beat) + "\n"
        ser.write(payload.encode("utf-8"))

        # Read response line from Pico
        resp = ser.readline().decode("utf-8", errors="ignore").strip()
        print(f"[{i+1:2d}]     {true_name:<15} {resp}")
        time.sleep(0.2)

    ser.close()
    print("\nStreaming test complete.")


if __name__ == "__main__":
    main()
