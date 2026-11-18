#!/usr/bin/env python3
"""
live_ecg_monitor.py — Zero-Dependency Live ECG Oscilloscope, Recorder & Playback Console.

Connects to the Raspberry Pi Pico WH over USB CDC serial (/dev/ttyACM0), enables
100 Hz decimated ECG oscilloscope telemetry, and serves a dark-terminal live UI
at http://localhost:8080 using Python stdlib + pyserial + numpy + scipy.

Three-Tier Demo Support (All Neural Inference Runs on Physical Pico WH Core 1):
  - Tier 1 (Live AD8232): Real-time oscilloscope + manual armed ENROLL 5B / AUTH 5B / CONT
    + Selective per-user UNENROLL + Gallery Scope (LIVE ONLY vs ALL 89+N) + Threshold presets
  - Tier 2 (Pre-Recorded AD8232 Fallback — Approach A):
      * 1-Click Live Recorder ([ RECORD 20s ]) -> saves Nyquist-safe filtered trace +
        500 Hz resampled SQI beats into private/recordings/<label>_<ts>.npz (with Delete)
      * 1-Click Playback of saved recordings AND data/MLEE-live-recordings-2026-11-16.zip
        (.frames) -> animates real AD8232 trace & runs unified 5-beat enroll/auth on Pico Core 1
  - Tier 3 (PhysioNet 89-Subject Replay): 1-click Person_01 / Person_03 / Impostor parity

Usage:
  uv run python hardware/scripts/live_ecg_monitor.py --port /dev/ttyACM0 --http-port 8080
"""

import argparse
import collections
from datetime import datetime
import json
import re
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy import signal as sig

ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = ROOT / "private" / "recordings"
ARCHIVE_ZIP = ROOT / "data" / "MLEE-live-recordings-2026-11-16.zip"

ARCHIVE_CATALOG = [
    ("zip::P1_enroll", "P1 Enroll (60s, AD8232)", "MLEE-live-recordings-2026-11-16/raw/P1-s1-enroll-174755.frames"),
    ("zip::P1_r1", "P1 Test r1 (20s, same placement)", "MLEE-live-recordings-2026-11-16/raw/P1-config-174605-r1.frames"),
    ("zip::P1_r2", "P1 Test r2 (20s, same placement)", "MLEE-live-recordings-2026-11-16/raw/P1-config-174605-r2.frames"),
    ("zip::P1_r3", "P1 Test r3 (20s, same placement)", "MLEE-live-recordings-2026-11-16/raw/P1-config-174605-r3.frames"),
    ("zip::P1_r4", "P1 Test r4 (20s, same placement)", "MLEE-live-recordings-2026-11-16/raw/P1-config-174605-r4.frames"),
    ("zip::P1_s2_retake1", "P1 Placement-2 Retake 1 (30s)", "MLEE-live-recordings-2026-11-16/raw/P1-s2-test1-retake-175419.frames"),
    ("zip::P2_enroll", "P2 Impostor/Probe Enroll (60s)", "MLEE-live-recordings-2026-11-16/raw/P2-s1-enroll-175659.frames"),
    ("zip::P2_test1", "P2 Impostor/Probe Test 1 (30s)", "MLEE-live-recordings-2026-11-16/raw/P2-s1-test1-175803.frames"),
]

MATCH_RE = re.compile(
    r"\[(\d+)\]\s+Match:\s+(\S+)\s+\(probe:\s+([^)]+)\)\s+->\s+(ACCEPT|REJECT)"
    r"\s+sim=([-\d.]+)\s+t=(\d+)\s+us\s+\(([\d.]+)\s+ms\)"
)

STATE_NAMES = {0: "LEADS OFF", 1: "SETTLING", 2: "SIGNAL READY"}


def _extract_beats_500hz(y_500: np.ndarray, adc_500=None, lo_500=None, skip_samples: int = 0):
    """Extract SQI-verified 300-sample z-scored beats from a 500 Hz filtered signal."""
    if len(y_500) <= skip_samples + 400:
        return [], []
    deriv = np.diff(y_500, prepend=y_500[0])
    energy = np.convolve(deriv ** 2, np.ones(50) / 50.0, mode="same")
    tail = energy[skip_samples:]
    prom = max(float(np.percentile(tail, 90)) * 0.15, 1e-12)
    cands, _ = sig.find_peaks(tail, distance=175, prominence=prom)
    cands = cands + skip_samples

    beats = []
    peaks = []
    for p in cands:
        w0 = max(skip_samples, int(p) - 60)
        w1 = min(len(y_500), int(p) + 61)
        r = w0 + int(np.argmax(np.abs(y_500[w0:w1])))
        if r - 100 < skip_samples or r + 200 > len(y_500):
            continue
        if lo_500 is not None and np.any(lo_500[r - 100:r + 200] != 0):
            continue
        if adc_500 is not None:
            seg_adc = adc_500[r - 100:r + 200]
            if np.any((seg_adc < 50) | (seg_adc > 4045)):
                continue
        seg = y_500[r - 100:r + 200]
        std = float(np.std(seg))
        if std < 0.004:
            continue
        mean = float(np.mean(seg))
        if abs(float(seg[100]) - mean) / std < 2.2:
            continue
        beats.append(((seg - mean) / std).astype(np.float32))
        peaks.append(r)
    return beats, peaks


def load_archive_recording(member: str):
    """Load a 500 Hz .frames capture from ARCHIVE_ZIP -> (wave_100hz, beats_300, rpeaks_100)."""
    with zipfile.ZipFile(ARCHIVE_ZIP) as zf:
        lines = zf.read(member).decode("utf-8", errors="ignore").splitlines()
    adc, lo = [], []
    for line in lines:
        if not line.startswith("ECG1,"):
            continue
        parts = line.split(",")
        if len(parts) >= 5:
            adc.append(int(parts[3]))
            lo.append(int(parts[4]))
    adc_arr = np.asarray(adc, dtype=np.float64)
    lo_arr = np.asarray(lo, dtype=np.int32)

    fs = 500.0
    x = (adc_arr - 2048.0) / 2048.0
    b_hp, a_hp = sig.butter(2, 0.5, btype="highpass", fs=fs)
    b_lp, a_lp = sig.butter(2, 40.0, btype="lowpass", fs=fs)
    b_n, a_n = sig.iirnotch(50.0, 15.0, fs=fs)
    y_500 = sig.lfilter(b_n, a_n, sig.lfilter(b_lp, a_lp, sig.lfilter(b_hp, a_hp, x)))

    beats, peaks_500 = _extract_beats_500hz(y_500, adc_arr, lo_arr, skip_samples=1500)
    y_clean = y_500[1500:]
    wave_100 = y_clean[::5].astype(np.float32)
    rpeaks_100 = [int((r - 1500) // 5) for r in peaks_500]
    return wave_100, beats, rpeaks_100


def load_saved_recording(path: Path):
    """Load a user-recorded private/recordings/*.npz capture."""
    d = np.load(path, allow_pickle=True)
    wave_100 = d["wave_100hz"].astype(np.float32)
    beats = [b.astype(np.float32) for b in d["beats_300"]]
    rpeaks_100 = [int(r) for r in d["rpeaks_100"]]
    return wave_100, beats, rpeaks_100


class PicoBridge:
    """Thread-safe serial bridge, live recorder, and AD8232 playback engine."""

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.error_msg = ""

        self.wave = collections.deque([0.0] * 500, maxlen=500)
        self.rpeaks = collections.deque(maxlen=32)
        self.sample_seq = 0

        self.mode = "STANDBY"
        self.sensor = "LEADS OFF"
        self.bpm = 0.0
        self.live_users = 0
        self.enrolled_users = []
        self.scope = "ALL"
        self.threshold = 0.50
        self.prog = "0/0"
        self.vote = "---"
        self.sqi_rej = 0
        self.last_rej = "NONE"

        # 1-Click Live Recorder state
        self.rec_active = False
        self.rec_label = ""
        self.rec_target_samples = 2000
        self.rec_buf = []
        self.rec_status = "IDLE"

        # Pre-Recorded AD8232 Playback state
        self.play_active = False
        self.play_stop_evt = threading.Event()
        self.play_status = "IDLE"

        self.latest_match = {
            "beat": 0,
            "name": "---",
            "probe": "---",
            "decision": "STANDBY",
            "sim": 0.0,
            "lat_ms": 0.0,
        }
        self.history = collections.deque(maxlen=10)
        self.logs = collections.deque(maxlen=40)

        self._test_data = None
        self._load_test_data()
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_test_data(self):
        te_path = ROOT / "private" / "features_test.npz"
        if te_path.exists():
            te = np.load(te_path, allow_pickle=True)
            self._test_data = {
                "X_raw": te["X_raw"],
                "y": te["y"],
                "classes": te["classes"].tolist(),
            }

    def list_recordings(self):
        items = []
        if ARCHIVE_ZIP.exists():
            for key, label, _ in ARCHIVE_CATALOG:
                items.append({"id": key, "label": f"[Archive] {label}", "deletable": False})
        if RECORDINGS_DIR.exists():
            for p in sorted(RECORDINGS_DIR.glob("*.npz"), reverse=True):
                items.append({"id": f"file::{p.name}", "label": f"[Saved] {p.stem}", "deletable": True})
        return items

    def delete_saved_recording(self, rec_id: str):
        if not rec_id.startswith("file::"):
            return False, "Archive recordings are read-only; only saved recordings can be deleted"
        fname = rec_id[len("file::"):]
        path = (RECORDINGS_DIR / fname).resolve()
        if path.parent != RECORDINGS_DIR.resolve() or not path.exists():
            return False, f"Recording not found: {fname}"
        path.unlink()
        msg = f"Deleted saved recording: {fname}"
        with self.lock:
            self.rec_status = msg
            self.logs.appendleft(f"[rec] {msg}")
        return True, msg

    def start(self):
        import serial

        self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
        self.connected = True
        self.running = True
        time.sleep(0.5)
        self.send_cmd("TELEM ON")
        self.send_cmd("STATUS")
        t = threading.Thread(target=self._reader_loop, daemon=True)
        t.start()

    def stop(self):
        self.play_stop_evt.set()
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.send_cmd("TELEM OFF")
                time.sleep(0.1)
                self.ser.close()
        except Exception:
            pass

    def send_raw(self, text: str):
        with self.lock:
            if self.ser and self.ser.is_open:
                self.ser.write(text.encode("utf-8"))
                self.ser.flush()

    def send_cmd(self, cmd: str):
        self.send_raw(f"CMD:{cmd}\n")

    # ---- 1-Click Live AD8232 Recorder ---------------------------------------
    def start_recording(self, label: str = "", seconds: int = 20):
        clean = re.sub(r"[^A-Za-z0-9_-]", "_", label.strip()) or "Live_AD8232"
        with self.lock:
            self.rec_label = clean
            self.rec_target_samples = max(500, min(6000, int(seconds) * 100))
            self.rec_buf = []
            self.rec_active = True
            self.rec_status = f"RECORDING {clean} (0.0/{seconds:.0f}s)"
            self.logs.appendleft(f"[rec] Started {seconds}s capture: {clean}")
        return True, self.rec_status

    def stop_recording(self):
        with self.lock:
            if not self.rec_active:
                return False, "Not recording"
            self.rec_active = False
            buf = list(self.rec_buf)
            label = self.rec_label
        return self._finalize_recording(label, buf)

    def _finalize_recording(self, label: str, samples_100hz):
        if len(samples_100hz) < 500:
            msg = f"Too short ({len(samples_100hz) / 100.0:.1f}s < 5.0s) -- discarded"
            with self.lock:
                self.rec_status = msg
                self.logs.appendleft(f"[rec] {msg}")
            return False, msg

        wave_100 = np.asarray(samples_100hz, dtype=np.float32)
        y_500 = sig.resample_poly(wave_100.astype(np.float64), 5, 1)
        beats, peaks_500 = _extract_beats_500hz(y_500, skip_samples=100)
        rpeaks_100 = [int(r // 5) for r in peaks_500]

        ts = datetime.now().strftime("%H%M%S")
        out_path = RECORDINGS_DIR / f"{label}_{ts}.npz"
        np.savez(
            out_path,
            wave_100hz=wave_100,
            beats_300=np.asarray(beats, dtype=np.float32) if beats else np.zeros((0, 300), np.float32),
            rpeaks_100=np.asarray(rpeaks_100, dtype=np.int32),
            label=label,
            saved_at=datetime.now().isoformat(),
        )
        msg = f"SAVED {out_path.name} ({len(wave_100) / 100.0:.1f}s, {len(beats)} beats)"
        with self.lock:
            self.rec_status = msg
            self.logs.appendleft(f"[rec] {msg}")
        return True, msg

    # ---- Pre-Recorded AD8232 Playback Engine (Unified 5-Beat On-Chip Vote) --
    def start_playback(self, rec_id: str, action: str = "auth", enroll_name: str = ""):
        if self.play_active:
            self.play_stop_evt.set()
            time.sleep(0.15)

        try:
            if rec_id.startswith("zip::"):
                match = [m for k, _, m in ARCHIVE_CATALOG if k == rec_id]
                if not match:
                    return False, f"Unknown archive id: {rec_id}"
                wave_100, beats, rpeaks_100 = load_archive_recording(match[0])
                default_name = rec_id.replace("zip::", "")
            elif rec_id.startswith("file::"):
                fname = rec_id[len("file::"):]
                path = (RECORDINGS_DIR / fname).resolve()
                if path.parent != RECORDINGS_DIR.resolve() or not path.exists():
                    return False, f"Recording not found: {fname}"
                wave_100, beats, rpeaks_100 = load_saved_recording(path)
                default_name = path.stem[:14]
            else:
                return False, "Invalid recording id"
        except Exception as exc:
            return False, f"Load error: {exc}"

        if len(beats) < 5:
            return False, f"Only {len(beats)} clean beats in recording (need >= 5)"

        self.play_stop_evt.clear()
        self.play_active = True
        name = (enroll_name.strip() or default_name)[:14]
        t = threading.Thread(
            target=self._playback_worker,
            args=(wave_100, beats[:5], rpeaks_100[:5], action, name),
            daemon=True,
        )
        t.start()
        return True, f"Playing {default_name} ({action}, 5 beats)"

    def stop_playback(self):
        self.play_stop_evt.set()
        with self.lock:
            self.play_active = False
            self.play_status = "STOPPED"

    def _playback_worker(self, wave_100, beats, rpeaks_100, action, name):
        with self.lock:
            self.play_status = f"PLAYBACK {action.upper()}: {name} (0/5)"
            self.logs.appendleft(f"[play] Starting AD8232 {action.upper()} for '{name}'")

        if action == "enroll":
            self.send_cmd(f"ENROLL {name}")
            time.sleep(0.25)
        else:
            # Arm on-chip 5-beat majority voting so Tier 2 matches Tier 1 identically
            self.send_cmd("AUTH")
            time.sleep(0.25)

        end_idx = min(len(wave_100), (rpeaks_100[-1] + 60) if rpeaks_100 else 600)
        start_idx = max(0, (rpeaks_100[0] - 80) if rpeaks_100 else 0)
        beat_idx = 0
        last_sent_time = 0.0

        for i in range(start_idx, end_idx):
            if self.play_stop_evt.is_set() or not self.running:
                break
            with self.lock:
                self.wave.append(float(wave_100[i]))
                self.sample_seq += 1
                self.sensor = "PLAYBACK AD8232"
                if beat_idx < len(rpeaks_100) and i == rpeaks_100[beat_idx]:
                    self.rpeaks.append(self.sample_seq)

            if beat_idx < len(beats) and i >= rpeaks_100[beat_idx] + 40:
                now = time.time()
                if now - last_sent_time < 0.70:
                    time.sleep(0.70 - (now - last_sent_time))
                payload = ",".join(f"{v:.6f}" for v in beats[beat_idx]) + "\n"
                self.send_raw(payload)
                last_sent_time = time.time()
                beat_idx += 1
                with self.lock:
                    self.play_status = f"PLAYBACK {action.upper()}: {name} ({beat_idx}/5)"

            time.sleep(0.01)

        with self.lock:
            self.play_active = False
            self.play_status = f"DONE {action.upper()}: {name} ({beat_idx}/5)"
            self.logs.appendleft(f"[play] Finished {action.upper()} ({beat_idx}/5 beats sent to Pico)")

    # ---- Tier 3 PhysioNet Replay --------------------------------------------
    def replay_subject(self, subject: str, count: int = 1):
        if subject == "IMPOSTOR":
            saw = ((np.arange(300) % 20) / 20.0).astype(np.float32)
            payload = ",".join(f"{v:.6f}" for v in saw) + "\n"
            for _ in range(count):
                self.send_raw(payload)
                time.sleep(0.65)
            return True, "Streamed synthetic Impostor waveform"

        if not self._test_data:
            return False, "private/features_test.npz not found"

        classes = self._test_data["classes"]
        if subject not in classes:
            return False, f"Subject {subject} not in test set"

        cls_idx = classes.index(subject)
        idxs = np.where(self._test_data["y"] == cls_idx)[0]
        if len(idxs) == 0:
            return False, f"No test beats for {subject}"

        for k in range(min(count, len(idxs))):
            beat = self._test_data["X_raw"][idxs[k]]
            payload = ",".join(f"{v:.6f}" for v in beat) + "\n"
            self.send_raw(payload)
            if count > 1:
                time.sleep(0.85)
        return True, f"Streamed {min(count, len(idxs))} beat(s) of {subject}"

    def _reader_loop(self):
        while self.running:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self._parse_line(line)
            except Exception as exc:
                self.connected = False
                self.error_msg = str(exc)
                time.sleep(0.5)

    def _parse_line(self, line: str):
        if line.startswith("ECG,"):
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    filt = float(parts[2])
                    st_code = int(parts[3])
                    bpm = float(parts[4])
                    rec_done_args = None
                    with self.lock:
                        if not self.play_active:
                            self.wave.append(filt)
                            self.sample_seq += 1
                            self.sensor = STATE_NAMES.get(st_code, "LEADS OFF")
                            self.bpm = bpm
                        if self.rec_active:
                            self.rec_buf.append(filt)
                            cur = len(self.rec_buf)
                            tot = self.rec_target_samples
                            self.rec_status = f"RECORDING {self.rec_label} ({cur/100.0:.1f}/{tot/100.0:.0f}s)"
                            if cur >= tot:
                                self.rec_active = False
                                rec_done_args = (self.rec_label, list(self.rec_buf))
                    if rec_done_args is not None:
                        threading.Thread(
                            target=self._finalize_recording, args=rec_done_args, daemon=True
                        ).start()
                except ValueError:
                    pass
            return

        if line.startswith("EVENT:RPEAK"):
            with self.lock:
                if not self.play_active:
                    self.rpeaks.append(self.sample_seq)
            return

        if line.startswith("EVENT:USERS,"):
            body = line[len("EVENT:USERS,"):]
            users = []
            if body != "NONE":
                for tok in body.split(","):
                    if ":" in tok:
                        idx_s, uname = tok.split(":", 1)
                        try:
                            users.append({"slot": int(idx_s), "name": uname})
                        except ValueError:
                            pass
            with self.lock:
                self.enrolled_users = users
                self.live_users = len(users)
            return

        if line.startswith("EVENT:STATUS,"):
            kv = {}
            for item in line[len("EVENT:STATUS,"):].split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    kv[k] = v
            with self.lock:
                self.mode = kv.get("mode", self.mode)
                if not self.play_active:
                    self.sensor = kv.get("sensor", self.sensor)
                try:
                    self.bpm = float(kv.get("bpm", self.bpm))
                    self.live_users = int(kv.get("live", self.live_users))
                    self.sqi_rej = int(kv.get("sqi_rej", self.sqi_rej))
                    self.threshold = float(kv.get("thr", self.threshold))
                except ValueError:
                    pass
                self.scope = kv.get("scope", self.scope)
                self.prog = kv.get("prog", self.prog)
                self.vote = kv.get("vote", self.vote)
                self.last_rej = kv.get("last_rej", self.last_rej)
            return

        if line.startswith("EVENT:VOTE,"):
            parts = line.split(",")
            with self.lock:
                if len(parts) >= 5:
                    self.vote = f"{parts[2]} {parts[3]} ({parts[1]}, {parts[4]})"
                self.logs.appendleft(line)
            return

        if line.startswith("EVENT:ENROLLED,") or line.startswith("EVENT:UNENROLLED,"):
            with self.lock:
                self.logs.appendleft(line)
            return

        m = MATCH_RE.search(line)
        if m:
            entry = {
                "beat": int(m.group(1)),
                "name": m.group(2),
                "probe": m.group(3),
                "decision": m.group(4),
                "sim": float(m.group(5)),
                "lat_ms": float(m.group(7)),
            }
            with self.lock:
                self.latest_match = entry
                self.history.appendleft(entry)
                self.logs.appendleft(line)
            return

        with self.lock:
            self.logs.appendleft(line)

    def snapshot(self):
        with self.lock:
            seq = self.sample_seq
            rp_idx = [500 - (seq - r) for r in self.rpeaks if 0 <= (seq - r) < 500]
            return {
                "connected": self.connected,
                "error": self.error_msg,
                "mode": self.mode,
                "sensor": self.sensor,
                "bpm": round(self.bpm, 1),
                "live_users": self.live_users,
                "enrolled_users": list(self.enrolled_users),
                "scope": self.scope,
                "threshold": round(self.threshold, 2),
                "prog": self.prog,
                "vote": self.vote,
                "sqi_rej": self.sqi_rej,
                "last_rej": self.last_rej,
                "rec_active": self.rec_active,
                "rec_status": self.rec_status,
                "play_active": self.play_active,
                "play_status": self.play_status,
                "latest": self.latest_match,
                "history": list(self.history),
                "logs": list(self.logs)[:14],
                "wave": list(self.wave),
                "rpeaks": rp_idx,
            }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECG Biometric Edge Monitor // Pico WH</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0a;color:#c8c8c8;font-family:'Courier New',Courier,monospace;padding:14px}
  .wrap{max-width:1140px;margin:0 auto}
  .top{display:flex;justify-content:space-between;align-items:center;border:1px solid #2a2a2a;
       background:#111;padding:10px 14px;margin-bottom:10px;flex-wrap:wrap;gap:6px}
  .title{font-size:13px;letter-spacing:2px;color:#eee}
  .pill{font-size:11px;padding:3px 8px;border:1px solid #333;background:#161616}
  .grid{display:grid;grid-template-columns:2fr 1fr;gap:10px}
  @media(max-width:850px){.grid{grid-template-columns:1fr}}
  .card{border:1px solid #2a2a2a;background:#111;padding:12px;margin-bottom:10px}
  .chdr{font-size:10px;letter-spacing:2px;color:#666;border-bottom:1px solid #1e1e1e;
        padding-bottom:6px;margin-bottom:10px;display:flex;justify-content:space-between}
  canvas{width:100%;height:210px;background:#070707;border:1px solid #1e1e1e;display:block}
  .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1a1a1a;
       font-size:12px}
  .lbl{color:#666}
  .dec{font-size:20px;font-weight:bold;letter-spacing:3px;text-align:center;padding:10px;
       border:1px solid #222;background:#0d0d0d;margin:8px 0}
  .barwrap{position:relative;height:16px;background:#1a1a1a;border:1px solid #2c2c2c;margin:8px 0}
  .barfill{height:100%;width:0%;transition:width 0.15s}
  .barthr{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#f1fa8c}
  .btns{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;align-items:center}
  button{background:#181818;color:#ddd;border:1px solid #333;padding:6px 10px;cursor:pointer;
         font-family:inherit;font-size:11px;letter-spacing:1px}
  button:hover{background:#242424}
  .g{border-color:#5af78e;color:#5af78e}
  .c{border-color:#8be9fd;color:#8be9fd}
  .r{border-color:#f97583;color:#f97583}
  .y{border-color:#f1fa8c;color:#f1fa8c}
  .act{background:#262626;font-weight:bold}
  input,select{background:#0b0b0b;color:#eee;border:1px solid #333;padding:6px 8px;
               font-family:inherit;font-size:11px}
  input{width:130px}
  select{min-width:230px}
  table{width:100%;border-collapse:collapse;font-size:11px}
  th{text-align:left;color:#555;font-size:9px;padding:4px 2px;border-bottom:1px solid #1f1f1f}
  td{padding:4px 2px;border-bottom:1px solid #151515;color:#bbb}
  .urow{display:flex;justify-content:space-between;align-items:center;padding:5px 6px;
        border:1px solid #1f1f1f;background:#151515;margin-bottom:5px;font-size:11px}
  .log{font-size:10px;color:#888;max-height:130px;overflow-y:auto;line-height:1.45}
  .ok{color:#5af78e}.bad{color:#f97583}.warn{color:#f1fa8c}.cyan{color:#8be9fd}
  .statline{font-size:10px;color:#8be9fd;margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="title">ECG-BIOMETRIC EDGE MONITOR // RP2040 PICO WH + AD8232</div>
    <div>
      <span class="pill" id="p-sensor">SENSOR: ---</span>
      <span class="pill" id="p-mode">MODE: ---</span>
      <span class="pill" id="p-scope">SCOPE: ALL</span>
      <span class="pill" id="p-db">DB: 0 LIVE + 89</span>
    </div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <div class="chdr">
          <span>LIVE AD8232 OSCILLOSCOPE (500 Hz DSP / 50 Hz NOTCH)</span>
          <span id="p-bpm" class="cyan">BPM: --</span>
        </div>
        <canvas id="ecg" width="700" height="210"></canvas>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#555;margin-top:6px">
          <span id="p-sqi">SQI REJECTS: 0 (LAST: NONE)</span>
          <span>WINDOW: 5.0s // 3s SETTLING GUARD ACTIVE</span>
        </div>
      </div>

      <div class="card">
        <div class="chdr"><span>TIER 1: LIVE AD8232 ARMED CONTROLS, GALLERY SCOPE &amp; THRESHOLD</span></div>
        <div class="btns">
          <input id="uname" type="text" placeholder="Name (e.g. Orion)" maxlength="14">
          <button class="g" onclick="sendCmd('ENROLL ' + document.getElementById('uname').value.trim())">[ ENROLL 5 BEATS ]</button>
          <button class="c" onclick="sendCmd('AUTH')">[ AUTHENTICATE 5 BEATS ]</button>
          <button class="y" onclick="sendCmd('CONT')">[ CONTINUOUS ]</button>
          <button onclick="sendCmd('STANDBY')">[ STANDBY ]</button>
        </div>
        <div class="btns" style="margin-top:10px;padding-top:8px;border-top:1px solid #1e1e1e">
          <span style="font-size:10px;color:#666">DB SCOPE:</span>
          <button id="b-sc-live" class="y" onclick="sendCmd('SCOPE LIVE')">[ LIVE ONLY ]</button>
          <button id="b-sc-all" onclick="sendCmd('SCOPE ALL')">[ ALL 89+N ]</button>
          <span style="font-size:10px;color:#666;margin-left:8px">THRESHOLD:</span>
          <button id="b-t-50" onclick="sendCmd('THRESH 0.50')">[ 0.50 STD ]</button>
          <button id="b-t-60" onclick="sendCmd('THRESH 0.60')">[ 0.60 STRICT ]</button>
          <button id="b-t-68" onclick="sendCmd('THRESH 0.68')">[ 0.68 BANK ]</button>
        </div>
      </div>

      <div class="card">
        <div class="chdr"><span>TIER 2: 1-CLICK AD8232 RECORDER &amp; PRE-RECORDED FALLBACK (APPROACH A)</span></div>
        <div class="btns">
          <input id="reclabel" type="text" placeholder="Rec name" maxlength="18">
          <button class="r" onclick="recStart()">[ RECORD 20s ]</button>
          <button onclick="recStop()">[ STOP &amp; SAVE ]</button>
          <span id="p-rec" class="statline" style="margin:0 0 0 6px">REC: IDLE</span>
        </div>
        <div class="btns" style="margin-top:10px">
          <select id="rec-select"><option value="">Loading recordings...</option></select>
          <button class="g" onclick="playRec('enroll')">[ PLAYBACK: ENROLL 5B ]</button>
          <button class="c" onclick="playRec('auth')">[ PLAYBACK: AUTH 5B ]</button>
          <button onclick="playStop()">[ STOP ]</button>
          <button class="r" onclick="delRec()">[ DELETE SAVED ]</button>
        </div>
        <div id="p-play" class="statline">PLAYBACK: IDLE</div>
      </div>

      <div class="card">
        <div class="chdr"><span>TIER 3: PHYSIONET 89-SUBJECT REPLAY BENCHMARK</span></div>
        <div class="btns">
          <button class="g" onclick="replay('Person_01',1)">[ REPLAY Person_01 ]</button>
          <button class="g" onclick="replay('Person_03',1)">[ REPLAY Person_03 ]</button>
          <button class="g" onclick="replay('Person_88',1)">[ REPLAY Person_88 ]</button>
          <button class="r" onclick="replay('IMPOSTOR',1)">[ REPLAY IMPOSTOR ]</button>
          <button class="c" onclick="enrollDryRun()">[ DRY-RUN: ENROLL Person_05 ]</button>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="chdr"><span>BIOMETRIC VERDICT</span><span id="p-prog">PROG: 0/0</span></div>
        <div class="dec" id="d-banner">[ STANDBY ]</div>
        <div class="row"><span class="lbl">IDENTITY</span><span id="d-name">---</span></div>
        <div class="row"><span class="lbl">COSINE SIM</span><span id="d-sim">0.0000 / 0.50</span></div>
        <div class="barwrap"><div class="barfill" id="d-bar"></div><div class="barthr" id="d-thr"></div></div>
        <div class="row"><span class="lbl">5-BEAT VOTE</span><span id="d-vote" class="cyan">---</span></div>
        <div class="row"><span class="lbl">CORE-1 LATENCY</span><span id="d-lat">0.0 ms</span></div>
        <div class="row"><span class="lbl">PROBE SOURCE</span><span id="d-probe">---</span></div>
      </div>

      <div class="card">
        <div class="chdr">
          <span>ENROLLED LIVE USERS (PICO SRAM)</span>
          <button class="r" style="padding:2px 6px;font-size:9px" onclick="sendCmd('CLEAR')">[ CLEAR ALL ]</button>
        </div>
        <div id="user-list" style="font-size:11px;color:#555">No live users enrolled yet.</div>
      </div>

      <div class="card">
        <div class="chdr"><span>RECENT INFERENCES</span></div>
        <table>
          <thead><tr><th>#</th><th>MATCH</th><th>SIM</th><th>MS</th><th>DEC</th></tr></thead>
          <tbody id="hist-body"><tr><td colspan="5" style="color:#444">AWAITING...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="chdr"><span>PICO WH SERIAL LOG</span><span id="p-conn">CONNECTING...</span></div>
    <div class="log" id="log-box"></div>
  </div>
</div>

<script>
const cvs = document.getElementById('ecg');
const ctx = cvs.getContext('2d');
let prevRecActive = false;

function drawWave(wave, rpeaks) {
  const W = cvs.width, H = cvs.height;
  ctx.fillStyle = '#070707';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#141414';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 35) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 21) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  ctx.strokeStyle = '#222';
  ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();

  if (!wave || wave.length === 0) return;

  let peak = 0.15;
  for (let i = 0; i < wave.length; i++) {
    const a = Math.abs(wave[i]);
    if (a > peak) peak = a;
  }
  const scale = (H * 0.42) / peak;

  ctx.strokeStyle = 'rgba(241, 250, 140, 0.35)';
  for (const idx of (rpeaks || [])) {
    const x = (idx / (wave.length - 1)) * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  ctx.strokeStyle = '#5af78e';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < wave.length; i++) {
    const x = (i / (wave.length - 1)) * W;
    const y = (H / 2) - wave[i] * scale;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

async function postJson(url, obj) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(obj || {})
  });
  return r.json();
}

async function sendCmd(cmd) { await postJson('/api/cmd', {cmd}); }
async function unenrollSlot(slot) { await postJson('/api/unenroll', {slot}); }
async function replay(subject, count) { await postJson('/api/replay', {subject, count}); }
async function enrollDryRun() {
  const name = document.getElementById('uname').value.trim() || 'DryRun_P05';
  await sendCmd('ENROLL ' + name);
  setTimeout(() => replay('Person_05', 5), 250);
}
async function recStart() {
  const label = document.getElementById('reclabel').value.trim() || document.getElementById('uname').value.trim() || 'Live_AD8232';
  await postJson('/api/record/start', {label, seconds: 20});
}
async function recStop() {
  await postJson('/api/record/stop', {});
  setTimeout(loadRecordings, 400);
}
async function playRec(action) {
  const id = document.getElementById('rec-select').value;
  if (!id) return;
  const name = document.getElementById('uname').value.trim();
  await postJson('/api/playback/start', {id, action, name});
}
async function playStop() { await postJson('/api/playback/stop', {}); }
async function delRec() {
  const id = document.getElementById('rec-select').value;
  if (!id) return;
  const res = await postJson('/api/recording/delete', {id});
  if (!res.ok) alert(res.msg || 'Cannot delete archive recording');
  loadRecordings();
}

async function loadRecordings() {
  try {
    const r = await fetch('/api/recordings');
    const items = await r.json();
    const sel = document.getElementById('rec-select');
    const prev = sel.value;
    sel.innerHTML = items.map(it => `<option value="${it.id}">${it.label}</option>`).join('');
    if (prev && items.some(it => it.id === prev)) sel.value = prev;
  } catch (e) {}
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();

    const ps = document.getElementById('p-sensor');
    ps.textContent = 'SENSOR: ' + d.sensor;
    ps.className = 'pill ' + (
      d.sensor === 'SIGNAL READY' ? 'ok' :
      d.sensor === 'PLAYBACK AD8232' ? 'cyan' :
      d.sensor === 'SETTLING' ? 'warn' : 'bad'
    );

    document.getElementById('p-mode').textContent = 'MODE: ' + d.mode;
    const sc = document.getElementById('p-scope');
    sc.textContent = 'SCOPE: ' + (d.scope === 'LIVE' ? 'LIVE ONLY' : 'ALL 89+N') + ' // THR ' + d.threshold.toFixed(2);
    sc.className = 'pill ' + (d.scope === 'LIVE' ? 'warn' : '');
    document.getElementById('p-db').textContent = 'DB: ' + d.live_users + ' LIVE + 89';
    document.getElementById('p-bpm').textContent = d.bpm > 20 ? ('BPM: ' + d.bpm.toFixed(0)) : 'BPM: --';
    document.getElementById('p-sqi').textContent = 'SQI REJECTS: ' + d.sqi_rej + ' (LAST: ' + d.last_rej + ')';
    document.getElementById('p-prog').textContent = 'PROG: ' + d.prog;
    document.getElementById('p-conn').textContent = d.connected ? 'USB CDC CONNECTED' : ('DISCONNECTED: ' + d.error);
    document.getElementById('p-rec').textContent = 'REC: ' + d.rec_status;
    document.getElementById('p-play').textContent = 'PLAYBACK: ' + d.play_status;

    document.getElementById('b-sc-live').className = d.scope === 'LIVE' ? 'y act' : '';
    document.getElementById('b-sc-all').className = d.scope !== 'LIVE' ? 'c act' : '';
    document.getElementById('b-t-50').className = Math.abs(d.threshold - 0.50) < 0.02 ? 'g act' : '';
    document.getElementById('b-t-60').className = Math.abs(d.threshold - 0.60) < 0.02 ? 'y act' : '';
    document.getElementById('b-t-68').className = Math.abs(d.threshold - 0.68) < 0.02 ? 'r act' : '';

    if (prevRecActive && !d.rec_active) loadRecordings();
    prevRecActive = d.rec_active;

    const ul = document.getElementById('user-list');
    if (d.enrolled_users && d.enrolled_users.length) {
      ul.innerHTML = d.enrolled_users.map(u =>
        `<div class="urow"><span>#${u.slot} <b>${u.name}</b> <span style="color:#555">(192 B)</span></span>` +
        `<button class="r" style="padding:2px 7px;font-size:10px" onclick="unenrollSlot(${u.slot})">[ UNENROLL ]</button></div>`
      ).join('');
    } else {
      ul.innerHTML = '<div style="color:#444;font-size:11px">No live users enrolled yet.</div>';
    }

    const lt = d.latest;
    const banner = document.getElementById('d-banner');
    banner.textContent = '[ ' + lt.decision + ' ]';
    banner.className = 'dec ' + (lt.decision === 'ACCEPT' ? 'ok' : (lt.decision === 'REJECT' ? 'bad' : ''));

    document.getElementById('d-name').textContent = lt.name;
    document.getElementById('d-sim').textContent = lt.sim.toFixed(4) + ' / ' + d.threshold.toFixed(2);
    document.getElementById('d-vote').textContent = d.vote;
    document.getElementById('d-lat').textContent = lt.lat_ms.toFixed(2) + ' ms';
    document.getElementById('d-probe').textContent = lt.probe;

    const pct = Math.max(0, Math.min(100, lt.sim * 100));
    const bar = document.getElementById('d-bar');
    bar.style.width = pct + '%';
    bar.style.background = lt.sim >= d.threshold ? '#5af78e' : '#f97583';
    document.getElementById('d-thr').style.left = Math.max(5, Math.min(95, d.threshold * 100)) + '%';

    const hb = document.getElementById('hist-body');
    if (d.history.length) {
      hb.innerHTML = d.history.slice(0, 6).map(h =>
        `<tr><td>#${h.beat}</td><td>${h.name}</td><td>${h.sim.toFixed(4)}</td>` +
        `<td>${h.lat_ms.toFixed(0)}</td><td class="${h.decision==='ACCEPT'?'ok':'bad'}">[${h.decision}]</td></tr>`
      ).join('');
    }
    document.getElementById('log-box').innerHTML = d.logs.map(l => `<div>${l}</div>`).join('');
    drawWave(d.wave, d.rpeaks);
  } catch (e) {}
  setTimeout(poll, 100);
}
loadRecordings();
poll();
</script>
</body>
</html>
"""


def make_handler(bridge: PicoBridge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/state":
                self._send_json(bridge.snapshot())
                return
            if self.path == "/api/recordings":
                self._send_json(bridge.list_recordings())
                return
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {}

            if self.path == "/api/cmd":
                cmd = str(data.get("cmd", "")).strip()
                if cmd:
                    bridge.send_cmd(cmd)
                self._send_json({"ok": True, "cmd": cmd})
                return

            if self.path == "/api/unenroll":
                slot = data.get("slot")
                if slot is not None:
                    bridge.send_cmd(f"UNENROLL {slot}")
                self._send_json({"ok": True, "slot": slot})
                return

            if self.path == "/api/replay":
                subj = str(data.get("subject", "Person_01"))
                cnt = int(data.get("count", 1))
                threading.Thread(
                    target=bridge.replay_subject, args=(subj, cnt), daemon=True
                ).start()
                self._send_json({"ok": True, "subject": subj, "count": cnt})
                return

            if self.path == "/api/record/start":
                ok, msg = bridge.start_recording(
                    label=str(data.get("label", "")),
                    seconds=int(data.get("seconds", 20)),
                )
                self._send_json({"ok": ok, "msg": msg})
                return

            if self.path == "/api/record/stop":
                ok, msg = bridge.stop_recording()
                self._send_json({"ok": ok, "msg": msg})
                return

            if self.path == "/api/playback/start":
                ok, msg = bridge.start_playback(
                    rec_id=str(data.get("id", "")),
                    action=str(data.get("action", "auth")),
                    enroll_name=str(data.get("name", "")),
                )
                self._send_json({"ok": ok, "msg": msg})
                return

            if self.path == "/api/playback/stop":
                bridge.stop_playback()
                self._send_json({"ok": True})
                return

            if self.path == "/api/recording/delete":
                ok, msg = bridge.delete_saved_recording(str(data.get("id", "")))
                self._send_json({"ok": ok, "msg": msg})
                return

            self._send_json({"ok": False}, code=404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Live ECG Edge Monitor (Pico WH)")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Pico USB serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--http-port", type=int, default=8080, help="Local web UI port")
    args = parser.parse_args()

    bridge = PicoBridge(args.port, args.baud)
    try:
        bridge.start()
    except Exception as exc:
        print(f"ERROR: Could not open {args.port}: {exc}")
        sys.exit(1)

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(bridge))
    print(f"[monitor] Connected to Pico WH on {args.port}")
    print(f"[monitor] Open Live Oscilloscope -> http://localhost:{args.http_port}")
    print("[monitor] Press Ctrl+C to stop (auto-disables telemetry cleanly).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] Shutting down...")
    finally:
        bridge.stop()
        server.server_close()


if __name__ == "__main__":
    main()
