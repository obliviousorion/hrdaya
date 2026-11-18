#!/usr/bin/env python3
"""
live_ecg_monitor.py — Zero-Dependency Live ECG Oscilloscope & Control Console.

Connects to the Raspberry Pi Pico WH over USB CDC serial (/dev/ttyACM0), enables
100 Hz decimated ECG oscilloscope telemetry, and serves a dark-terminal live UI
at http://localhost:8080 using Python's standard library + pyserial + numpy.

Features:
  - Real-time scrolling HTML5 <canvas> ECG trace (5 s window @ 100 Hz display)
  - Live sensor state (LEADS OFF / SETTLING / SIGNAL READY), BPM & SQI counters
  - Cosine similarity bar (0.50 threshold), 5-beat vote banner & rolling log
  - 1-click Live Controls: ENROLL 5B, AUTH 5B, CONTINUOUS, STANDBY, CLEAR LIVE
  - 1-click Stage-1 Replay Controls: Person_01, Person_03, Impostor, 5-Beat Enroll Dry Run

Usage:
  uv run python hardware/scripts/live_ecg_monitor.py --port /dev/ttyACM0 --http-port 8080
"""

import argparse
import collections
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MATCH_RE = re.compile(
    r"\[(\d+)\]\s+Match:\s+(\S+)\s+\(probe:\s+([^)]+)\)\s+->\s+(ACCEPT|REJECT)"
    r"\s+sim=([-\d.]+)\s+t=(\d+)\s+us\s+\(([\d.]+)\s+ms\)"
)

STATE_NAMES = {0: "LEADS OFF", 1: "SETTLING", 2: "SIGNAL READY"}


class PicoBridge:
    """Thread-safe serial bridge between Pico WH and HTTP dashboard."""

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.error_msg = ""

        # 5 seconds of 100 Hz decimated telemetry
        self.wave = collections.deque([0.0] * 500, maxlen=500)
        self.rpeaks = collections.deque(maxlen=32)
        self.sample_seq = 0

        self.mode = "STANDBY"
        self.sensor = "LEADS OFF"
        self.bpm = 0.0
        self.live_users = 0
        self.prog = "0/0"
        self.vote = "---"
        self.sqi_rej = 0
        self.last_rej = "NONE"

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

        # Load test dataset lazily for Stage-1 replay controls
        self._test_data = None
        self._load_test_data()

    def _load_test_data(self):
        te_path = ROOT / "private" / "features_test.npz"
        if te_path.exists():
            te = np.load(te_path, allow_pickle=True)
            self._test_data = {
                "X_raw": te["X_raw"],
                "y": te["y"],
                "classes": te["classes"].tolist(),
            }

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

    def replay_subject(self, subject: str, count: int = 1):
        """Stream 1 or more held-out beats from features_test.npz (Stage-1 test)."""
        if subject == "IMPOSTOR":
            saw = ((np.arange(300) % 20) / 20.0).astype(np.float32)
            payload = ",".join(f"{v:.6f}" for v in saw) + "\n"
            for _ in range(count):
                self.send_raw(payload)
                time.sleep(0.45)
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
                time.sleep(0.55)
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
                    with self.lock:
                        self.wave.append(filt)
                        self.sample_seq += 1
                        self.sensor = STATE_NAMES.get(st_code, "LEADS OFF")
                        self.bpm = bpm
                except ValueError:
                    pass
            return

        if line.startswith("EVENT:RPEAK"):
            with self.lock:
                self.rpeaks.append(self.sample_seq)
            return

        if line.startswith("EVENT:STATUS,"):
            kv = {}
            for item in line[len("EVENT:STATUS,"):].split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    kv[k] = v
            with self.lock:
                self.mode = kv.get("mode", self.mode)
                self.sensor = kv.get("sensor", self.sensor)
                try:
                    self.bpm = float(kv.get("bpm", self.bpm))
                    self.live_users = int(kv.get("live", self.live_users))
                    self.sqi_rej = int(kv.get("sqi_rej", self.sqi_rej))
                except ValueError:
                    pass
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

        if line.startswith("EVENT:ENROLLED,"):
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
            # Convert absolute rpeak sequence numbers to 0..499 canvas indices
            rp_idx = [500 - (seq - r) for r in self.rpeaks if 0 <= (seq - r) < 500]
            return {
                "connected": self.connected,
                "error": self.error_msg,
                "mode": self.mode,
                "sensor": self.sensor,
                "bpm": round(self.bpm, 1),
                "live_users": self.live_users,
                "prog": self.prog,
                "vote": self.vote,
                "sqi_rej": self.sqi_rej,
                "last_rej": self.last_rej,
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
  .wrap{max-width:1100px;margin:0 auto}
  .top{display:flex;justify-content:space-between;align-items:center;border:1px solid #2a2a2a;
       background:#111;padding:10px 14px;margin-bottom:10px}
  .title{font-size:13px;letter-spacing:2px;color:#eee}
  .pill{font-size:11px;padding:3px 8px;border:1px solid #333;background:#161616}
  .grid{display:grid;grid-template-columns:2fr 1fr;gap:10px}
  @media(max-width:850px){.grid{grid-template-columns:1fr}}
  .card{border:1px solid #2a2a2a;background:#111;padding:12px;margin-bottom:10px}
  .chdr{font-size:10px;letter-spacing:2px;color:#666;border-bottom:1px solid #1e1e1e;
        padding-bottom:6px;margin-bottom:10px;display:flex;justify-content:space-between}
  canvas{width:100%;height:230px;background:#070707;border:1px solid #1e1e1e;display:block}
  .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1a1a1a;
       font-size:12px}
  .lbl{color:#666}
  .dec{font-size:20px;font-weight:bold;letter-spacing:3px;text-align:center;padding:10px;
       border:1px solid #222;background:#0d0d0d;margin:8px 0}
  .barwrap{position:relative;height:16px;background:#1a1a1a;border:1px solid #2c2c2c;margin:8px 0}
  .barfill{height:100%;width:0%;transition:width 0.15s}
  .barthr{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#f1fa8c}
  .btns{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  button{background:#181818;color:#ddd;border:1px solid #333;padding:7px 10px;cursor:pointer;
         font-family:inherit;font-size:11px;letter-spacing:1px}
  button:hover{background:#242424}
  .g{border-color:#5af78e;color:#5af78e}
  .c{border-color:#8be9fd;color:#8be9fd}
  .r{border-color:#f97583;color:#f97583}
  .y{border-color:#f1fa8c;color:#f1fa8c}
  input{background:#0b0b0b;color:#eee;border:1px solid #333;padding:6px 8px;
        font-family:inherit;font-size:11px;width:130px}
  table{width:100%;border-collapse:collapse;font-size:11px}
  th{text-align:left;color:#555;font-size:9px;padding:4px 2px;border-bottom:1px solid #1f1f1f}
  td{padding:4px 2px;border-bottom:1px solid #151515;color:#bbb}
  .log{font-size:10px;color:#888;max-height:145px;overflow-y:auto;line-height:1.45}
  .ok{color:#5af78e}.bad{color:#f97583}.warn{color:#f1fa8c}.cyan{color:#8be9fd}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="title">ECG-BIOMETRIC EDGE MONITOR // RP2040 PICO WH + AD8232</div>
    <div>
      <span class="pill" id="p-sensor">SENSOR: ---</span>
      <span class="pill" id="p-mode">MODE: ---</span>
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
        <canvas id="ecg" width="700" height="230"></canvas>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#555;margin-top:6px">
          <span id="p-sqi">SQI REJECTS: 0 (LAST: NONE)</span>
          <span>WINDOW: 5.0s // LEADS-ON 3s SETTLING GUARD ACTIVE</span>
        </div>
      </div>

      <div class="card">
        <div class="chdr"><span>LIVE AD8232 CONTROLS (MANUAL ARMED TRIGGERS)</span></div>
        <div class="btns">
          <input id="uname" type="text" placeholder="Name (e.g. Orion)" maxlength="14">
          <button class="g" onclick="sendCmd('ENROLL ' + document.getElementById('uname').value.trim())">[ ENROLL 5 BEATS ]</button>
          <button class="c" onclick="sendCmd('AUTH')">[ AUTHENTICATE 5 BEATS ]</button>
          <button class="y" onclick="sendCmd('CONT')">[ TOGGLE CONTINUOUS ]</button>
          <button onclick="sendCmd('STANDBY')">[ STANDBY ]</button>
          <button class="r" onclick="sendCmd('CLEAR')">[ CLEAR LIVE USERS ]</button>
        </div>
      </div>

      <div class="card">
        <div class="chdr"><span>STAGE-1 USB REPLAY &amp; DRY-RUN CONTROLS (PHYSIONET 89-SUBJECT DB)</span></div>
        <div class="btns">
          <button class="g" onclick="replay('Person_01',1)">[ REPLAY Person_01 ]</button>
          <button class="g" onclick="replay('Person_03',1)">[ REPLAY Person_03 ]</button>
          <button class="g" onclick="replay('Person_88',1)">[ REPLAY Person_88 ]</button>
          <button class="r" onclick="replay('IMPOSTOR',1)">[ REPLAY IMPOSTOR ]</button>
          <button class="c" onclick="enrollDryRun()">[ DRY-RUN: 5-BEAT ENROLL (Person_05) ]</button>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="chdr"><span>BIOMETRIC VERDICT</span><span id="p-prog">PROG: 0/0</span></div>
        <div class="dec" id="d-banner">[ STANDBY ]</div>
        <div class="row"><span class="lbl">IDENTITY</span><span id="d-name">---</span></div>
        <div class="row"><span class="lbl">COSINE SIM</span><span id="d-sim">0.0000 / 0.5000</span></div>
        <div class="barwrap"><div class="barfill" id="d-bar"></div><div class="barthr"></div></div>
        <div class="row"><span class="lbl">5-BEAT VOTE</span><span id="d-vote" class="cyan">---</span></div>
        <div class="row"><span class="lbl">CORE-1 LATENCY</span><span id="d-lat">0.0 ms</span></div>
        <div class="row"><span class="lbl">PROBE SOURCE</span><span id="d-probe">---</span></div>
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

function drawWave(wave, rpeaks) {
  const W = cvs.width, H = cvs.height;
  ctx.fillStyle = '#070707';
  ctx.fillRect(0, 0, W, H);

  // Subtle ECG grid
  ctx.strokeStyle = '#141414';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 35) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 23) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  // Zero baseline
  ctx.strokeStyle = '#222';
  ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();

  if (!wave || wave.length === 0) return;

  // Auto-scale around signal peak with sensible floor
  let peak = 0.15;
  for (let i = 0; i < wave.length; i++) {
    const a = Math.abs(wave[i]);
    if (a > peak) peak = a;
  }
  const scale = (H * 0.42) / peak;

  // R-peak vertical markers
  ctx.strokeStyle = 'rgba(241, 250, 140, 0.35)';
  for (const idx of (rpeaks || [])) {
    const x = (idx / (wave.length - 1)) * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  // Waveform path
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

async function sendCmd(cmd) {
  await fetch('/api/cmd', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cmd})
  });
}

async function replay(subject, count) {
  await fetch('/api/replay', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({subject, count})
  });
}

async function enrollDryRun() {
  const name = document.getElementById('uname').value.trim() || 'DryRun_P05';
  await sendCmd('ENROLL ' + name);
  setTimeout(() => replay('Person_05', 5), 250);
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();

    const ps = document.getElementById('p-sensor');
    ps.textContent = 'SENSOR: ' + d.sensor;
    ps.className = 'pill ' + (d.sensor === 'SIGNAL READY' ? 'ok' : (d.sensor === 'SETTLING' ? 'warn' : 'bad'));

    document.getElementById('p-mode').textContent = 'MODE: ' + d.mode;
    document.getElementById('p-db').textContent = 'DB: ' + d.live_users + ' LIVE + 89';
    document.getElementById('p-bpm').textContent = d.bpm > 20 ? ('BPM: ' + d.bpm.toFixed(0)) : 'BPM: --';
    document.getElementById('p-sqi').textContent = 'SQI REJECTS: ' + d.sqi_rej + ' (LAST: ' + d.last_rej + ')';
    document.getElementById('p-prog').textContent = 'PROG: ' + d.prog;
    document.getElementById('p-conn').textContent = d.connected ? 'USB CDC CONNECTED' : ('DISCONNECTED: ' + d.error);

    const lt = d.latest;
    const banner = document.getElementById('d-banner');
    banner.textContent = '[ ' + lt.decision + ' ]';
    banner.className = 'dec ' + (lt.decision === 'ACCEPT' ? 'ok' : (lt.decision === 'REJECT' ? 'bad' : ''));

    document.getElementById('d-name').textContent = lt.name;
    document.getElementById('d-sim').textContent = lt.sim.toFixed(4) + ' / 0.5000';
    document.getElementById('d-vote').textContent = d.vote;
    document.getElementById('d-lat').textContent = lt.lat_ms.toFixed(2) + ' ms';
    document.getElementById('d-probe').textContent = lt.probe;

    const pct = Math.max(0, Math.min(100, lt.sim * 100));
    const bar = document.getElementById('d-bar');
    bar.style.width = pct + '%';
    bar.style.background = lt.sim >= 0.5 ? '#5af78e' : '#f97583';

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
poll();
</script>
</body>
</html>
"""


def make_handler(bridge: PicoBridge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # quiet HTTP logs

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

            if self.path == "/api/replay":
                subj = str(data.get("subject", "Person_01"))
                cnt = int(data.get("count", 1))
                threading.Thread(
                    target=bridge.replay_subject, args=(subj, cnt), daemon=True
                ).start()
                self._send_json({"ok": True, "subject": subj, "count": cnt})
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
