"""
Mining Shovel Dashboard — Jebi Hackathon 2026 Grupo 04
Real-time analysis of Hitachi EX-5600 shovel operations.

Usage:
    python solution/main.py            # Live dashboard server
    python solution/main.py --batch    # Batch mode (writes outputs/, then exits)
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
OUTPUTS.mkdir(exist_ok=True)

# ─── Mining constants (calibrated from manual measurements) ──────────────────

BUCKET_M3 = 34.0
FILL_FACTOR = 0.90
EFFECTIVE_M3 = BUCKET_M3 * FILL_FACTOR          # 30.6 m³
DENSITY_T_M3 = 1.6                               # loose material
TONS_PER_PASS = EFFECTIVE_M3 * DENSITY_T_M3     # ~49 t

# From manual measurements
PASSES_PER_TRUCK = 7
TRUCK_CAPACITY_TONS = 250.0                      # 240–260 t range
TARGET_CYCLE_S = 40.0                            # avg seconds/cycle
TRUCK_CHANGE_S = 21.0                            # 20–23 s range
TARGET_TRUCK_LOAD_S = 128.0                      # 2 min 8 s measured

# Phase timing targets (seconds)
PHASE_TARGETS = {
    "EXCAVATING":    11.0,   # 10–12 s
    "SWING_LOADED":  12.5,   # 10–15 s
    "DUMPING":        4.0,   # 3–5 s
    "SWING_EMPTY":   10.0,   # 8–12 s
    "TRUCK_CHANGE":  21.0,
    "IDLE":           0.0,
}

PHASE_COLORS = {
    "EXCAVATING":   "#e53e3e",
    "SWING_LOADED": "#dd6b20",
    "DUMPING":      "#38a169",
    "SWING_EMPTY":  "#3182ce",
    "TRUCK_CHANGE": "#805ad5",
    "IDLE":         "#718096",
    "INITIALIZING": "#4a5568",
}


# ─── Input discovery ─────────────────────────────────────────────────────────

def find_input(patterns: list[str]) -> Path | None:
    for p in patterns:
        matches = sorted(INPUTS.glob(p))
        if matches:
            return matches[0]
    return None

LEFT_VIDEO  = find_input(["shovel_left.mp4",  "*_left.mp4"])
RIGHT_VIDEO = find_input(["shovel_right.mp4", "*_right.mp4"])
IMU_FILE    = find_input(["imu_data.csv",     "*.npy",     "*.csv"])


# ─── IMU Loader ──────────────────────────────────────────────────────────────

def load_imu(path: Path) -> np.ndarray | None:
    """Load IMU data; returns (N, K) float array or None."""
    if path is None:
        return None
    try:
        if path.suffix == ".npy":
            data = np.load(str(path), allow_pickle=True)
            if data.dtype.names:                     # structured array
                names = list(data.dtype.names)
                data = np.column_stack([data[n] for n in names]).astype(float)
            return np.array(data, dtype=float)
        else:                                        # CSV
            df = pd.read_csv(path)
            return df.select_dtypes(include=[np.number]).values.astype(float)
    except Exception as exc:
        print(f"[IMU] Warning: could not load {path}: {exc}")
        return None


# ─── Cycle Detector (state machine driven by IMU gyro-Z) ─────────────────────

class CycleDetector:
    """
    State machine that parses the shovel's swing gyroscope signal.

    Column heuristic (tries several common orderings):
      • 7-col: [ts, ax, ay, az, gx, gy, gz]  → gyro_z = col 6
      • 6-col: [ax, ay, az, gx, gy, gz]       → gyro_z = col 5
      • fallback: column with highest variance → likely gyro_z
    """

    SWING_THRESH_FACTOR = 0.25   # fraction of max abs gyro for swing detection
    MIN_CYCLE_S = 20.0
    MAX_IDLE_S  = 60.0

    def __init__(self, imu: np.ndarray | None, video_fps: float = 30.0):
        self.imu = imu
        self.fps = video_fps
        self.gyro_z: np.ndarray | None = self._extract_gyro_z()
        self.swing_thresh = self._compute_threshold()

        self.state: MiningState = MiningState()
        self._lock = threading.Lock()

    def _extract_gyro_z(self) -> np.ndarray | None:
        if self.imu is None:
            return None
        n_cols = self.imu.shape[1] if self.imu.ndim == 2 else 1
        if n_cols >= 7:
            return self.imu[:, 6]
        elif n_cols == 6:
            return self.imu[:, 5]
        elif n_cols > 1:
            # pick column with highest variance (most likely gyro_z / yaw)
            variances = np.var(self.imu, axis=0)
            return self.imu[:, int(np.argmax(variances))]
        return None

    def _compute_threshold(self) -> float:
        if self.gyro_z is None or len(self.gyro_z) == 0:
            return 0.5
        peak = np.percentile(np.abs(self.gyro_z), 90)
        return max(peak * self.SWING_THRESH_FACTOR, 0.05)

    def imu_value_at_frame(self, frame_idx: int) -> float | None:
        """Get gyro_z value corresponding to the current video frame."""
        if self.gyro_z is None:
            return None
        # assume IMU and video share the same duration
        ratio = len(self.gyro_z) / max(1, self._total_frames)
        imu_idx = min(int(frame_idx * ratio), len(self.gyro_z) - 1)
        return float(self.gyro_z[imu_idx])

    _total_frames: int = 1  # set by VideoProcessor

    def update(self, frame_idx: int, wall_now: float) -> None:
        """Process one video frame tick; update state machine."""
        gz = self.imu_value_at_frame(frame_idx)
        with self._lock:
            self._tick(gz, wall_now)

    def _tick(self, gz: float | None, now: float) -> None:
        s = self.state
        phase = s.phase

        # ── Simulation fallback when no IMU ──────────────────────────────────
        if gz is None:
            self._simulate_tick(now)
            return

        thr = self.swing_thresh

        # ── Phase transitions ─────────────────────────────────────────────────
        if phase == "INITIALIZING":
            s.phase = "EXCAVATING"
            s.phase_start = now
            s.cycle_start = now

        elif phase == "EXCAVATING":
            if gz > thr:
                self._enter_phase("SWING_LOADED", now)

        elif phase == "SWING_LOADED":
            if gz < thr * 0.3:       # gyro slowing → near dump position
                self._enter_phase("DUMPING", now)

        elif phase == "DUMPING":
            if gz < -thr:             # swinging back (negative direction)
                self._enter_phase("SWING_EMPTY", now)

        elif phase == "SWING_EMPTY":
            if abs(gz) < thr * 0.3:  # stopped at dig face
                self._complete_cycle(now)

        elif phase == "TRUCK_CHANGE":
            if now - s.phase_start > TRUCK_CHANGE_S:
                self._enter_phase("EXCAVATING", now)

        elif phase == "IDLE":
            if gz is not None and abs(gz) > thr:
                self._enter_phase("EXCAVATING", now)

    # ── Simulation tick (used when no IMU data available) ────────────────────
    _sim_cycle_elapsed: float = 0.0

    def _simulate_tick(self, now: float) -> None:
        s = self.state
        elapsed = now - s.phase_start
        phase = s.phase

        thresholds = {
            "INITIALIZING":  0.5,
            "EXCAVATING":   11.0,
            "SWING_LOADED": 12.5,
            "DUMPING":       4.0,
            "SWING_EMPTY":  10.0,
            "TRUCK_CHANGE": TRUCK_CHANGE_S,
        }
        next_phase = {
            "INITIALIZING":  "EXCAVATING",
            "EXCAVATING":    "SWING_LOADED",
            "SWING_LOADED":  "DUMPING",
            "DUMPING":       "SWING_EMPTY",
            "SWING_EMPTY":   "_CYCLE_DONE",
            "TRUCK_CHANGE":  "EXCAVATING",
        }

        limit = thresholds.get(phase, 40.0)
        if elapsed >= limit:
            nxt = next_phase.get(phase)
            if nxt == "_CYCLE_DONE":
                self._complete_cycle(now)
            else:
                self._enter_phase(nxt or "EXCAVATING", now)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _enter_phase(self, new_phase: str, now: float) -> None:
        s = self.state
        duration = now - s.phase_start
        if s.phase in s.phase_durations:
            s.phase_durations[s.phase].append(duration)
        s.phase = new_phase
        s.phase_start = now

    def _complete_cycle(self, now: float) -> None:
        s = self.state
        cycle_time = now - s.cycle_start
        if cycle_time < self.MIN_CYCLE_S:
            # too short — likely noise, restart excavating
            self._enter_phase("EXCAVATING", now)
            return

        s.cycle_count += 1
        s.passes_this_truck += 1
        s.cycle_times.append(cycle_time)
        s.tons_this_truck += TONS_PER_PASS
        s.total_tons += TONS_PER_PASS

        if s.passes_this_truck >= PASSES_PER_TRUCK:
            # truck is full
            s.trucks_completed += 1
            s.passes_this_truck = 0
            s.tons_this_truck = 0.0
            self._enter_phase("TRUCK_CHANGE", now)
        else:
            self._enter_phase("EXCAVATING", now)
        s.cycle_start = now

    def get_metrics(self) -> dict:
        with self._lock:
            s = self.state
            now = time.time()
            elapsed_session = now - s.session_start
            recent = list(s.cycle_times)[-10:]
            avg_cycle = float(np.mean(recent)) if recent else TARGET_CYCLE_S

            # Production rate in t/h
            if elapsed_session > 0:
                tph = (s.total_tons / elapsed_session) * 3600
            else:
                tph = 0.0

            # Efficiency: actual cycle vs target
            eff = min(100.0, (TARGET_CYCLE_S / avg_cycle * 100)) if avg_cycle > 0 else 100.0

            phase_avgs = {
                ph: float(np.mean(durs)) if durs else PHASE_TARGETS[ph]
                for ph, durs in s.phase_durations.items()
            }

            return {
                "phase":                s.phase,
                "phase_color":          PHASE_COLORS.get(s.phase, "#718096"),
                "phase_elapsed_s":      round(now - s.phase_start, 1),
                "cycle_count":          s.cycle_count,
                "passes_this_truck":    s.passes_this_truck,
                "tons_this_truck":      round(s.tons_this_truck, 1),
                "truck_fill_pct":       round(s.passes_this_truck / PASSES_PER_TRUCK * 100, 1),
                "trucks_completed":     s.trucks_completed,
                "total_tons":           round(s.total_tons, 1),
                "avg_cycle_s":          round(avg_cycle, 1),
                "target_cycle_s":       TARGET_CYCLE_S,
                "production_tph":       round(tph, 1),
                "efficiency_pct":       round(eff, 1),
                "session_elapsed_s":    round(elapsed_session, 0),
                "recent_cycle_times":   [round(t, 1) for t in recent],
                "phase_avg_durations":  {k: round(v, 1) for k, v in phase_avgs.items()},
                # constants for UI
                "bucket_m3":            BUCKET_M3,
                "effective_m3":         EFFECTIVE_M3,
                "tons_per_pass":        round(TONS_PER_PASS, 1),
                "passes_per_truck":     PASSES_PER_TRUCK,
                "truck_capacity_tons":  TRUCK_CAPACITY_TONS,
            }


class MiningState:
    def __init__(self):
        self.phase = "INITIALIZING"
        self.phase_start = time.time()
        self.cycle_start = time.time()
        self.session_start = time.time()
        self.cycle_count = 0
        self.passes_this_truck = 0
        self.tons_this_truck = 0.0
        self.total_tons = 0.0
        self.trucks_completed = 0
        self.cycle_times: deque[float] = deque(maxlen=50)
        self.phase_durations: dict[str, deque] = {
            ph: deque(maxlen=30) for ph in PHASE_TARGETS
        }


# ─── Video Processor ─────────────────────────────────────────────────────────

class VideoProcessor:
    """Reads video in a background thread; provides annotated JPEG frames."""

    JPEG_QUALITY = 75

    def __init__(self, path: Path | None, label: str, detector: CycleDetector):
        self.path = path
        self.label = label
        self.detector = detector
        self._cap: cv2.VideoCapture | None = None
        self._frame_bytes: bytes = b""
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_idx = 0
        self.fps = 30.0
        self.total_frames = 1

    def start(self) -> None:
        if self.path is None:
            print(f"[Video/{self.label}] No file found — will show placeholder.")
            return
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            print(f"[Video/{self.label}] Cannot open {self.path}")
            return
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.detector._total_frames = self.total_frames
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()

    def _loop(self) -> None:
        delay = 1.0 / self.fps
        while self._running:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                continue

            self._frame_idx += 1
            # Only left camera drives the detector
            if self.label == "LEFT":
                self.detector.update(self._frame_idx, time.time())

            annotated = self._annotate(frame)
            ok, buf = cv2.imencode(".jpg", annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            if ok:
                with self._lock:
                    self._frame_bytes = buf.tobytes()

            elapsed = time.time() - t0
            sleep = max(0.0, delay - elapsed)
            time.sleep(sleep)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        metrics = self.detector.get_metrics()
        phase = metrics["phase"]
        color_hex = PHASE_COLORS.get(phase, "#718096")
        bgr = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 45), (15, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Phase badge
        cv2.rectangle(frame, (0, 0), (w, 40), bgr, -1)
        txt = f"  {phase}  |  Ciclo #{metrics['cycle_count']}  |  Pase {metrics['passes_this_truck']}/{metrics['passes_per_truck']}"
        cv2.putText(frame, txt, (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom bar
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 17, 23), -1)
        bottom_txt = (f"  {self.label} CAM  |  "
                      f"{metrics['avg_cycle_s']}s/ciclo  |  "
                      f"{metrics['production_tph']} t/h  |  "
                      f"Camion #{metrics['trucks_completed'] + 1}")
        cv2.putText(frame, bottom_txt, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Truck fill progress bar
        bar_w = int(w * metrics["truck_fill_pct"] / 100)
        cv2.rectangle(frame, (0, h - 38), (bar_w, h - 35), bgr, -1)
        return frame

    def get_frame_bytes(self) -> bytes:
        with self._lock:
            return self._frame_bytes

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        """Yields multipart MJPEG chunks."""
        placeholder = self._make_placeholder()
        while True:
            data = self.get_frame_bytes() or placeholder
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(1.0 / 30)

    def _make_placeholder(self) -> bytes:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"No video: {self.label}", (180, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()


# ─── App bootstrap ────────────────────────────────────────────────────────────

imu_data = load_imu(IMU_FILE)
detector = CycleDetector(imu_data)

left_proc  = VideoProcessor(LEFT_VIDEO,  "LEFT",  detector)
right_proc = VideoProcessor(RIGHT_VIDEO, "RIGHT", detector)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    left_proc.start()
    right_proc.start()
    print(f"[App] Left  video : {LEFT_VIDEO}")
    print(f"[App] Right video : {RIGHT_VIDEO}")
    print(f"[App] IMU file    : {IMU_FILE}")
    print(f"[App] Dashboard   : http://localhost:8000")
    yield
    left_proc.stop()
    right_proc.stop()


app = FastAPI(title="Mining Shovel Dashboard — Grupo 04", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video/left")
async def video_left():
    return StreamingResponse(
        left_proc.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/right")
async def video_right():
    return StreamingResponse(
        right_proc.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/metrics")
async def api_metrics():
    return detector.get_metrics()


@app.get("/metrics/stream")
async def metrics_sse(request: Request):
    """Server-Sent Events — pushes metrics every 500 ms."""
    async def event_gen():
        while True:
            if await request.is_disconnected():
                break
            data = json.dumps(detector.get_metrics())
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/reset")
async def api_reset():
    with detector._lock:
        detector.state = MiningState()
    return {"status": "reset"}


# ─── Batch mode ──────────────────────────────────────────────────────────────

def run_batch() -> None:
    """
    Non-interactive mode for hackathon evaluation.
    Processes full video, writes JSON + HTML report to ./outputs/.
    """
    print("[Batch] Starting analysis...")

    cap_left  = cv2.VideoCapture(str(LEFT_VIDEO))  if LEFT_VIDEO  else None
    cap_right = cv2.VideoCapture(str(RIGHT_VIDEO)) if RIGHT_VIDEO else None

    if cap_left is None and cap_right is None:
        print("[Batch] ERROR: No video files found in inputs/")
        sys.exit(1)

    cap = cap_left or cap_right
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector._total_frames = total

    print(f"[Batch] Processing {total} frames @ {fps:.1f} fps...")
    start_wall = time.time()
    frame_idx = 0
    report_interval = int(fps * 30)   # status every 30 s of video

    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_idx += 1
        video_time = frame_idx / fps
        detector.update(frame_idx, start_wall + video_time)

        if frame_idx % report_interval == 0:
            pct = frame_idx / total * 100
            m = detector.get_metrics()
            print(f"  {pct:.0f}%  |  {m['cycle_count']} ciclos  |  "
                  f"{m['trucks_completed']} camiones  |  {m['production_tph']} t/h")

    if cap_left:  cap_left.release()
    if cap_right: cap_right.release()

    metrics = detector.get_metrics()
    _write_outputs(metrics, total, fps)
    print("[Batch] Done. Outputs written to ./outputs/")


def _write_outputs(metrics: dict, total_frames: int, fps: float) -> None:
    duration_min = total_frames / fps / 60

    summary = {
        "grupo": "04",
        "video_duration_min": round(duration_min, 2),
        "total_cycles": metrics["cycle_count"],
        "trucks_completed": metrics["trucks_completed"],
        "total_tons_moved": metrics["total_tons"],
        "avg_cycle_time_s": metrics["avg_cycle_s"],
        "target_cycle_time_s": TARGET_CYCLE_S,
        "production_tph": metrics["production_tph"],
        "efficiency_pct": metrics["efficiency_pct"],
        "passes_per_truck": PASSES_PER_TRUCK,
        "tons_per_pass": round(TONS_PER_PASS, 1),
        "bucket_m3": BUCKET_M3,
        "fill_factor": FILL_FACTOR,
        "constants": {
            "bucket_capacity_m3": BUCKET_M3,
            "fill_factor": FILL_FACTOR,
            "effective_m3": EFFECTIVE_M3,
            "material_density_t_m3": DENSITY_T_M3,
            "tons_per_pass": round(TONS_PER_PASS, 1),
            "target_truck_capacity_t": TRUCK_CAPACITY_TONS,
            "measured_passes_per_truck": PASSES_PER_TRUCK,
            "measured_truck_load_time_s": TARGET_TRUCK_LOAD_S,
            "measured_truck_change_s": TRUCK_CHANGE_S,
        },
        "recommendations": _generate_recommendations(metrics),
    }

    # JSON
    json_path = OUTPUTS / "analysis.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Batch] Wrote {json_path}")

    # HTML report
    html_path = OUTPUTS / "report.html"
    html_path.write_text(_generate_html_report(summary))
    print(f"[Batch] Wrote {html_path}")


def _generate_recommendations(m: dict) -> list[str]:
    recs = []
    avg = m["avg_cycle_s"]
    if avg > TARGET_CYCLE_S * 1.1:
        recs.append(
            f"Tiempo de ciclo promedio ({avg}s) supera el objetivo ({TARGET_CYCLE_S}s). "
            "Revisar tiempos de giro y descarga.")
    if m["efficiency_pct"] < 90:
        recs.append(
            f"Eficiencia actual ({m['efficiency_pct']}%) por debajo del 90%. "
            "Analizar fases de mayor duración.")
    if not recs:
        recs.append("Operación dentro de parámetros óptimos. Mantener consistencia.")
    return recs


def _generate_html_report(s: dict) -> str:
    recs_html = "".join(f"<li>{r}</li>" for r in s["recommendations"])
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Análisis Pala — Jebi Hackathon 2026 Grupo 04</title>
<style>
  body{{font-family:Arial,sans-serif;background:#0f1117;color:#e2e8f0;margin:0;padding:24px}}
  h1{{color:#f6a623}}h2{{color:#90cdf4;margin-top:2rem}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}
  .card{{background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;padding:20px;text-align:center}}
  .big{{font-size:2.4rem;font-weight:700;color:#f6a623}}
  .lbl{{font-size:.8rem;color:#718096;text-transform:uppercase;letter-spacing:.05em}}
  table{{width:100%;border-collapse:collapse;background:#1a1f2e;border-radius:10px;overflow:hidden}}
  th{{background:#2d3748;padding:12px;text-align:left;font-size:.85rem}}
  td{{padding:10px 12px;border-top:1px solid #2d3748}}
  ul{{background:#1a1f2e;border-radius:10px;padding:20px 20px 20px 36px;border-left:4px solid #f6a623}}
  li{{margin-bottom:8px}}
</style>
</head>
<body>
<h1>⛏ Análisis de Productividad — Pala Hitachi EX-5600</h1>
<p style="color:#718096">Jebi Hackathon 2026 · Grupo 04 · Video: {s["video_duration_min"]} min</p>

<div class="grid">
  <div class="card"><div class="big">{s["total_cycles"]}</div><div class="lbl">Ciclos Totales</div></div>
  <div class="card"><div class="big">{s["trucks_completed"]}</div><div class="lbl">Camiones Cargados</div></div>
  <div class="card"><div class="big">{s["total_tons_moved"]:.0f} t</div><div class="lbl">Toneladas Movidas</div></div>
  <div class="card"><div class="big">{s["production_tph"]} t/h</div><div class="lbl">Tasa de Producción</div></div>
</div>

<h2>Parámetros de Operación</h2>
<table>
  <tr><th>Parámetro</th><th>Medido / Calculado</th><th>Referencia</th></tr>
  <tr><td>Tiempo ciclo promedio</td><td>{s["avg_cycle_time_s"]}s</td><td>{TARGET_CYCLE_S}s</td></tr>
  <tr><td>Eficiencia del ciclo</td><td>{s["efficiency_pct"]}%</td><td>≥ 90%</td></tr>
  <tr><td>Pases por camión</td><td>{s["passes_per_truck"]}</td><td>7 (medido)</td></tr>
  <tr><td>Toneladas por pase</td><td>{s["tons_per_pass"]} t</td><td>~49 t</td></tr>
  <tr><td>Capacidad efectiva</td><td>{s["constants"]["effective_m3"]} m³</td><td>34 × 0.90</td></tr>
  <tr><td>Densidad material</td><td>{s["constants"]["material_density_t_m3"]} t/m³</td><td>suelto</td></tr>
</table>

<h2>Recomendaciones</h2>
<ul>{recs_html}</ul>

<p style="color:#4a5568;margin-top:3rem;font-size:.8rem">
  Generado automáticamente · Jebi Hackathon 2026 · Grupo 04
</p>
</body>
</html>"""


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--batch" in sys.argv:
        run_batch()
    else:
        uvicorn.run("solution.main:app", host="0.0.0.0", port=8000,
                    reload=False, log_level="info")
