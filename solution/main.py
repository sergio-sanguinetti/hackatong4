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
from fastapi.staticfiles import StaticFiles
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

# ─── Spillage detection parameters ───────────────────────────────────────────
SPILLAGE_PHASES        = {"SWING_LOADED", "DUMPING"}
SPILLAGE_MIN_BLOB      = 180      # px² minimum contour area counted as debris
SPILLAGE_MIN_TOTAL     = 900      # px² total motion to trigger an event
SPILLAGE_DEBOUNCE_S    = 1.2      # seconds between consecutive events
SPILLAGE_ROI_TOP       = 0.45     # start ROI at 45% of image height (below bucket)
SEVERITY_BANDS         = [(4500, "alta"), (2200, "media"), (0, "baja")]
SPILLAGE_FILE          = OUTPUTS / "spillage_events.json"


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


# ─── Spillage Detector (rocas que se desprenden durante el giro/descarga) ───

class SpillageDetector:
    """
    Detecta pérdidas de carga (rocas que caen de la cuchara) usando
    sustracción de fondo sobre el feed de video.

    Sólo registra eventos durante las fases SWING_LOADED y DUMPING, en la
    mitad inferior del frame (debajo de la trayectoria de la cuchara),
    y aplica un debounce temporal para evitar duplicados.

    Los eventos se persisten en outputs/spillage_events.json.
    """

    def __init__(self, out_path: Path = SPILLAGE_FILE):
        # Un sustractor de fondo por cámara (se crea bajo demanda)
        self._bgsubs: dict[str, "cv2.BackgroundSubtractor"] = {}
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._lock = threading.Lock()

        self.events: list[dict] = []
        self.last_event_ts: dict[str, float] = {}
        self.last_event_global: float = 0.0
        self.last_bbox: dict[str, tuple[int, int, int, int] | None] = {}
        self.last_bbox_until: dict[str, float] = {}

        self.out_path = out_path
        self._load_existing()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        """No cargamos eventos previos: cada sesión empieza limpia."""
        try:
            self.out_path.write_text(json.dumps({
                "events": [],
                "summary": self._empty_summary(),
            }, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"[Spillage] Warning creando {self.out_path}: {exc}")

    def _empty_summary(self) -> dict:
        return {
            "total_events":  0,
            "by_severity":   {"baja": 0, "media": 0, "alta": 0},
            "by_phase":      {"SWING_LOADED": 0, "DUMPING": 0},
            "by_camera":     {"LEFT": 0, "RIGHT": 0},
            "last_event":    None,
        }

    def _persist(self) -> None:
        """Guarda eventos + resumen en outputs/spillage_events.json."""
        try:
            payload = {
                "events":  self.events,
                "summary": self._build_summary(),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"[Spillage] Warning escribiendo {self.out_path}: {exc}")

    def _build_summary(self) -> dict:
        summary = self._empty_summary()
        for ev in self.events:
            summary["total_events"] += 1
            summary["by_severity"][ev["severity"]] = \
                summary["by_severity"].get(ev["severity"], 0) + 1
            summary["by_phase"][ev["phase"]] = \
                summary["by_phase"].get(ev["phase"], 0) + 1
            summary["by_camera"][ev["camera"]] = \
                summary["by_camera"].get(ev["camera"], 0) + 1
        summary["last_event"] = self.events[-1] if self.events else None
        return summary

    # ── Detección ────────────────────────────────────────────────────────────

    def _severity_for(self, area: float) -> str:
        for thr, label in SEVERITY_BANDS:
            if area >= thr:
                return label
        return "baja"

    def process(self, frame: np.ndarray, camera: str, phase: str,
                cycle: int, wall_ts: float) -> tuple[int, int, int, int] | None:
        """
        Procesa un frame. Devuelve un bounding box (x, y, w, h) si se
        dibujó algo sobre esta llamada (para anotación), o None.
        """
        if frame is None or frame.size == 0:
            return None

        if camera not in self._bgsubs:
            self._bgsubs[camera] = cv2.createBackgroundSubtractorMOG2(
                history=400, varThreshold=35, detectShadows=False
            )

        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        fg = self._bgsubs[camera].apply(small)

        # Aun cuando no estemos en fase crítica seguimos alimentando el BG subtractor
        if phase not in SPILLAGE_PHASES:
            self._maybe_expire_bbox(camera, wall_ts)
            return self.last_bbox.get(camera) if self._bbox_alive(camera, wall_ts) else None

        h, w = fg.shape
        y0 = int(h * SPILLAGE_ROI_TOP)
        roi = fg[y0:, :]
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, self._kernel)
        roi = cv2.dilate(roi, self._kernel, iterations=1)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) >= SPILLAGE_MIN_BLOB]
        total_area = float(sum(cv2.contourArea(c) for c in blobs))

        if not blobs or total_area < SPILLAGE_MIN_TOTAL:
            self._maybe_expire_bbox(camera, wall_ts)
            return self.last_bbox.get(camera) if self._bbox_alive(camera, wall_ts) else None

        # Debounce por cámara y global (una pérdida real se vería en ambas)
        last_cam   = self.last_event_ts.get(camera, 0.0)
        if wall_ts - last_cam < SPILLAGE_DEBOUNCE_S:
            return self.last_bbox.get(camera) if self._bbox_alive(camera, wall_ts) else None
        if wall_ts - self.last_event_global < SPILLAGE_DEBOUNCE_S * 0.6:
            return self.last_bbox.get(camera) if self._bbox_alive(camera, wall_ts) else None

        # Bounding box combinado (en coords del frame pequeño)
        xs, ys, xe, ye = w, h, 0, 0
        for c in blobs:
            x, y, ww, hh = cv2.boundingRect(c)
            xs = min(xs, x);    ys = min(ys, y + y0)
            xe = max(xe, x+ww); ye = max(ye, y + y0 + hh)
        # Escalar a coords del frame original
        fh, fw = frame.shape[:2]
        sx, sy = fw / w, fh / h
        bbox = (int(xs*sx), int(ys*sy), int((xe-xs)*sx), int((ye-ys)*sy))

        severity = self._severity_for(total_area)
        event = {
            "timestamp":    round(wall_ts, 3),
            "time_str":     time.strftime("%H:%M:%S", time.localtime(wall_ts)),
            "cycle":        cycle,
            "phase":        phase,
            "camera":       camera,
            "area_px":      int(total_area),
            "n_blobs":      len(blobs),
            "severity":     severity,
        }

        with self._lock:
            self.events.append(event)
            self.last_event_ts[camera]  = wall_ts
            self.last_event_global      = wall_ts
            self.last_bbox[camera]      = bbox
            self.last_bbox_until[camera] = wall_ts + 0.8
            self._persist()

        print(f"[Spillage] {event['time_str']}  cam={camera}  "
              f"phase={phase}  area={int(total_area)}px²  sev={severity}")
        return bbox

    def _bbox_alive(self, camera: str, now: float) -> bool:
        return now < self.last_bbox_until.get(camera, 0.0)

    def _maybe_expire_bbox(self, camera: str, now: float) -> None:
        if not self._bbox_alive(camera, now):
            self.last_bbox[camera] = None

    # ── API helpers ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            summary = self._build_summary()
            recent = self.events[-10:][::-1]
        return {
            "total_events":  summary["total_events"],
            "by_severity":   summary["by_severity"],
            "by_phase":      summary["by_phase"],
            "by_camera":     summary["by_camera"],
            "recent":        recent,
        }

    def all_events(self) -> list[dict]:
        with self._lock:
            return list(self.events)


# ─── Video Processor ─────────────────────────────────────────────────────────

class VideoProcessor:
    """Reads video in a background thread; provides annotated JPEG frames."""

    JPEG_QUALITY = 75

    def __init__(self, path: Path | None, label: str,
                 detector: CycleDetector,
                 spillage: SpillageDetector | None = None):
        self.path = path
        self.label = label
        self.detector = detector
        self.spillage = spillage
        self._cap: cv2.VideoCapture | None = None
        self._frame_bytes: bytes = b""
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_idx = 0
        self.fps = 30.0
        self.total_frames = 1
        self.finished = False
        self._final_frame: np.ndarray | None = None

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
                # End of video — keep showing the last frame with a banner.
                self.finished = True
                print(f"[Video/{self.label}] Fin del video "
                      f"({self._frame_idx}/{self.total_frames} frames).")
                on_video_finished()
                self._emit_final_frame()
                break

            self._frame_idx += 1
            wall_now = time.time()
            # Only left camera drives the cycle detector
            if self.label == "LEFT":
                self.detector.update(self._frame_idx, wall_now)

            # Spillage detection (runs on every frame, both cameras)
            spillage_bbox: tuple[int, int, int, int] | None = None
            if self.spillage is not None:
                metrics = self.detector.get_metrics()
                spillage_bbox = self.spillage.process(
                    frame, self.label,
                    metrics["phase"],
                    metrics["cycle_count"],
                    wall_now,
                )

            annotated = self._annotate(frame, spillage_bbox)
            ok, buf = cv2.imencode(".jpg", annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            if ok:
                with self._lock:
                    self._frame_bytes = buf.tobytes()

            self._final_frame = annotated

            elapsed = time.time() - t0
            sleep = max(0.0, delay - elapsed)
            time.sleep(sleep)

    def _emit_final_frame(self) -> None:
        """When the video ends, overlay a big 'ANÁLISIS TERMINADO' banner
        on the last frame and publish it so the MJPEG stream keeps showing it."""
        base = self._final_frame
        if base is None:
            base = np.zeros((360, 640, 3), dtype=np.uint8)
        frame = base.copy()
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        banner_h = max(70, int(h * 0.18))
        y1 = (h - banner_h) // 2
        y2 = y1 + banner_h
        cv2.rectangle(frame, (0, y1), (w, y2), (20, 120, 40), -1)
        cv2.rectangle(frame, (0, y1), (w, y1 + 4), (80, 220, 130), -1)
        cv2.rectangle(frame, (0, y2 - 4), (w, y2), (80, 220, 130), -1)

        text = "ANALISIS TERMINADO"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.9, w / 900)
        thick = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(frame, text,
                    ((w - tw) // 2, y1 + (banner_h + th) // 2),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        sub = f"{self.label} CAM - fin de reproduccion"
        (sw, sh), _ = cv2.getTextSize(sub, font, 0.55, 1)
        cv2.putText(frame, sub,
                    ((w - sw) // 2, y2 + sh + 14),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
        if ok:
            with self._lock:
                self._frame_bytes = buf.tobytes()

    def _annotate(self, frame: np.ndarray,
                  spillage_bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
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

        # ── Spillage marker (se dibuja al final para no perder intensidad) ──
        if spillage_bbox is not None:
            x, y, ww, hh = spillage_bbox
            x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
            ww = max(1, min(ww, w - x)); hh = max(1, min(hh, h - y))
            pad = 8
            x1 = max(0, x - pad);          y1 = max(0, y - pad)
            x2 = min(w, x + ww + pad);     y2 = min(h, y + hh + pad)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = "PERDIDA DE CARGA"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(th + 6, y1 - 4)
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 10, ty + 4),
                          (0, 0, 255), -1)
            cv2.putText(frame, label, (x1 + 5, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
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
spillage = SpillageDetector()

left_proc  = VideoProcessor(LEFT_VIDEO,  "LEFT",  detector, spillage)
right_proc = VideoProcessor(RIGHT_VIDEO, "RIGHT", detector, spillage)

# ─── Analysis-complete tracking ──────────────────────────────────────────────

_analysis_complete_flag = False
_analysis_complete_at: float | None = None
_analysis_complete_lock = threading.Lock()


def _active_processors() -> list["VideoProcessor"]:
    """Returns the processors that actually have a video loaded."""
    return [p for p in (left_proc, right_proc) if p.path is not None]


def is_analysis_complete() -> bool:
    """True when every loaded video has reached its end."""
    procs = _active_processors()
    if not procs:
        return False
    return all(p.finished for p in procs)


def on_video_finished() -> None:
    """
    Called from each VideoProcessor thread when its video ends.
    The first call that sees both videos finished writes the outputs.
    """
    global _analysis_complete_flag, _analysis_complete_at
    with _analysis_complete_lock:
        if _analysis_complete_flag:
            return
        if not is_analysis_complete():
            return
        _analysis_complete_flag = True
        _analysis_complete_at = time.time()

    print("[App] Ambos videos finalizados — escribiendo outputs/")
    try:
        metrics = detector.get_metrics()
        # Use the left video's frame count if available, else right.
        ref = left_proc if left_proc.path else right_proc
        _write_outputs(metrics, ref.total_frames, ref.fps)
    except Exception as exc:
        print(f"[App] Error escribiendo outputs: {exc}")

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
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS), html=True), name="outputs")

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


def _metrics_with_spillage() -> dict:
    m = detector.get_metrics()
    m["spillage"] = spillage.snapshot()
    m["analysis_complete"] = is_analysis_complete()
    m["video_status"] = {
        p.label: {
            "finished":    p.finished,
            "frame_idx":   p._frame_idx,
            "total":       p.total_frames,
            "progress_pct": round(
                100.0 * p._frame_idx / max(1, p.total_frames), 1
            ),
        }
        for p in _active_processors()
    }
    return m


@app.get("/api/metrics")
async def api_metrics():
    return _metrics_with_spillage()


@app.get("/api/spillage")
async def api_spillage():
    return {
        "summary": spillage.snapshot(),
        "events":  spillage.all_events(),
    }


@app.get("/metrics/stream")
async def metrics_sse(request: Request):
    """Server-Sent Events — pushes metrics every 500 ms."""
    async def event_gen():
        while True:
            if await request.is_disconnected():
                break
            data = json.dumps(_metrics_with_spillage())
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/reset")
async def api_reset():
    global _analysis_complete_flag, _analysis_complete_at
    with detector._lock:
        detector.state = MiningState()
    with spillage._lock:
        spillage.events.clear()
        spillage.last_event_ts.clear()
        spillage.last_event_global = 0.0
    spillage._persist()
    with _analysis_complete_lock:
        _analysis_complete_flag = False
        _analysis_complete_at = None
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
        "spillage": spillage.snapshot(),
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
    snap = spillage.snapshot()
    total_losses = snap["total_events"]
    if total_losses > 0:
        high = snap["by_severity"].get("alta", 0)
        med  = snap["by_severity"].get("media", 0)
        recs.append(
            f"Se detectaron {total_losses} pérdidas de carga "
            f"({high} altas, {med} medias). Revisar técnica de giro/descarga "
            "y estado de la cuchara.")
    if not recs:
        recs.append("Operación dentro de parámetros óptimos. Mantener consistencia.")
    return recs


def _generate_html_report(s: dict) -> str:
    recs_html = "".join(f"<li>{r}</li>" for r in s["recommendations"])

    sp = s.get("spillage", {"total_events": 0, "by_severity": {}, "recent": []})
    sev = sp.get("by_severity", {})
    spillage_rows = "".join(
        f"<tr><td>{e.get('time_str','-')}</td><td>{e.get('cycle','-')}</td>"
        f"<td>{e.get('phase','-')}</td><td>{e.get('camera','-')}</td>"
        f"<td>{e.get('area_px','-')}</td><td>{e.get('severity','-')}</td></tr>"
        for e in sp.get("recent", [])
    ) or "<tr><td colspan='6' style='text-align:center;color:#718096'>Sin pérdidas detectadas</td></tr>"

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

<h2>Pérdidas de Carga Detectadas</h2>
<div class="grid" style="grid-template-columns:repeat(4,1fr)">
  <div class="card"><div class="big" style="color:#fc8181">{sp.get("total_events", 0)}</div><div class="lbl">Total eventos</div></div>
  <div class="card"><div class="big" style="color:#fc8181">{sev.get("alta", 0)}</div><div class="lbl">Severidad alta</div></div>
  <div class="card"><div class="big" style="color:#f6a623">{sev.get("media", 0)}</div><div class="lbl">Severidad media</div></div>
  <div class="card"><div class="big" style="color:#a0aec0">{sev.get("baja", 0)}</div><div class="lbl">Severidad baja</div></div>
</div>
<table>
  <tr><th>Hora</th><th>Ciclo</th><th>Fase</th><th>Cámara</th><th>Área (px²)</th><th>Severidad</th></tr>
  {spillage_rows}
</table>
<p style="color:#718096;font-size:.8rem;margin-top:8px">
  Detalle completo en <code>outputs/spillage_events.json</code>.
</p>

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
