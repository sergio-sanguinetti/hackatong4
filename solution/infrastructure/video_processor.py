from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    fps: float
    total_frames: int
    duration_seconds: float


@dataclass
class BucketEvent:
    frame_idx: int
    timestamp_seconds: float
    confidence: float
    event_type: str


class VideoProcessor:
    """
    Adaptador de salida para visión por computador.

    Este componente está preparado para:
    - Leer metadatos desde video estéreo (izquierdo/derecho).
    - Detectar eventos del cucharón (mock robusto de integración).

    TODO integración real:
    1) Reemplazar `_mock_bucket_events` por inferencia real con YOLO:
       - cargar modelo entrenado en `__init__` (ej. ultralytics.YOLO).
       - ejecutar inferencia por frame o por batch.
       - mapear detecciones a eventos de ciclo.
    2) Usar ambos streams (izq/der) para triangulación estereoscópica y
       cálculo de trayectoria/volumen del cucharón.
    """

    def read_metadata(self, left_video_path: Path) -> VideoMetadata:
        capture = cv2.VideoCapture(str(left_video_path))
        if not capture.isOpened():
            raise ValueError(f"No se pudo abrir el video: {left_video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        if fps <= 0:
            fps = 30.0
        duration_seconds = float(total_frames / fps) if fps > 0 else 0.0

        return VideoMetadata(fps=fps, total_frames=total_frames, duration_seconds=duration_seconds)

    def detect_bucket_events(
        self,
        left_video_path: Path,
        right_video_path: Optional[Path] = None,
    ) -> Dict[str, List[BucketEvent]]:
        """
        Retorna eventos sintéticos para que el pipeline sea ejecutable end-to-end.
        """
        metadata = self.read_metadata(left_video_path)
        events = self._mock_bucket_events(metadata)
        return {"bucket_events": events}

    def _mock_bucket_events(self, metadata: VideoMetadata) -> List[BucketEvent]:
        if metadata.duration_seconds <= 0:
            return []

        cycle_hint = 28.0
        n_events = max(1, int(metadata.duration_seconds // cycle_hint))
        timestamps = np.linspace(8.0, metadata.duration_seconds - 5.0, n_events).tolist()

        mock_events: List[BucketEvent] = []
        for idx, ts in enumerate(timestamps):
            frame_idx = int(ts * metadata.fps)
            confidence = float(np.clip(0.75 + (idx % 4) * 0.05, 0.0, 0.98))
            mock_events.append(
                BucketEvent(
                    frame_idx=frame_idx,
                    timestamp_seconds=float(ts),
                    confidence=confidence,
                    event_type="dump",
                )
            )
        return mock_events
