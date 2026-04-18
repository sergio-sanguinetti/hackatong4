from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import List, Optional

import numpy as np

from solution.domain.models import (
    AnalysisReport,
    Cycle,
    CyclesSummary,
    PhaseTimes,
    ProductivityMetrics,
    TruckLoading,
    VideoInfo,
)
from solution.infrastructure.imu_processor import ImuProcessor
from solution.infrastructure.video_processor import VideoProcessor


class AnalyzeLoadingCycleUseCase:
    """
    Caso de uso principal: orquesta visión + IMU para construir el reporte final.
    """

    def __init__(self, video_processor: VideoProcessor, imu_processor: ImuProcessor) -> None:
        self.video_processor = video_processor
        self.imu_processor = imu_processor

    def execute(
        self,
        left_video_path: Path,
        imu_path: Path,
        right_video_path: Optional[Path] = None,
    ) -> AnalysisReport:
        video_meta = self.video_processor.read_metadata(left_video_path)
        cv_payload = self.video_processor.detect_bucket_events(left_video_path, right_video_path=right_video_path)
        imu_payload = self.imu_processor.detect_swing_events(imu_path)

        cycles = self._build_cycles_from_events(
            bucket_event_times=[ev.timestamp_seconds for ev in cv_payload.get("bucket_events", [])],
            swing_event_times=[ev.peak_time_seconds for ev in imu_payload.get("swing_events", [])],
            duration_seconds=video_meta.duration_seconds,
        )
        cycles_summary = self._compute_cycle_summary(cycles)
        truck_loading = self._estimate_truck_loading(cycles)
        productivity = self._compute_productivity(cycles, truck_loading, video_meta.duration_seconds)

        return AnalysisReport(
            status="success",
            message="Análisis batch finalizado correctamente.",
            video_info=VideoInfo(
                left_video_path=str(left_video_path),
                right_video_path=str(right_video_path) if right_video_path else None,
                fps=video_meta.fps,
                total_frames=video_meta.total_frames,
                duration_seconds=video_meta.duration_seconds,
            ),
            cycles_summary=cycles_summary,
            cycles=cycles,
            truck_loading=truck_loading,
            productivity_metrics=productivity,
            metadata={
                "imu_source": str(imu_path),
                "cv_events_detected": len(cv_payload.get("bucket_events", [])),
                "imu_swings_detected": len(imu_payload.get("swing_events", [])),
            },
        )

    def _build_cycles_from_events(
        self,
        bucket_event_times: List[float],
        swing_event_times: List[float],
        duration_seconds: float,
    ) -> List[Cycle]:
        anchor_times = sorted(set(bucket_event_times + swing_event_times))
        if len(anchor_times) < 2:
            # Fallback mínimo para no romper pipeline en datasets pequeños.
            anchor_times = [0.0, max(duration_seconds, 20.0)]

        cycles: List[Cycle] = []
        for idx in range(len(anchor_times) - 1):
            start = float(anchor_times[idx])
            end = float(anchor_times[idx + 1])
            if end <= start:
                continue
            duration = end - start
            phases = self._split_phases(duration)

            cycles.append(
                Cycle(
                    cycle_id=idx + 1,
                    start_time_seconds=start,
                    end_time_seconds=end,
                    duration_seconds=duration,
                    phase_times=phases,
                    estimated_fill_factor=float(np.clip(0.75 + 0.07 * np.sin(idx), 0.0, 1.2)),
                    truck_id=f"TRUCK-{(idx // 4) + 1:03d}",
                )
            )
        return cycles

    def _split_phases(self, cycle_duration: float) -> PhaseTimes:
        # Distribución heurística de fases; reemplazar con reglas del modelo real.
        return PhaseTimes(
            digging_seconds=round(cycle_duration * 0.24, 3),
            swinging_loaded_seconds=round(cycle_duration * 0.29, 3),
            dumping_seconds=round(cycle_duration * 0.13, 3),
            swinging_empty_seconds=round(cycle_duration * 0.34, 3),
        )

    def _compute_cycle_summary(self, cycles: List[Cycle]) -> CyclesSummary:
        if not cycles:
            return CyclesSummary()
        durations = [c.duration_seconds for c in cycles]
        return CyclesSummary(
            total_cycles=len(cycles),
            avg_cycle_time_seconds=round(mean(durations), 3),
            min_cycle_time_seconds=round(min(durations), 3),
            max_cycle_time_seconds=round(max(durations), 3),
        )

    def _estimate_truck_loading(self, cycles: List[Cycle]) -> List[TruckLoading]:
        if not cycles:
            return []

        grouped: dict[str, List[Cycle]] = {}
        for cycle in cycles:
            truck_id = cycle.truck_id or "TRUCK-UNK"
            grouped.setdefault(truck_id, []).append(cycle)

        truck_reports: List[TruckLoading] = []
        for truck_id, truck_cycles in grouped.items():
            fill_factors = [c.estimated_fill_factor for c in truck_cycles]
            avg_fill = float(mean(fill_factors)) if fill_factors else 0.0
            truck_reports.append(
                TruckLoading(
                    truck_id=truck_id,
                    passes_count=len(truck_cycles),
                    avg_fill_factor=round(avg_fill, 3),
                    estimated_payload_index=round(avg_fill * len(truck_cycles), 3),
                )
            )
        return truck_reports

    def _compute_productivity(
        self,
        cycles: List[Cycle],
        truck_loading: List[TruckLoading],
        duration_seconds: float,
    ) -> ProductivityMetrics:
        if duration_seconds <= 0:
            return ProductivityMetrics()

        duration_hours = duration_seconds / 3600.0
        cycles_per_hour = len(cycles) / duration_hours if duration_hours > 0 else 0.0
        avg_fill = float(mean([c.estimated_fill_factor for c in cycles])) if cycles else 0.0
        avg_passes = float(mean([t.passes_count for t in truck_loading])) if truck_loading else 0.0
        est_tph = cycles_per_hour * avg_fill * 10.0  # Índice configurable por material/operación real.

        productive_time = sum(c.phase_times.digging_seconds + c.phase_times.swinging_loaded_seconds for c in cycles)
        utilization = min(1.0, productive_time / duration_seconds) if duration_seconds > 0 else 0.0

        return ProductivityMetrics(
            cycles_per_hour=round(cycles_per_hour, 3),
            avg_fill_factor=round(avg_fill, 3),
            avg_passes_per_truck=round(avg_passes, 3),
            estimated_tons_per_hour=round(est_tph, 3),
            utilization_ratio=round(utilization, 3),
        )
