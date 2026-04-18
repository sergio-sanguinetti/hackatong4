from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    left_video_path: str
    right_video_path: Optional[str] = None
    fps: float
    total_frames: int
    duration_seconds: float
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class PhaseTimes(BaseModel):
    digging_seconds: float = 0.0
    swinging_loaded_seconds: float = 0.0
    dumping_seconds: float = 0.0
    swinging_empty_seconds: float = 0.0


class Cycle(BaseModel):
    cycle_id: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    phase_times: PhaseTimes = Field(default_factory=PhaseTimes)
    estimated_fill_factor: float = Field(default=0.0, ge=0.0, le=1.2)
    truck_id: Optional[str] = None


class CyclesSummary(BaseModel):
    total_cycles: int = 0
    avg_cycle_time_seconds: float = 0.0
    min_cycle_time_seconds: float = 0.0
    max_cycle_time_seconds: float = 0.0


class TruckLoading(BaseModel):
    truck_id: str
    passes_count: int = 0
    avg_fill_factor: float = 0.0
    estimated_payload_index: float = 0.0


class ProductivityMetrics(BaseModel):
    cycles_per_hour: float = 0.0
    avg_fill_factor: float = 0.0
    avg_passes_per_truck: float = 0.0
    estimated_tons_per_hour: float = 0.0
    utilization_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalysisReport(BaseModel):
    status: Literal["success", "warning", "error"] = "success"
    message: str = "Análisis completado."
    video_info: VideoInfo
    cycles_summary: CyclesSummary
    cycles: List[Cycle] = Field(default_factory=list)
    truck_loading: List[TruckLoading] = Field(default_factory=list)
    productivity_metrics: ProductivityMetrics = Field(default_factory=ProductivityMetrics)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def output_filename(self) -> str:
        left_name = Path(self.video_info.left_video_path).stem
        timestamp = self.video_info.analyzed_at.strftime("%Y%m%dT%H%M%S")
        return f"{left_name}_productivity_report_{timestamp}.json"
