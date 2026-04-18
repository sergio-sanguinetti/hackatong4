from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from solution.application.analyzer_service import AnalyzeLoadingCycleUseCase
from solution.domain.models import AnalysisReport
from solution.infrastructure.imu_processor import ImuProcessor
from solution.infrastructure.video_processor import VideoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = FastAPI(title="Mining Productivity 2.0 API", version="0.1.0")


class BatchResponse(BaseModel):
    status: str
    processed_reports: int
    output_files: List[str]
    message: str


def _build_use_case() -> AnalyzeLoadingCycleUseCase:
    return AnalyzeLoadingCycleUseCase(video_processor=VideoProcessor(), imu_processor=ImuProcessor())


def discover_inputs(inputs_dir: Path) -> Tuple[List[Tuple[Path, Optional[Path]]], Optional[Path]]:
    mp4_files = sorted(inputs_dir.glob("*.mp4"))
    json_files = sorted(inputs_dir.glob("*.json"))
    csv_files = sorted(inputs_dir.glob("*.csv"))
    npy_files = sorted(inputs_dir.glob("*.npy"))

    json_imu_files = [p for p in json_files if "imu" in p.stem.lower()]
    csv_imu_files = [p for p in csv_files if "imu" in p.stem.lower()]
    npy_imu_files = [p for p in npy_files if "imu" in p.stem.lower()]

    # Prioridad: archivo explícito con "imu" en el nombre.
    if json_imu_files:
        imu_path: Optional[Path] = json_imu_files[0]
    elif csv_imu_files:
        imu_path = csv_imu_files[0]
    elif npy_imu_files:
        imu_path = npy_imu_files[0]
    else:
        # Fallback flexible: primer JSON/CSV/NPY disponible que no sea output/report.
        candidate_json = [p for p in json_files if "report" not in p.stem.lower() and "output" not in p.stem.lower()]
        candidate_csv = [p for p in csv_files if "report" not in p.stem.lower() and "output" not in p.stem.lower()]
        candidate_npy = [p for p in npy_files if "report" not in p.stem.lower() and "output" not in p.stem.lower()]
        imu_path = (
            candidate_json[0]
            if candidate_json
            else (candidate_csv[0] if candidate_csv else (candidate_npy[0] if candidate_npy else None))
        )

    left = [f for f in mp4_files if "left" in f.stem.lower()]
    right = [f for f in mp4_files if "right" in f.stem.lower()]
    pairs: List[Tuple[Path, Optional[Path]]] = []

    if left:
        for idx, left_video in enumerate(left):
            right_video = right[idx] if idx < len(right) else None
            pairs.append((left_video, right_video))
    elif mp4_files:
        # Fallback cuando no vienen etiquetados como left/right.
        pairs.append((mp4_files[0], mp4_files[1] if len(mp4_files) > 1 else None))

    return pairs, imu_path


def run_batch(inputs_dir: Path = INPUTS_DIR, outputs_dir: Path = OUTPUTS_DIR) -> BatchResponse:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pairs, imu_path = discover_inputs(inputs_dir)

    if not pairs:
        raise RuntimeError("No se encontraron videos .mp4 en /inputs.")
    if imu_path is None:
        raise RuntimeError("No se encontró telemetría IMU (JSON, CSV o NPY) en /inputs.")

    use_case = _build_use_case()
    generated_files: List[str] = []

    for left_video_path, right_video_path in pairs:
        report: AnalysisReport = use_case.execute(
            left_video_path=left_video_path,
            right_video_path=right_video_path,
            imu_path=imu_path,
        )
        output_file = outputs_dir / report.output_filename()
        output_file.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        generated_files.append(str(output_file))

    return BatchResponse(
        status="success",
        processed_reports=len(generated_files),
        output_files=generated_files,
        message=f"Batch completado en {datetime.utcnow().isoformat()}Z",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/batch/run", response_model=BatchResponse)
def batch_run() -> BatchResponse:
    try:
        return run_batch()
    except Exception as exc:  # noqa: BLE001 - exponer error para diagnóstico en hackathon.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/batch/discover")
def batch_discover() -> dict:
    pairs, imu_path = discover_inputs(INPUTS_DIR)
    return {
        "video_pairs": [{"left": str(left), "right": str(right) if right else None} for left, right in pairs],
        "imu_path": str(imu_path) if imu_path else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mining Productivity 2.0 - Batch + API")
    parser.add_argument("--batch", action="store_true", help="Ejecuta procesamiento batch y termina.")
    parser.add_argument("--host", default="0.0.0.0", help="Host FastAPI/uvicorn")
    parser.add_argument("--port", default=8000, type=int, help="Puerto FastAPI/uvicorn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch:
        response = run_batch()
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        return

    uvicorn.run("solution.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
