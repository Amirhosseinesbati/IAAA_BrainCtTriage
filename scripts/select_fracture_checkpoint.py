"""Prepare and verify a study-selected fracture checkpoint for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_arrays(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), empty, empty
    return (
        boxes.xyxy.detach().cpu().numpy(),
        boxes.conf.detach().cpu().numpy(),
        boxes.cls.detach().cpu().numpy(),
    )


def _log_artifact_with_retry(path: Path, artifact_path: str, attempts: int = 8) -> None:
    import mlflow

    for attempt in range(1, attempts + 1):
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(5 * (2 ** (attempt - 1)), 30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-source",
        type=Path,
        required=True,
        help="A YOLO validation image directory or a text file containing image paths.",
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mlflow-run-id")
    parser.add_argument("--screening-dir", type=Path)
    parser.add_argument("--paired-dir", type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.validation_source.exists():
        raise FileNotFoundError(args.validation_source)
    if not args.metrics.is_file():
        raise FileNotFoundError(args.metrics)
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.output)
    strip_optimizer(str(args.output))

    if args.validation_source.is_dir():
        image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        images = [
            str(path)
            for path in sorted(args.validation_source.rglob("*"))
            if path.is_file() and path.suffix.lower() in image_extensions
        ][: args.samples]
    else:
        images = [
            line.strip()
            for line in args.validation_source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: args.samples]
    if not images:
        raise ValueError(f"No validation images found in {args.validation_source}")

    source_model = YOLO(str(args.source))
    output_model = YOLO(str(args.output))
    source_results = source_model.predict(
        images,
        imgsz=args.image_size,
        conf=args.confidence,
        device=args.device,
        verbose=False,
    )
    output_results = output_model.predict(
        images,
        imgsz=args.image_size,
        conf=args.confidence,
        device=args.device,
        verbose=False,
    )

    max_absolute_difference = 0.0
    total_detections = 0
    for source_result, output_result in zip(source_results, output_results, strict=True):
        source_arrays = _prediction_arrays(source_result)
        output_arrays = _prediction_arrays(output_result)
        for source_array, output_array in zip(source_arrays, output_arrays, strict=True):
            if source_array.shape != output_array.shape:
                raise RuntimeError(
                    "Prediction shape mismatch after stripping: "
                    f"{source_array.shape} != {output_array.shape}"
                )
            if source_array.size:
                max_absolute_difference = max(
                    max_absolute_difference,
                    float(np.max(np.abs(source_array - output_array))),
                )
        total_detections += int(source_arrays[1].size)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    payload = {
        "source": str(args.source),
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "size_bytes": args.output.stat().st_size,
        "verification": {
            "n_images": len(images),
            "n_detections": total_detections,
            "max_absolute_difference": max_absolute_difference,
            "confidence": args.confidence,
            "image_size": args.image_size,
        },
        "study_metrics": metrics,
    }
    if max_absolute_difference != 0.0:
        raise RuntimeError(
            f"Stripped checkpoint changed predictions (max diff={max_absolute_difference})"
        )

    manifest = args.manifest or args.output.with_suffix(".selection.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.mlflow_run_id:
        import mlflow

        pooling = metrics.get("pooling", {})
        selected_metrics = {
            f"selected_study_{method}_auc": float(values["auc"])
            for method, values in pooling.items()
            if isinstance(values, dict) and isinstance(values.get("auc"), (int, float))
        }
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_metrics(selected_metrics)
            mlflow.set_tags(
                {
                    "selected_checkpoint": str(args.source),
                    "selected_checkpoint_sha256": payload["sha256"],
                    "selection_basis": "periodic study-level checkpoint screening",
                    "study_selection_status": "provisional_independent_fold",
                }
            )
            _log_artifact_with_retry(args.output, "selected_model")
            _log_artifact_with_retry(manifest, "selected_model")
            for path in sorted(args.metrics.parent.iterdir()):
                if path.is_file():
                    _log_artifact_with_retry(path, "selected_study_evaluation")
            if args.screening_dir:
                for path in sorted(args.screening_dir.glob("*/metrics.json")):
                    _log_artifact_with_retry(
                        path,
                        f"checkpoint_screening/{path.parent.name}",
                    )
            if args.paired_dir:
                for path in sorted(args.paired_dir.glob("*.json")):
                    _log_artifact_with_retry(path, "paired_bootstrap")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
