"""Prepare and verify a study-selected fracture checkpoint for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-list", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.validation_list.is_file():
        raise FileNotFoundError(args.validation_list)
    if not args.metrics.is_file():
        raise FileNotFoundError(args.metrics)
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.output)
    strip_optimizer(str(args.output))

    images = [
        line.strip()
        for line in args.validation_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.samples]
    if not images:
        raise ValueError(f"No validation images found in {args.validation_list}")

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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
