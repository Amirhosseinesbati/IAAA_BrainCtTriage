"""Evaluate a YOLO fracture detector with inference-matched study pooling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from src.fracture.pooling import aggregate_study_scores, compute_study_features


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _predict_slices(
    model: YOLO,
    dataset_root: Path,
    manifest: pd.DataFrame,
    *,
    device: str,
    image_size: int,
    batch_size: int,
    confidence: float,
) -> tuple[np.ndarray, float]:
    scores: list[float] = []
    started = time.perf_counter()
    image_paths = [str(dataset_root / relative) for relative in manifest["image"]]
    for start in range(0, len(image_paths), batch_size):
        results = model.predict(
            source=image_paths[start : start + batch_size],
            imgsz=image_size,
            conf=confidence,
            device=device,
            batch=batch_size,
            verbose=False,
        )
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                scores.append(0.0)
            else:
                scores.append(float(result.boxes.conf.max().detach().cpu()))
    return np.asarray(scores, dtype=np.float64), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence", type=float, default=0.001)
    args = parser.parse_args()

    manifest = pd.read_csv(args.dataset / "manifest.csv")
    validation = manifest.loc[manifest["split"] == "val"].copy()
    validation = validation.sort_values(["study_id", "slice_index"]).reset_index(drop=True)
    if validation.empty:
        raise ValueError("Validation manifest is empty")
    model = YOLO(str(args.checkpoint))
    validation["slice_score"], runtime = _predict_slices(
        model,
        args.dataset,
        validation,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        confidence=args.confidence,
    )

    study_rows: list[dict[str, float | int | str]] = []
    for study_id, group in validation.groupby("study_id", sort=True):
        ordered = group.sort_values("slice_index")
        scores = ordered["slice_score"].to_numpy(float)
        row: dict[str, float | int | str] = {
            "study_id": str(study_id),
            "truth": int(ordered["study_fracture"].max()),
        }
        row.update(compute_study_features(scores))
        row.update({f"prob_{key}": value for key, value in aggregate_study_scores(scores).items()})
        study_rows.append(row)
    studies = pd.DataFrame.from_records(study_rows)
    truth = studies["truth"].to_numpy(bool)

    metrics: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "n_studies": int(len(studies)),
        "n_positive": int(truth.sum()),
        "n_slices": int(len(validation)),
        "runtime_s": runtime,
        "runtime_per_study_s": runtime / len(studies),
        "confidence_floor": args.confidence,
        "pooling": {},
    }
    for column in [name for name in studies if name.startswith("prob_")]:
        probabilities = studies[column].to_numpy(float)
        metrics["pooling"][column.removeprefix("prob_")] = {
            "auc": float(roc_auc_score(truth.astype(int), probabilities)),
            "threshold_0_5": _binary_metrics(truth, probabilities >= 0.5),
            "negative_mean": float(probabilities[~truth].mean()),
            "positive_mean": float(probabilities[truth].mean()),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output / "slice_predictions.csv", index=False)
    studies.to_csv(args.output / "study_predictions.csv", index=False)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
