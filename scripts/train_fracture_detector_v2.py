"""Train and study-evaluate a fracture detector inside one MLflow run."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from ultralytics import YOLO, settings

from src.mlops import ExperimentContext, experiment_run, log_run_summary


def _finite_metrics(values: dict[str, object]) -> dict[str, float]:
    mapping = {
        "metrics/precision(B)": "box_precision",
        "metrics/recall(B)": "box_recall",
        "metrics/mAP50(B)": "box_map50",
        "metrics/mAP50-95(B)": "box_map50_95",
        "fitness": "box_fitness",
    }
    result: dict[str, float] = {}
    for key, name in mapping.items():
        value = values.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[name] = float(value)
    return result


def _study_metrics(payload: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    pooling = payload.get("pooling", {})
    if not isinstance(pooling, dict):
        return result
    for method, metrics in pooling.items():
        if not isinstance(metrics, dict):
            continue
        auc = metrics.get("auc")
        threshold = metrics.get("threshold_0_5", {})
        if isinstance(auc, (int, float)) and math.isfinite(float(auc)):
            result[f"study_{method}_auc"] = float(auc)
        if isinstance(threshold, dict):
            f1 = threshold.get("f1")
            if isinstance(f1, (int, float)) and math.isfinite(float(f1)):
                result[f"study_{method}_f1_at_0_5"] = float(f1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/fracture_v2"))
    parser.add_argument("--evaluation-confidence", type=float, default=0.001)
    args = parser.parse_args()

    dataset_yaml = args.dataset / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    config = {
        "dataset": str(args.dataset),
        "weights": str(args.weights),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "workers": args.workers,
        "device": args.device,
        "optimizer": "AdamW",
        "mosaic": 0.0,
        "mixup": 0.0,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.10,
        "fliplr": 0.5,
        "evaluation_confidence": args.evaluation_confidence,
    }
    context = ExperimentContext(
        task_key="fracture",
        run_name=args.run_name,
        run_config=config,
        strategy="yolo-study-aware-v2",
        tags={"fold": "0", "stage": "controlled-finetune"},
        notes=(
            "Controlled experiment: keep YOLOv8s/512 checkpoint and change only "
            "study coverage, negative sampling, validation coverage and study-level evaluation."
        ),
    )
    settings.update({"mlflow": False})
    model = YOLO(str(args.weights))

    with experiment_run(context):
        results = model.train(
            data=str(dataset_yaml),
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch_size,
            project=str(args.output_root),
            name=args.run_name,
            device=args.device,
            workers=args.workers,
            optimizer="AdamW",
            lr0=args.learning_rate,
            patience=args.patience,
            mosaic=0.0,
            mixup=0.0,
            degrees=5.0,
            translate=0.05,
            scale=0.10,
            fliplr=0.5,
            seed=42,
            deterministic=True,
            plots=True,
            save=True,
            save_period=5,
        )
        save_dir = Path(results.save_dir)
        weights_dir = save_dir / "weights"
        for name in ("best.pt", "last.pt"):
            candidate = weights_dir / name
            if candidate.is_file():
                mlflow.log_artifact(str(candidate), artifact_path="models")
        yolo_metrics = _finite_metrics(getattr(results, "results_dict", {}) or {})
        if yolo_metrics:
            mlflow.log_metrics(yolo_metrics)

        evaluation_dir = save_dir / "study_evaluation"
        evaluation_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_fracture_detector_v2.py"),
            "--checkpoint", str(weights_dir / "best.pt"),
            "--dataset", str(args.dataset),
            "--output", str(evaluation_dir),
            "--device", args.device,
            "--image-size", str(args.image_size),
            "--batch-size", str(args.batch_size),
            "--confidence", str(args.evaluation_confidence),
        ]
        subprocess.run(evaluation_command, cwd=PROJECT_ROOT, check=True)
        study_payload = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
        study_metrics = _study_metrics(study_payload)
        if study_metrics:
            mlflow.log_metrics(study_metrics)
        mlflow.log_artifacts(str(evaluation_dir), artifact_path="study_evaluation")
        for plot in save_dir.glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path="plots")
        log_run_summary({
            "task": "fracture",
            "strategy": "yolo-study-aware-v2",
            "save_dir": str(save_dir),
            "yolo_metrics": yolo_metrics,
            "study_metrics": study_metrics,
        })
        print(json.dumps({"yolo": yolo_metrics, "study": study_metrics}, indent=2))


if __name__ == "__main__":
    main()
