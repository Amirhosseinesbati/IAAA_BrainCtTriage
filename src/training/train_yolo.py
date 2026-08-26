import math
import numbers
import re
from pathlib import Path
import mlflow
from ultralytics import YOLO, settings
from src.config import YOLO_DEFAULTS, config_section
from src.mlops import context_from_environment, experiment_run, log_run_summary


_YOLO_METRIC_NAMES = {
    "metrics/precision(B)": "box_precision",
    "metrics/recall(B)": "box_recall",
    "metrics/mAP50(B)": "box_map50",
    "metrics/mAP50-95(B)": "box_map50_95",
    "fitness": "fitness",
}


def _mlflow_safe_yolo_metrics(metrics: dict) -> dict[str, float]:
    """Return finite YOLO metrics with DagsHub/MLflow-safe names."""
    cleaned: dict[str, float] = {}
    for raw_name, raw_value in metrics.items():
        if not isinstance(raw_value, numbers.Real):
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        name = _YOLO_METRIC_NAMES.get(str(raw_name))
        if name is None:
            name = re.sub(r"[^a-zA-Z0-9_.\-/ ]+", "_", str(raw_name)).strip(" _")
        if name:
            cleaned[name[:250]] = value
    return cleaned

def train_fracture_detector(config=None):
    print("=== Starting YOLO Fracture Detection Training ===")
    resolved = dict(YOLO_DEFAULTS)
    if config:
        resolved.update(config)
    # A single explicit MLflow run owns every artifact; disabling the built-in
    # callback prevents duplicate, partially-described runs.
    settings.update({'mlflow': False})
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CUSTOM_OUTPUT_DIR = BASE_DIR / "experiments" / "yolo_results"

    fold = int(resolved.get("fold", 0))
    dataset_yaml = str(
        BASE_DIR / "Data" / "processed" / "yolo_fracture" /
        f"fold_{fold}" / "dataset.yaml"
    )
    
    weights_dir = BASE_DIR / "models" / "pretrained"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights_dir / resolved["pretrained"])

    context = context_from_environment("fracture", "yolo_fracture", resolved, strategy="yolo")
    with experiment_run(context):
        results = model.train(
            data=dataset_yaml, epochs=resolved["epochs"], imgsz=resolved["image_size"],
            batch=resolved["batch_size"], project=str(CUSTOM_OUTPUT_DIR),
            name=context.run_name, device=0, workers=4,
            mosaic=0.0, mixup=0.0, degrees=10.0, translate=0.1, fliplr=0.5,
            optimizer=resolved["optimizer"], lr0=resolved["lr"], patience=resolved["patience"],
        )
        save_dir = Path(results.save_dir)
        for candidate in (save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"):
            if candidate.exists():
                mlflow.log_artifact(str(candidate), artifact_path=config_section("mlflow", "artifact_paths", "models"))
        for plot in save_dir.glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path=config_section("mlflow", "artifact_paths", "plots"))
        metrics = getattr(results, "results_dict", {}) or {}
        numeric_metrics = _mlflow_safe_yolo_metrics(metrics)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        log_run_summary({"task": "fracture", "strategy": "yolo", "save_dir": str(save_dir), "metrics": numeric_metrics})

    print(f"=== YOLO Training Completed! Results saved to: {results.save_dir} ===")
    print("Check MLflow UI. YOLO has automatically logged metrics, parameters, and the best model!")

if __name__ == "__main__":
    train_fracture_detector()
