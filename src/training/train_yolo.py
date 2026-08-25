import os
from pathlib import Path
import mlflow
from ultralytics import YOLO, settings
from src.config import YOLO_DEFAULTS, config_section
from src.mlops import context_from_environment, experiment_run, log_run_summary

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

    dataset_yaml = str(BASE_DIR / "Data" / "processed" / "yolo_fracture" / "dataset.yaml")
    
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
        numeric_metrics = {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        log_run_summary({"task": "fracture", "strategy": "yolo", "save_dir": str(save_dir), "metrics": numeric_metrics})

    print(f"=== YOLO Training Completed! Results saved to: {results.save_dir} ===")
    print("Check MLflow UI. YOLO has automatically logged metrics, parameters, and the best model!")

if __name__ == "__main__":
    train_fracture_detector()
