"""
yolo_seg/strategy.py — YOLO Segmentation strategy for ICH.

Uses Ultralytics YOLO's segmentation models (YOLOv8-seg, etc.)
adapted for 5-class ICH slice-level segmentation.
"""

from __future__ import annotations

from typing import ClassVar

from src.strategies.base import ICHStrategy
from src.strategies.config_models import YOLOSegConfig
from src.strategies.registry import StrategyRegistry


class YOLOSegStrategy(ICHStrategy):
    """
    YOLO Segmentation strategy for ICH detection + segmentation.

    Uses Ultralytics YOLO's instance segmentation models trained
    on slice-level ICH masks. Fast inference and lightweight.
    """

    name: ClassVar[str] = "yolo_seg"
    display_name: ClassVar[str] = "🎯 YOLO Segmentation (v8/v9/v10/v11-seg)"
    description: ClassVar[str] = (
        "Ultralytics YOLO segmentation models adapted for ICH. Very fast "
        "training and inference. Supports n/s/m/l/x model scales. "
        "Best for rapid experimentation."
    )

    # ── Config ────────────────────────────────────────────────────

    def get_config_class(self):
        return YOLOSegConfig

    # ── Data preparation ──────────────────────────────────────────

    def prepare_data(self, config: YOLOSegConfig | None = None) -> bool:
        """
        Convert NIfTI segmentation data to YOLO instance segmentation
        format (images + per-instance polygon labels).
        """
        from src.strategies.yolo_seg.data_prep import prepare_yolo_seg_data

        print("=== [YOLO Seg] Preparing Data (NIfTI → YOLO format) ===")
        resolved = config or YOLOSegConfig()
        prepare_yolo_seg_data(
            fold=resolved.fold,
            use_competition_folds=resolved.use_competition_folds,
        )
        print("=== [YOLO Seg] Data preparation complete ===")
        return True

    # ── Training ──────────────────────────────────────────────────

    def train(self, config: YOLOSegConfig) -> bool:
        import os
        import mlflow
        from ultralytics import YOLO
        from src.config import config_section
        from src.mlops import context_from_environment, experiment_run, log_run_summary

        print(f"=== [YOLO Seg] Starting training | size={config.model_size} "
              f"| epochs={config.epochs} ===")

        run_name = f"yolov8{config.model_size}-seg_ep{config.epochs}"

        from ultralytics import settings as ultralytics_settings
        ultralytics_settings.update({"mlflow": False})

        context = context_from_environment("ich_yolo_seg", run_name, config.model_dump(), strategy="yolo_seg")
        with experiment_run(context):

            model = YOLO(f"yolov8{config.model_size}-seg.pt")

            # Path to YOLO-format dataset
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            data_yaml = (
                BASE_DIR / "Data" / "processed" / "yolo_ich_seg" /
                f"fold_{config.fold}" / "dataset.yaml"
            )

            model.train(
                data=str(data_yaml),
                epochs=config.epochs,
                imgsz=config.imgsz,
                batch=config.batch_size,
                lr0=config.lr,
                patience=config.patience,
                device=0,
                verbose=True,
            )

            # Upload best model
            best_pt = BASE_DIR / "runs" / "segment" / "train" / "weights" / "best.pt"
            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path=config_section("mlflow", "artifact_paths", "models"))

            log_run_summary({
                "task": "ich", "strategy": "yolo_seg",
                "best_checkpoint": str(best_pt) if best_pt.exists() else None,
            })

        print("=== [YOLO Seg] Training complete ===")
        return True

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, study_dir: str) -> dict:
        raise NotImplementedError(
            "YOLO Seg inference is available via submission/model.py."
        )


# ── Auto-register ──────────────────────────────────────────────────
from pathlib import Path
StrategyRegistry.register(YOLOSegStrategy())
