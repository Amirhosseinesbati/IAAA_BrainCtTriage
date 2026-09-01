"""
_strategy.py — MLSHeatmapStrategy class definition.

Uses lazy imports inside each method so auto-registration succeeds
even when sub-modules (train, predict, etc.) are being developed.
"""

from __future__ import annotations

from typing import ClassVar

from src.strategies.mls_base import MLSStrategy
from src.strategies.mls_registry import MLSStrategyRegistry
from src.strategies.config_models import MLSHeatmapConfig


class MLSHeatmapStrategy(MLSStrategy):
    """
    HRNet heatmap-based keypoint regression strategy for MLS estimation.

    Uses an HRNet backbone (via timm) that outputs 3 Gaussian heatmap
    channels (one per keypoint: AnteriorFalxAttachment, PosteriorFalxAttachment,
    OutermostPointOfTheFalx) at 1/4 input resolution. Keypoints are decoded
    with DARK sub-pixel refinement. MLS is computed as the perpendicular
    distance from the outermost falx point to the ideal falx line.
    """

    name: ClassVar[str] = "mls_heatmap"
    display_name: ClassVar[str] = "🧠 HRNet Heatmap (MLS Regression)"
    description: ClassVar[str] = (
        "Heatmap-based keypoint regression for Midline Shift (MLS) estimation. "
        "Uses an HRNet-W32/W18 backbone to predict 3 Gaussian heatmaps "
        "(AnteriorFalxAttachment, PosteriorFalxAttachment, OutermostPointOfTheFalx) "
        "at 1/4 resolution. Sub-pixel keypoint decoding via DARK (Distribution-Aware "
        "coordinate Representation). Top-K slice aggregation for robust MLS estimation. "
        "Pure PyTorch training with MLflow logging."
    )

    # ── Config ────────────────────────────────────────────────────

    def get_config_class(self):
        return MLSHeatmapConfig

    # ── Data preparation ──────────────────────────────────────────

    def prepare_data(self, config: MLSHeatmapConfig | None = None) -> bool:
        """
        Prepare MLS dataset using the existing MlsDatasetBuilder.

        Reuses the already-built PNG + CSV output from the legacy builder
        so no data duplication is needed.
        """
        from pathlib import Path
        from src.config import MLS_DIR

        if config is not None and config.dataset_variant == "multitask_v2":
            output = Path(MLS_DIR).parent / "mls_multitask_v2" / "mls_labels_multitask.csv"
            if output.exists():
                print(f"=== [MLS Multitask v2] Data already exists at {output.parent} ===")
                return True
            print("=== [MLS Multitask v2] Building explicit-negative dataset ===")
            from scripts.build_mls_multitask_dataset import main as build_multitask_dataset
            build_multitask_dataset()
            return True

        mls_dir = Path(MLS_DIR)
        csv_path = mls_dir / "mls_labels.csv"
        img_dir = mls_dir / "images"

        if csv_path.exists() and img_dir.exists():
            print(f"=== [MLS Heatmap] Data already exists at {mls_dir} ===")
            return True

        print("=== [MLS Heatmap] Building dataset via MlsDatasetBuilder ===")
        from src.preprocessing.builders.mls_builder import MlsDatasetBuilder

        builder = MlsDatasetBuilder()
        builder.build()
        print("=== [MLS Heatmap] Data preparation complete ===")
        return True

    # ── Training ──────────────────────────────────────────────────

    def train(self, config: MLSHeatmapConfig) -> bool:
        print(f"=== [MLS Heatmap] Starting training | backbone={config.backbone} "
              f"| epochs={config.epochs} ===")
        if config.dataset_variant == "multitask_v2":
            from src.strategies.mls_heatmap.train_multitask import train_mls_multitask
            train_mls_multitask(config)
        else:
            from src.strategies.mls_heatmap.train import train_mls_heatmap
            train_mls_heatmap(config)
        return True

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, study_dir: str) -> float:
        from src.strategies.mls_heatmap.predict import predict_mls

        return predict_mls(study_dir)


# ── Auto-register ──────────────────────────────────────────────────
MLSStrategyRegistry.register(MLSHeatmapStrategy())
