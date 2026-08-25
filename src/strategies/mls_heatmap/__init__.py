"""
mls_heatmap — HRNet heatmap-based MLS (Midline Shift) regression strategy.

This strategy uses an HRNet backbone (via timm) with a lightweight heatmap
head to predict 3 Gaussian heatmaps (one per keypoint) at 1/4 input resolution.
Keypoint coordinates are extracted via DARK sub-pixel decoding, and MLS is
computed as the perpendicular distance from the outermost falx point to
the ideal falx line connecting the two attachment points.

Components:
    utils.py     — Gaussian heatmap generation, DARK decoding, MLS computation
    model.py     — HRNetHeatmapModel with configurable backbone
    dataset.py   — MLSHeatmapDataset with data augmentation
    train.py     — Pure PyTorch training loop with MLflow logging
    predict.py   — Inference: predict_mls() with Top-K slice aggregation
    __init__.py  — MLSHeatmapStrategy class + auto-registration
"""

from src.strategies.mls_heatmap._strategy import MLSHeatmapStrategy

__all__ = ["MLSHeatmapStrategy"]
