"""
leaderboard — Personal leaderboard for IAAA 2026 Brain CT Triage Challenge.

Evaluates trained models (placed in submission/models/) against ground truth
from Data/metadata/training_df.csv using Quadratic Weighted Kappa (QWK),
the official competition metric.

Supports multiple ICH strategies:
  - ``nnunet``   : nnU-Net v2 (default)
  - ``smp``      : Segmentation Models PyTorch (U-Net, DeepLabV3+, FPN, ...)
  - ``monai``    : MONAI (UNETR, SwinUNETR, SegResNet, DynUNet)
  - ``yolo_seg`` : Ultralytics YOLO Segmentation

Usage:
    python -m leaderboard.evaluate
    python -m leaderboard.evaluate --ich-strategy smp
    python -m leaderboard.evaluate --ich-strategy monai --device cpu
    python -m leaderboard.evaluate --list-strategies
"""
