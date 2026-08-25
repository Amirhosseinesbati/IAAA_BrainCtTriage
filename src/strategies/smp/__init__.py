"""
smp/strategy.py — Segmentation Models PyTorch (SMP) ICH strategy.

Provides a flexible 2D segmentation approach using the SMP library
with encoder backbones from timm, trained with PyTorch Lightning.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from src.strategies.base import ICHStrategy
from src.strategies.config_models import SMPConfig
from src.strategies.registry import StrategyRegistry


class SMPStrategy(ICHStrategy):
    """
    Segmentation Models PyTorch strategy for ICH segmentation.

    Uses SMP's extensive model zoo (U-Net, U-Net++, DeepLabV3+, FPN,
    PAN, LinkNet, MAnet, PSPNet) with pretrained encoders (ResNet,
    EfficientNet, MobileNet, etc.) via timm. Training is managed
    with PyTorch Lightning and logged to MLflow.
    """

    name: ClassVar[str] = "smp"
    display_name: ClassVar[str] = "🔬 SMP (U-Net / DeepLabV3+ / FPN)"
    description: ClassVar[str] = (
        "Flexible segmentation using SMP (Segmentation Models PyTorch). "
        "Supports 8 architectures (U-Net, U-Net++, DeepLabV3+, FPN, PAN, "
        "LinkNet, MAnet, PSPNet) with 14+ pretrained encoders from timm. "
        "Trained with PyTorch Lightning + MLflow."
    )

    # ── Config ────────────────────────────────────────────────────

    def get_config_class(self):
        return SMPConfig

    # ── Data preparation ──────────────────────────────────────────

    def prepare_data(self, config: SMPConfig | None = None) -> bool:
        """
        Prepare NIfTI data using the generic (strategy-agnostic) builder.

        The data is stored under *ICH_NIFTI_DIR* and is **not** tied to
        nnU-Net naming conventions.  See :class:`NiftiDatasetBuilder`.
        """
        from src.preprocessing.builders.nifti_builder import NiftiDatasetBuilder

        print("=== [SMP] Preparing Data (NIfTI via NiftiDatasetBuilder) ===")
        builder = NiftiDatasetBuilder()
        builder.build()
        print("=== [SMP] Data preparation complete ===")
        return True

    # ── Training ──────────────────────────────────────────────────

    def train(self, config: SMPConfig) -> bool:
        from src.strategies.smp.train import train_smp

        print(f"=== [SMP] Starting training | arch={config.architecture} "
              f"| encoder={config.encoder} | epochs={config.epochs} ===")

        train_smp(config)
        return True

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, study_dir: str) -> dict:
        """
        Run SMP inference on a single DICOM study.

        Note: Full SMP inference is integrated in submission/model.py.
        This is a placeholder for the strategy interface.
        """
        raise NotImplementedError(
            "SMP inference is available via submission/model.py. "
            "Use load_models('smp') and predict() from the submission package."
        )


# ── Auto-register ──────────────────────────────────────────────────
StrategyRegistry.register(SMPStrategy())
