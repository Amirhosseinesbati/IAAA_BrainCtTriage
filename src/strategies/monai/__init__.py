"""
monai/strategy.py — MONAI-based 3D ICH segmentation strategy.

Leverages MONAI's medical imaging networks (UNETR, SwinUNETR,
SegResNet, DynUNet) for 3D volumetric segmentation of ICH subtypes.
"""

from __future__ import annotations

from typing import ClassVar

from src.strategies.base import ICHStrategy
from src.strategies.config_models import MONAIConfig
from src.strategies.registry import StrategyRegistry


class MONAIStrategy(ICHStrategy):
    """
    MONAI 3D segmentation strategy for ICH.

    Uses MONAI's state-of-the-art medical imaging networks with 3D
    patch-based training. Supports UNETR (ViT encoder), SwinUNETR
    (Swin Transformer), SegResNet, and DynUNet architectures.
    """

    name: ClassVar[str] = "monai"
    display_name: ClassVar[str] = "🏥 MONAI (UNETR / SwinUNETR / SegResNet)"
    description: ClassVar[str] = (
        "3D volumetric segmentation using MONAI's medical imaging networks. "
        "Supports UNETR (ViT-based), SwinUNETR (Swin Transformer), SegResNet, "
        "and DynUNet. Patch-based training with 3D augmentations."
    )

    # ── Config ────────────────────────────────────────────────────

    def get_config_class(self):
        return MONAIConfig

    # ── Data preparation ──────────────────────────────────────────

    def prepare_data(self, config: MONAIConfig | None = None) -> bool:
        """
        Prepare NIfTI data using the generic (strategy-agnostic) builder.

        The data is stored under *ICH_NIFTI_DIR* and is **not** tied to
        nnU-Net naming conventions.  See :class:`NiftiDatasetBuilder`.
        """
        from src.preprocessing.builders.nifti_builder import NiftiDatasetBuilder

        print("=== [MONAI] Preparing Data (NIfTI via NiftiDatasetBuilder) ===")
        builder = NiftiDatasetBuilder()
        builder.build()
        print("=== [MONAI] Data preparation complete ===")
        return True

    # ── Training ──────────────────────────────────────────────────

    def train(self, config: MONAIConfig) -> bool:
        from src.strategies.monai.train import train_monai

        print(f"=== [MONAI] Starting training | model={config.model} "
              f"| epochs={config.epochs} ===")

        train_monai(config)
        return True

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, study_dir: str) -> dict:
        raise NotImplementedError(
            "MONAI inference is available via submission/model.py. "
            "Use load_models('monai') and predict() from the submission package."
        )


# ── Auto-register ──────────────────────────────────────────────────
StrategyRegistry.register(MONAIStrategy())
