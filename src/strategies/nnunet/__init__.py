"""
nnunet/strategy.py — nnU-Net v2 ICH segmentation strategy.

Wraps the existing NNUnetDatasetBuilder and train_nnunet_pipeline()
under the uniform ICHStrategy interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch

from src.strategies.base import ICHStrategy
from src.strategies.config_models import NNUNetConfig
from src.strategies.registry import StrategyRegistry


class NNUNetStrategy(ICHStrategy):
    """
    nnU-Net v2 strategy for 5-class ICH segmentation.

    Uses the self-configuring nnU-Net framework. Data is prepared as
    NIfTI volumes in nnUNet_raw format, and training is handled by
    nnUNetv2_plan_and_preprocess + nnUNetv2_train subprocesses with
    real-time MLflow monitoring.
    """

    name: ClassVar[str] = "nnunet"
    display_name: ClassVar[str] = "🧠 nnU-Net v2 (ICH Segmentation)"
    description: ClassVar[str] = (
        "Self-configuring nnU-Net v2 framework. Proven state-of-the-art "
        "for medical image segmentation. Supports 2D, 3D_fullres, and "
        "3D_lowres configurations with automatic hyper-parameter tuning."
    )

    # ── Config ────────────────────────────────────────────────────

    def get_config_class(self):
        return NNUNetConfig

    # ── Data preparation ──────────────────────────────────────────

    def prepare_data(self) -> bool:
        from src.preprocessing.builders.nnunet_builder import NNUnetDatasetBuilder

        print("=== [nnU-Net] Preparing Data ===")
        builder = NNUnetDatasetBuilder()
        builder.build()
        print("=== [nnU-Net] Data preparation complete ===")
        return True

    # ── Training ──────────────────────────────────────────────────

    def train(self, config: NNUNetConfig) -> bool:
        from src.training.train_nnunet import train_nnunet_pipeline

        print(f"=== [nnU-Net] Starting training | config={config.configuration} "
              f"| fold={config.fold} ===")

        train_nnunet_pipeline(
            dataset_id="501",
            fold=config.fold,
            configuration=config.configuration,
        )
        return True

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, study_dir: str) -> dict:
        """
        Run nnU-Net inference on a single DICOM study.

        Note: This is a thin wrapper for the strategy interface.
        For production inference, use submission/model.py which is
        self-contained and competition-ready.
        """
        import tempfile
        from pathlib import Path

        import nibabel as nib
        import numpy as np
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        from src.preprocessing.core.dicom_reader import BrainDicomReader

        # Load DICOM & export to NIfTI
        reader = BrainDicomReader(study_dir).load_and_sort()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pid = reader.metadata.get("patient_id", Path(study_dir).name)
            nifti_input = str(tmp_dir / f"{pid}_0000.nii.gz")
            reader.save_as_nifti(nifti_input)

            # Create a fresh predictor (avoids coupling with state)
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                verbose=False,
            )
            # Determine model folder from environment or config
            from src.config import NNUNET_RESULTS_DIR
            predictor.initialize_from_trained_model_folder(
                str(NNUNET_RESULTS_DIR),
                use_folds=(0,),
                checkpoint_name="checkpoint_best.pth",
            )

            nifti_output = str(tmp_dir / f"{pid}.nii.gz")
            predictor.predict_from_files(
                [[nifti_input]], [nifti_output],
                save_probabilities=False, overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            # Parse mask to volumes
            mask_nii = nib.load(nifti_output)
            mask_data = mask_nii.get_fdata()
            voxel_vol_ml = np.prod(mask_nii.header.get_zooms()) / 1000.0

            from src.config import ICH_LABELS
            volumes = {}
            for name, label_id in ICH_LABELS.items():
                if label_id == 0:
                    continue
                volumes[f"V_{name}"] = float(np.sum(mask_data == label_id) * voxel_vol_ml)

            # Ensure all 5 keys are present
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                volumes.setdefault(key, 0.0)

            return volumes


# ── Auto-register ──────────────────────────────────────────────────
StrategyRegistry.register(NNUNetStrategy())
