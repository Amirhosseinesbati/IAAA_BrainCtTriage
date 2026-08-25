"""
predict.py — Inference pipeline for MLS heatmap regression.

Given a DICOM study directory, this module:
1. Loads all slices via BrainDicomReader
2. Runs the SliceSelector to pick top-K candidate slices
3. Runs the HRNet heatmap model on those K slices (batched)
4. Decodes keypoints via DARK sub-pixel refinement
5. Computes MLS_mm for each slice
6. Aggregates across slices (max or p90) for robust final MLS
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import MLS_CHECKPOINTS_DIR
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.training.mls_models import SliceSelectorModel
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.utils import (
    decode_heatmap_dark_batch,
    compute_mls_from_keypoints,
)
from src.strategies.config_models import MLSHeatmapConfig

logger = logging.getLogger(__name__)


def _resolve_checkpoint_paths(
    slice_selector_path: Optional[str] = None,
    heatmap_model_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve the SliceSelector and heatmap model checkpoint paths.

    Resolution priority:
        1. Explicitly provided paths (function arguments)
        2. Environment variables ``MLS_SLICE_SELECTOR_PATH`` and
           ``MLS_HEATMAP_MODEL_PATH``
        3. Project defaults under ``models/checkpoints/``

    Args:
        slice_selector_path: Optional path to the SliceSelector checkpoint.
        heatmap_model_path: Optional path to the heatmap model checkpoint.

    Returns:
        (slice_selector_path, heatmap_model_path) — resolved absolute paths.

    Raises:
        FileNotFoundError: If a resolved path does not exist on disk, with
            a message explaining how to supply the missing checkpoints.
    """
    if slice_selector_path is None:
        slice_selector_path = os.environ.get("MLS_SLICE_SELECTOR_PATH")
    if heatmap_model_path is None:
        heatmap_model_path = os.environ.get("MLS_HEATMAP_MODEL_PATH")

    if slice_selector_path is None:
        slice_selector_path = str(MLS_CHECKPOINTS_DIR / "slice_selector_best.ckpt")
    if heatmap_model_path is None:
        heatmap_model_path = str(MLS_CHECKPOINTS_DIR / "mls_heatmap" / "mls_heatmap_best.pth")

    missing = [
        p for p in (slice_selector_path, heatmap_model_path)
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing MLS model checkpoint(s):\n  "
            + "\n  ".join(missing)
            + "\nTrain the models first, or set the MLS_SLICE_SELECTOR_PATH "
              "and MLS_HEATMAP_MODEL_PATH environment variables."
        )
    return slice_selector_path, heatmap_model_path


def _load_slice_selector(
    checkpoint_path: str,
    device: torch.device,
) -> SliceSelectorModel:
    """
    Load the pre-trained SliceSelector model (ResNet18).

    Args:
        checkpoint_path: Path to the .ckpt or .pth checkpoint.
        device: Torch device.

    Returns:
        Loaded SliceSelectorModel in eval mode.
    """
    model = SliceSelectorModel().to(device).eval()

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle Lightning checkpoint (wraps in 'state_dict' key, 'model.' prefix)
    if "state_dict" in state_dict:
        sd = state_dict["state_dict"]
    else:
        sd = state_dict

    # Strip 'model.' prefix if present
    cleaned = {}
    for key, value in sd.items():
        if key.startswith("model."):
            cleaned[key[6:]] = value
        else:
            cleaned[key] = value

    model.load_state_dict(cleaned, strict=False)
    return model


def _load_heatmap_model(
    checkpoint_path: str,
    config: MLSHeatmapConfig,
    device: torch.device,
) -> HRNetHeatmapModel:
    """
    Load the trained HRNet heatmap model.

    Args:
        checkpoint_path: Path to the .pth checkpoint (from train.py).
        config: MLSHeatmapConfig used during training.
        device: Torch device.

    Returns:
        Loaded HRNetHeatmapModel in eval mode.
    """
    model = HRNetHeatmapModel(
        backbone_name=config.backbone,
        in_channels=config.input_channels,
        num_keypoints=3,
        pretrained=False,  # Don't load pretrained — we load trained weights
        head_dropout=getattr(config, "head_dropout", 0.0),  # identity in eval mode
    ).to(device).eval()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
    else:
        sd = checkpoint

    model.load_state_dict(sd, strict=False)
    return model


def _create_windowed_input(hu_image: np.ndarray, input_channels: int = 3) -> np.ndarray:
    """
    Create the windowed input tensor for the heatmap model.

    Args:
        hu_image: 2D numpy array of Hounsfield Units.
        input_channels: 3 (brain + subdural + bone windows) or 1 (brain only).

    Returns:
        (C, H, W) numpy array normalized to [0, 1] where C = input_channels.
    """
    from src.preprocessing.core.dicom_reader import BrainDicomReader

    ch1 = BrainDicomReader.apply_windowing(hu_image, "brain")
    if input_channels == 1:
        return ch1[None, ...]  # (1, H, W)

    ch2 = BrainDicomReader.apply_windowing(hu_image, "subdural")
    ch3 = BrainDicomReader.apply_windowing(hu_image, "bone")
    return np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)


def _create_3channel_window(hu_image: np.ndarray) -> np.ndarray:
    """
    Create 3-channel windowed representation (brain + subdural + bone).

    Deprecated alias kept for backward compatibility — new code should use
    :func:`_create_windowed_input` with an explicit ``input_channels``.

    Args:
        hu_image: 2D numpy array of Hounsfield Units.

    Returns:
        (3, H, W) numpy array normalized to [0, 1].
    """
    return _create_windowed_input(hu_image, input_channels=3)


def predict_mls(
    study_dir: str,
    slice_selector_path: Optional[str] = None,
    heatmap_model_path: Optional[str] = None,
    config: Optional[MLSHeatmapConfig] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Predict Midline Shift (MLS) for a single DICOM study.

    Args:
        study_dir: Path to directory containing DICOM .dcm files.
        slice_selector_path: Path to SliceSelector checkpoint. If None, it is
            resolved from the ``MLS_SLICE_SELECTOR_PATH`` environment variable
            or the project default under ``models/checkpoints/``.
        heatmap_model_path: Path to HRNet heatmap model checkpoint. If None,
            it is resolved from the ``MLS_HEATMAP_MODEL_PATH`` environment
            variable or the project default under ``models/checkpoints/``.
        config: MLSHeatmapConfig. Uses defaults if None.
        device: Torch device. Auto-detects if None.

    Returns:
        MLS value in millimeters (float).
    """
    if config is None:
        config = MLSHeatmapConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint paths (explicit → env vars → config defaults)
    slice_selector_path, heatmap_model_path = _resolve_checkpoint_paths(
        slice_selector_path, heatmap_model_path
    )

    # ── 1. Load DICOM ────────────────────────────────────────────
    logger.info(f"Loading DICOM from {study_dir}")
    reader = BrainDicomReader(study_dir).load_and_sort()
    image_hu = reader.get_3d_volume_hu()  # (H, W, D)
    spacing_x = reader.metadata["spacing_x"]
    n_slices = image_hu.shape[2]

    # ── 2. Load models ───────────────────────────────────────────
    logger.info("Loading SliceSelector...")
    selector = _load_slice_selector(slice_selector_path, device)

    logger.info(f"Loading HRNet heatmap model ({config.backbone})...")
    heatmap_model = _load_heatmap_model(heatmap_model_path, config, device)

    # Infer heatmap size from model output (handles any backbone output stride)
    with torch.no_grad():
        dummy = torch.zeros(1, config.input_channels, config.image_size, config.image_size, device=device)
        dummy_out = heatmap_model(dummy)
        heatmap_size = dummy_out.shape[-1]  # spatial size of heatmap
    logger.info(f"Model heatmap size: {heatmap_size}×{heatmap_size}")

    # ── 3-8. Run the full pipeline (slice selection → top-K → DARK → MLS) ──
    return _run_pipeline(
        selector, heatmap_model, image_hu, spacing_x, config, device, heatmap_size
    )


def _run_pipeline(
    selector: SliceSelectorModel,
    heatmap_model: HRNetHeatmapModel,
    image_hu: np.ndarray,
    spacing_x: float,
    config: MLSHeatmapConfig,
    device: torch.device,
    heatmap_size: int,
) -> float:
    """
    Run slice selection + top-K heatmap inference on a HU volume.

    Shared by :func:`predict_mls` (loads models per call) and
    :class:`MLSHeatmapPredictor` (reuses cached models) so the pipeline
    logic lives in exactly one place.

    Args:
        selector: Loaded SliceSelector model (eval mode).
        heatmap_model: Loaded HRNet heatmap model (eval mode).
        image_hu: 3D HU volume (H, W, D).
        spacing_x: DICOM pixel spacing in mm/px.
        config: MLSHeatmapConfig.
        device: Torch device.
        heatmap_size: Spatial size of the predicted heatmap (square).

    Returns:
        Aggregated MLS value in mm.
    """
    n_slices = image_hu.shape[2]

    # ── 1. Run SliceSelector on all slices ───────────────────────
    logger.info(f"Running SliceSelector on {n_slices} slices...")
    slice_logits = []

    # Process in mini-batches for efficiency
    batch_size = 32
    for start_idx in range(0, n_slices, batch_size):
        end_idx = min(start_idx + batch_size, n_slices)
        batch_slices = []

        for z in range(start_idx, end_idx):
            windowed = _create_windowed_input(image_hu[:, :, z], config.input_channels)  # (C, 512, 512)
            slice_tensor = torch.from_numpy(windowed).float().unsqueeze(0)  # (1, C, 512, 512)
            # Resize to 256x256 for selector
            resized = F.interpolate(slice_tensor, size=(256, 256), mode="bilinear",
                                    align_corners=False)
            batch_slices.append(resized)

        batch_tensor = torch.cat(batch_slices, dim=0).to(device)
        with torch.no_grad():
            logits = selector(batch_tensor)
        slice_logits.append(logits.cpu())

    slice_logits = torch.cat(slice_logits).squeeze()  # (N,)

    # ── 2. Select top-K slices ───────────────────────────────────
    top_k = min(config.top_k_slices, n_slices)
    top_indices = torch.topk(slice_logits, k=top_k).indices.numpy()

    logger.info(f"Top-{top_k} slice indices: {top_indices}")

    # ── 3. Run heatmap model on top-K slices (batched) ───────────
    batch_images = []
    for z in top_indices:
        windowed = _create_windowed_input(image_hu[:, :, z], config.input_channels)  # (C, H, W)
        batch_images.append(windowed)

    batch_tensor = torch.from_numpy(np.stack(batch_images, axis=0)).float().to(device)

    with torch.no_grad():
        heatmap_pred = heatmap_model(batch_tensor)  # (K, 3, H/4, W/4)

    # ── 4. Decode keypoints via DARK ─────────────────────────────
    coords_pred, scores = decode_heatmap_dark_batch(
        heatmap_pred.cpu(), heatmap_size, config.image_size
    )
    # coords_pred: (K, 3, 2) — (x, y) for each keypoint on each slice

    # ── 5. Compute MLS for each slice ────────────────────────────
    mls_values = []
    for k in range(len(coords_pred)):
        kps = coords_pred[k]
        # Check that all 3 keypoints were detected (not -1)
        if (kps[:, 0] >= 0).all():
            mls = compute_mls_from_keypoints(kps, spacing_x)
            mls_values.append(mls)

    if not mls_values:
        logger.warning("No valid MLS measurements from any top-K slice. Returning 0.0.")
        return 0.0

    # ── 6. Aggregate ─────────────────────────────────────────────
    mls_array = np.array(mls_values)

    if config.aggregation == "max":
        final_mls = float(mls_array.max())
    elif config.aggregation == "p90":
        final_mls = float(np.percentile(mls_array, 90))
    else:
        final_mls = float(mls_array.max())

    logger.info(
        f"MLS results (top-{top_k} slices): {np.round(mls_array, 2)} mm, "
        f"aggregated ({config.aggregation}): {final_mls:.2f} mm"
    )

    return final_mls


def batch_predict_mls(
    study_dirs: list[str],
    slice_selector_path: Optional[str] = None,
    heatmap_model_path: Optional[str] = None,
    config: Optional[MLSHeatmapConfig] = None,
    device: Optional[torch.device] = None,
) -> list[float]:
    """
    Run predict_mls on multiple studies.

    Args:
        study_dirs: List of DICOM study directory paths.
        slice_selector_path, heatmap_model_path, config, device:
            See predict_mls.

    Returns:
        List of MLS_mm values, one per study.
    """
    results = []
    for study_dir in study_dirs:
        mls = predict_mls(study_dir, slice_selector_path, heatmap_model_path, config, device)
        results.append(mls)
    return results


class MLSHeatmapPredictor:
    """
    Inference wrapper for the HRNet heatmap MLS model.

    Loads the SliceSelector + heatmap models **once** in the constructor
    (suitable for caching with e.g. Streamlit's ``@st.cache_resource``) and
    exposes a ``predict(reader)`` duck-typed interface identical to the
    legacy :class:`MLSPredictor`, so it can drop into
    ``load_all_models()`` / ``predict_for_ui()`` without changing the UI.

    Usage::

        predictor = MLSHeatmapPredictor(device="cuda")
        mls_mm = predictor.predict(reader)   # reader = BrainDicomReader
    """

    def __init__(
        self,
        slice_selector_path: Optional[str] = None,
        heatmap_model_path: Optional[str] = None,
        config: Optional[MLSHeatmapConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            slice_selector_path: Path to the SliceSelector checkpoint.
                Resolved via _resolve_checkpoint_paths if None.
            heatmap_model_path: Path to the heatmap model checkpoint.
                Resolved via _resolve_checkpoint_paths if None.
            config: MLSHeatmapConfig. Uses defaults if None.
            device: Torch device. Auto-detects if None.
        """
        self.slice_selector_path, self.heatmap_model_path = _resolve_checkpoint_paths(
            slice_selector_path, heatmap_model_path
        )
        self.config = config or MLSHeatmapConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load both models once (eval mode)
        self.selector = _load_slice_selector(self.slice_selector_path, self.device)
        self.heatmap_model = _load_heatmap_model(
            self.heatmap_model_path, self.config, self.device
        )

        # Infer the heatmap spatial size from the model output
        with torch.no_grad():
            dummy = torch.zeros(
                1, self.config.input_channels,
                self.config.image_size, self.config.image_size,
                device=self.device,
            )
            self.heatmap_size = self.heatmap_model(dummy).shape[-1]
        logger.info(
            f"MLSHeatmapPredictor ready: backbone={self.config.backbone}, "
            f"heatmap={self.heatmap_size}×{self.heatmap_size}"
        )

    @torch.no_grad()
    def predict(self, reader) -> float:
        """
        Predict MLS (mm) from a loaded BrainDicomReader.

        Args:
            reader: A BrainDicomReader instance (already load_and_sort()-ed).

        Returns:
            MLS value in millimeters.
        """
        image_hu = reader.get_3d_volume_hu()          # (H, W, D)
        spacing_x = reader.metadata["spacing_x"]
        return _run_pipeline(
            self.selector, self.heatmap_model, image_hu,
            spacing_x, self.config, self.device, self.heatmap_size,
        )
