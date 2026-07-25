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

from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.training.mls_models import SliceSelectorModel
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.utils import (
    decode_heatmap_dark_batch,
    compute_mls_from_keypoints,
)
from src.strategies.config_models import MLSHeatmapConfig

logger = logging.getLogger(__name__)


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
    ).to(device).eval()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
    else:
        sd = checkpoint

    model.load_state_dict(sd, strict=False)
    return model


def _create_3channel_window(hu_image: np.ndarray) -> np.ndarray:
    """
    Create 3-channel windowed representation (brain + subdural + bone).

    Args:
        hu_image: 2D numpy array of Hounsfield Units.

    Returns:
        (3, H, W) numpy array normalized to [0, 1].
    """
    from src.preprocessing.core.dicom_reader import BrainDicomReader

    ch1 = BrainDicomReader.apply_windowing(hu_image, "brain")
    ch2 = BrainDicomReader.apply_windowing(hu_image, "subdural")
    ch3 = BrainDicomReader.apply_windowing(hu_image, "bone")
    return np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)


def predict_mls(
    study_dir: str,
    slice_selector_path: str,
    heatmap_model_path: str,
    config: Optional[MLSHeatmapConfig] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Predict Midline Shift (MLS) for a single DICOM study.

    Args:
        study_dir: Path to directory containing DICOM .dcm files.
        slice_selector_path: Path to SliceSelector checkpoint.
        heatmap_model_path: Path to HRNet heatmap model checkpoint.
        config: MLSHeatmapConfig. Uses defaults if None.
        device: Torch device. Auto-detects if None.

    Returns:
        MLS value in millimeters (float).
    """
    if config is None:
        config = MLSHeatmapConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    heatmap_size = config.image_size // 4

    # ── 3. Run SliceSelector on all slices ───────────────────────
    logger.info(f"Running SliceSelector on {n_slices} slices...")
    slice_logits = []

    # Process in mini-batches for efficiency
    batch_size = 32
    for start_idx in range(0, n_slices, batch_size):
        end_idx = min(start_idx + batch_size, n_slices)
        batch_slices = []

        for z in range(start_idx, end_idx):
            slice_3ch = _create_3channel_window(image_hu[:, :, z])  # (3, 512, 512)
            slice_tensor = torch.from_numpy(slice_3ch).float().unsqueeze(0)  # (1, 3, 512, 512)
            # Resize to 256x256 for selector
            resized = F.interpolate(slice_tensor, size=(256, 256), mode="bilinear",
                                    align_corners=False)
            batch_slices.append(resized)

        batch_tensor = torch.cat(batch_slices, dim=0).to(device)
        with torch.no_grad():
            logits = selector(batch_tensor)
        slice_logits.append(logits.cpu())

    slice_logits = torch.cat(slice_logits).squeeze()  # (N,)

    # ── 4. Select top-K slices ──────────────────────────────────
    top_k = min(config.top_k_slices, n_slices)
    top_indices = torch.topk(slice_logits, k=top_k).indices.numpy()

    logger.info(f"Top-{top_k} slice indices: {top_indices}")

    # ── 5. Run heatmap model on top-K slices (batched) ──────────
    batch_images = []
    for z in top_indices:
        slice_3ch = _create_3channel_window(image_hu[:, :, z])  # (3, H, W)
        batch_images.append(slice_3ch)

    batch_tensor = torch.from_numpy(np.stack(batch_images, axis=0)).float().to(device)

    with torch.no_grad():
        heatmap_pred = heatmap_model(batch_tensor)  # (K, 3, H/4, W/4)

    # ── 6. Decode keypoints via DARK ────────────────────────────
    coords_pred, scores = decode_heatmap_dark_batch(
        heatmap_pred.cpu(), heatmap_size, config.image_size
    )
    # coords_pred: (K, 3, 2) — (x, y) for each keypoint on each slice

    # ── 7. Compute MLS for each slice ───────────────────────────
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

    # ── 8. Aggregate ────────────────────────────────────────────
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
    slice_selector_path: str,
    heatmap_model_path: str,
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
