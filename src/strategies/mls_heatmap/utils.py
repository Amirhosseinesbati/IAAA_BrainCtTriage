"""
utils.py — Heatmap generation, DARK sub-pixel decoding, and MLS computation utilities.

This module provides the core algorithmic components for the heatmap-based
keypoint regression pipeline:

1. Gaussian heatmap generation with missing-keypoint masking
2. DARK (Distribution-Aware coordinate Representation) sub-pixel decoding
3. Soft-argmax decoding (fallback)
4. MLS computation from keypoint coordinates (pixels → mm)
5. MLS binning accuracy for triage-relevant validation
"""

from __future__ import annotations

from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════
# 1. Gaussian Heatmap Generation
# ═════════════════════════════════════════════════════════════════════════

def generate_gaussian_heatmap(
    keypoints: Sequence[Optional[Tuple[float, float]]],
    img_size: int,
    heatmap_size: int,
    sigma: float = 2.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 2D Gaussian heatmaps for a set of keypoints.

    For each keypoint that is not None, a 2D Gaussian is placed at the
    corresponding location (scaled to heatmap_size). For None keypoints,
    the heatmap channel is all zeros and the mask is 0.

    Args:
        keypoints: Sequence of 3 elements (one per keypoint).
            Each element is either (x, y) in **pixel coordinates** of the
            original image, or None if the keypoint is missing.
        img_size: Size of the input image (square) in pixels.
        heatmap_size: Size of the output heatmap (square) in pixels.
            Typically img_size // 4.
        sigma: Standard deviation of the Gaussian in **heatmap pixels**.
        dtype: Torch dtype for the output tensors.
        device: Torch device for the output tensors.

    Returns:
        heatmaps: Tensor of shape (K, heatmap_size, heatmap_size) where
            K = len(keypoints). Each channel is a Gaussian centered at the
            keypoint location, or all zeros if the keypoint is missing.
        mask: Tensor of shape (K,) where mask[i] = 1.0 if keypoint i is
            present, 0.0 if missing.
    """
    K = len(keypoints)
    scale = heatmap_size / img_size
    heatmaps = torch.zeros((K, heatmap_size, heatmap_size), dtype=dtype, device=device)
    mask = torch.zeros((K,), dtype=dtype, device=device)

    # Create coordinate grid once
    y_grid, x_grid = torch.meshgrid(
        torch.arange(heatmap_size, dtype=dtype, device=device),
        torch.arange(heatmap_size, dtype=dtype, device=device),
        indexing="ij",
    )  # both (H, W)

    for i, kp in enumerate(keypoints):
        if kp is None:
            continue  # heatmap stays zeros, mask stays 0

        x_px, y_px = float(kp[0]), float(kp[1])
        # Scale to heatmap coordinates
        x_c = x_px * scale
        y_c = y_px * scale

        # Compute squared distance from center
        dist_sq = (x_grid - x_c) ** 2 + (y_grid - y_c) ** 2

        # 2D Gaussian (unnormalized — peak = 1.0)
        heatmaps[i] = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        mask[i] = 1.0

    return heatmaps, mask


def generate_batch_heatmaps(
    keypoints_batch: Sequence[Sequence[Optional[Tuple[float, float]]]],
    img_size: int,
    heatmap_size: int,
    sigma: float = 2.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Gaussian heatmaps for a batch of keypoint sets.

    Args:
        keypoints_batch: List of B samples, each a sequence of K keypoints.
        img_size, heatmap_size, sigma: See generate_gaussian_heatmap.

    Returns:
        heatmaps: Tensor (B, K, H, W) of Gaussian heatmaps.
        masks: Tensor (B, K) of per-keypoint presence masks.
    """
    batch_heatmaps = []
    batch_masks = []
    for kps in keypoints_batch:
        hm, m = generate_gaussian_heatmap(kps, img_size, heatmap_size, sigma, dtype, device)
        batch_heatmaps.append(hm.unsqueeze(0))  # (1, K, H, W)
        batch_masks.append(m.unsqueeze(0))      # (1, K)
    return torch.cat(batch_heatmaps, dim=0), torch.cat(batch_masks, dim=0)


# ═════════════════════════════════════════════════════════════════════════
# 2. DARK (Distribution-Aware coordinate Representation) Decoding
# ═════════════════════════════════════════════════════════════════════════

def decode_heatmap_dark(
    heatmap: torch.Tensor,
    heatmap_size: int,
    img_size: int,
) -> Tuple[float, float]:
    """
    Decode a single heatmap channel to sub-pixel coordinates using DARK.

    DARK fits a 2D quadratic (Taylor expansion) around the maximum pixel
    to achieve sub-pixel precision. Reference:
        Zhang et al., "Distribution-Aware Coordinate Representation for
        Human Pose Estimation" (CVPR 2020).

    Args:
        heatmap: 2D tensor (H, W) of predicted heatmap values.
        heatmap_size: Size of the heatmap (square).
        img_size: Size of the original input image (square).

    Returns:
        (x, y) coordinates in **original image pixel space** with sub-pixel precision.
        Returns (-1, -1) if the heatmap is all zeros (no detection).
    """
    H, W = heatmap.shape
    scale = img_size / heatmap_size

    # Find maximum pixel location
    max_val = heatmap.max()
    if max_val < 1e-8:
        return (-1.0, -1.0)

    max_idx = heatmap.argmax()
    y0 = max_idx // W
    x0 = max_idx % W

    # Convert to int for indexing
    y0_i, x0_i = int(y0), int(x0)

    # If the maximum is at the border, fall back to integer coords
    if (y0_i == 0 or y0_i == H - 1 or x0_i == 0 or x0_i == W - 1):
        return (float(x0) * scale, float(y0) * scale)

    # Compute first derivatives (gradient) via central differences
    # g_x = ∂f/∂x, g_y = ∂f/∂y
    g_x = (heatmap[y0_i, x0_i + 1] - heatmap[y0_i, x0_i - 1]) / 2.0
    g_y = (heatmap[y0_i + 1, x0_i] - heatmap[y0_i - 1, x0_i]) / 2.0

    # Compute second derivatives (Hessian) via central differences
    # H_xx = ∂²f/∂x², H_yy = ∂²f/∂y², H_xy = ∂²f/∂x∂y
    H_xx = heatmap[y0_i, x0_i + 1] - 2.0 * heatmap[y0_i, x0_i] + heatmap[y0_i, x0_i - 1]
    H_yy = heatmap[y0_i + 1, x0_i] - 2.0 * heatmap[y0_i, x0_i] + heatmap[y0_i - 1, x0_i]
    H_xy = (heatmap[y0_i + 1, x0_i + 1] - heatmap[y0_i + 1, x0_i - 1]
            - heatmap[y0_i - 1, x0_i + 1] + heatmap[y0_i - 1, x0_i - 1]) / 4.0

    # Solve H * delta = -g => delta = -H⁻¹ * g
    det = H_xx * H_yy - H_xy * H_xy
    if abs(det) < 1e-12:
        # Degenerate Hessian — use integer coords
        return (float(x0) * scale, float(y0) * scale)

    delta_x = -(H_yy * g_x - H_xy * g_y) / det
    delta_y = -(H_xx * g_y - H_xy * g_x) / det

    # Clamp sub-pixel offset to [-0.5, 0.5] for numerical stability
    delta_x = max(-0.5, min(0.5, delta_x.item()))
    delta_y = max(-0.5, min(0.5, delta_y.item()))

    # Sub-pixel coordinate in heatmap space, then scale to image space
    x_sub = (float(x0) + delta_x) * scale
    y_sub = (float(y0) + delta_y) * scale

    # Clamp to image bounds
    x_sub = max(0.0, min(img_size - 1, x_sub))
    y_sub = max(0.0, min(img_size - 1, y_sub))

    return (x_sub, y_sub)


def decode_heatmap_dark_batch(
    heatmaps: torch.Tensor,
    heatmap_size: int,
    img_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a batch of heatmaps to sub-pixel coordinates using DARK.

    Args:
        heatmaps: Tensor (B, K, H, W) of predicted heatmaps.
        heatmap_size: Size of the heatmap (square).
        img_size: Size of the original input image (square).

    Returns:
        coords: numpy array (B, K, 2) of (x, y) coordinates in image pixels.
        scores: numpy array (B, K) of peak heatmap values (confidence).
    """
    B, K, H, W = heatmaps.shape
    coords = np.zeros((B, K, 2), dtype=np.float32)
    scores = np.zeros((B, K), dtype=np.float32)

    for b in range(B):
        for k in range(K):
            hm = heatmaps[b, k]
            scores[b, k] = hm.max().item()
            x, y = decode_heatmap_dark(hm, heatmap_size, img_size)
            coords[b, k, 0] = x
            coords[b, k, 1] = y

    return coords, scores


# ═════════════════════════════════════════════════════════════════════════
# 3. Soft-Argmax Decoding (fallback, less accurate than DARK)
# ═════════════════════════════════════════════════════════════════════════

def decode_soft_argmax(
    heatmap: torch.Tensor,
    heatmap_size: int,
    img_size: int,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    """
    Decode a heatmap to sub-pixel coordinates using soft-argmax.

    The heatmap is normalized with softmax (with temperature), then the
    expected coordinate is computed as the weighted sum over all pixels.

    Args:
        heatmap: 2D tensor (H, W) of predicted heatmap values.
        heatmap_size: Size of the heatmap (square).
        img_size: Size of the original input image (square).
        temperature: Softmax temperature. Lower = more peaked.

    Returns:
        (x, y) coordinates in original image pixel space.
    """
    H, W = heatmap.shape
    scale = img_size / heatmap_size

    # Flatten and apply softmax
    flat = heatmap.flatten()
    probs = F.softmax(flat / temperature, dim=0)

    # Coordinate grids
    y_grid = torch.arange(H, device=heatmap.device, dtype=torch.float32)
    x_grid = torch.arange(W, device=heatmap.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")

    # Expected coordinate (weighted mean)
    x_exp = (probs * xx.flatten()).sum().item()
    y_exp = (probs * yy.flatten()).sum().item()

    # Scale to image space
    return (x_exp * scale, y_exp * scale)


# ═════════════════════════════════════════════════════════════════════════
# 4. MLS Computation
# ═════════════════════════════════════════════════════════════════════════

def compute_mls_from_keypoints(
    keypoints_pixels: np.ndarray,
    spacing_x: float,
) -> float:
    """
    Compute Midline Shift (MLS) from three keypoint coordinates.

    MLS is defined as the perpendicular distance from the
    OutermostPointOfTheFalx (point 3) to the ideal falx line connecting
    the AnteriorFalxAttachment (point 1) and PosteriorFalxAttachment (point 2).

    Formula:
        mls_px = |(x₂-x₁)(y₁-y₃) - (x₁-x₃)(y₂-y₁)| / √[(x₂-x₁)² + (y₂-y₁)²]
        mls_mm = mls_px * spacing_x

    Args:
        keypoints_pixels: numpy array (3, 2) of (x, y) coordinates in pixels.
            Index 0: AnteriorFalxAttachment
            Index 1: PosteriorFalxAttachment
            Index 2: OutermostPointOfTheFalx
        spacing_x: Pixel spacing in the x-direction (mm/px) from DICOM metadata.

    Returns:
        MLS value in millimeters. Returns 0.0 if the falx line is degenerate
        (points 1 and 2 are coincident).
    """
    x1, y1 = keypoints_pixels[0]
    x2, y2 = keypoints_pixels[1]
    x3, y3 = keypoints_pixels[2]

    # Vector from point 1 to point 2 (the falx line direction)
    dx = x2 - x1
    dy = y2 - y1

    # Perpendicular distance from point 3 to the line through points 1 and 2
    denom = np.sqrt(dx ** 2 + dy ** 2)
    if denom < 1e-8:
        return 0.0  # Degenerate: attachment points are coincident

    numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    mls_px = numerator / denom

    # Convert to millimeters
    mls_mm = mls_px * spacing_x
    return mls_mm


def compute_mls_batch(
    keypoints_batch: np.ndarray,
    spacing_x: float,
) -> np.ndarray:
    """
    Compute MLS for a batch of keypoint sets.

    Args:
        keypoints_batch: numpy array (B, 3, 2) of keypoint coordinates.
        spacing_x: Pixel spacing in mm/px.

    Returns:
        numpy array (B,) of MLS values in mm.
    """
    x1 = keypoints_batch[:, 0, 0]
    y1 = keypoints_batch[:, 0, 1]
    x2 = keypoints_batch[:, 1, 0]
    y2 = keypoints_batch[:, 1, 1]
    x3 = keypoints_batch[:, 2, 0]
    y3 = keypoints_batch[:, 2, 1]

    dx = x2 - x1
    dy = y2 - y1
    denom = np.sqrt(dx ** 2 + dy ** 2)
    denom = np.maximum(denom, 1e-8)  # avoid division by zero

    numerator = np.abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    mls_px = numerator / denom
    mls_mm = mls_px * spacing_x
    return mls_mm


# ═════════════════════════════════════════════════════════════════════════
# 5. MLS Binning & Accuracy Metrics
# ═════════════════════════════════════════════════════════════════════════

# Triage-relevant MLS bins (same thresholds as competition)
MLS_BINS = [0.0, 1.0, 3.0, 5.0, float("inf")]
MLS_BIN_LABELS = ["<1mm", "1-3mm", "3-5mm", ">=5mm"]


def assign_mls_bin(mls_mm: float) -> int:
    """
    Assign an MLS value to its triage-relevant bin.

    Bins:
        0: < 1 mm    (no meaningful shift, noise level)
        1: 1-3 mm    (mild shift)
        2: 3-5 mm    (moderate shift, urgent)
        3: >= 5 mm   (critical shift)

    Args:
        mls_mm: MLS value in mm.

    Returns:
        Integer bin index (0-3).
    """
    if mls_mm < 1.0:
        return 0
    elif mls_mm < 3.0:
        return 1
    elif mls_mm < 5.0:
        return 2
    else:
        return 3


def compute_mls_binning_accuracy(
    mls_true: np.ndarray,
    mls_pred: np.ndarray,
) -> float:
    """
    Compute the accuracy of MLS bin classification.

    This metric directly relates to the competition's QWK score:
    if the predicted MLS falls in the correct triage-relevant bucket,
    it's more likely to produce the correct triage decision.

    Args:
        mls_true: numpy array (N,) of ground-truth MLS values in mm.
        mls_pred: numpy array (N,) of predicted MLS values in mm.

    Returns:
        Fraction of samples where predicted bin matches true bin.
    """
    true_bins = np.array([assign_mls_bin(v) for v in mls_true])
    pred_bins = np.array([assign_mls_bin(v) for v in mls_pred])
    return float((true_bins == pred_bins).mean())


def compute_mls_metrics(
    mls_true: np.ndarray,
    mls_pred: np.ndarray,
) -> dict:
    """
    Compute comprehensive MLS metrics for validation logging.

    Args:
        mls_true: numpy array (N,) of ground-truth MLS values in mm.
        mls_pred: numpy array (N,) of predicted MLS values in mm.

    Returns:
        Dictionary with keys:
            - mls_mae_mm: Mean Absolute Error (mm)
            - mls_rmse_mm: Root Mean Squared Error (mm)
            - mls_bin_acc: Binning accuracy (fraction correct)
            - bin_acc_per_bin: dict {bin_idx: accuracy} per triage bin
              (<1 / 1-3 / 3-5 / >=5 mm). Missing bins have value None.
            - mls_mae_critical: MAE for samples with MLS >= 3mm
            - mls_mae_low: MAE for samples with MLS < 3mm
            - n_samples: Number of samples
    """
    mae = float(np.abs(mls_true - mls_pred).mean())
    rmse = float(np.sqrt(((mls_true - mls_pred) ** 2).mean()))
    true_bins = np.array([assign_mls_bin(v) for v in mls_true])
    pred_bins = np.array([assign_mls_bin(v) for v in mls_pred])
    bin_acc = float((true_bins == pred_bins).mean())

    # Per-bin accuracy (triage-relevant: which bins is the model getting right?)
    per_bin_acc: dict = {}
    for b in range(len(MLS_BINS) - 1):
        mask = true_bins == b
        n_b = int(mask.sum())
        per_bin_acc[b] = (
            float((pred_bins[mask] == b).mean()) if n_b else None
        )

    # Critical regime (MLS >= 3mm) — higher sensitivity needed
    critical_mask = mls_true >= 3.0
    if critical_mask.sum() > 0:
        mae_critical = float(np.abs(mls_true[critical_mask] - mls_pred[critical_mask]).mean())
    else:
        mae_critical = 0.0

    low_mask = mls_true < 3.0
    if low_mask.sum() > 0:
        mae_low = float(np.abs(mls_true[low_mask] - mls_pred[low_mask]).mean())
    else:
        mae_low = 0.0

    return {
        "mls_mae_mm": mae,
        "mls_rmse_mm": rmse,
        "mls_bin_acc": bin_acc,
        "bin_acc_per_bin": per_bin_acc,
        "mls_mae_critical": mae_critical,
        "mls_mae_low": mae_low,
        "n_samples": len(mls_true),
    }
