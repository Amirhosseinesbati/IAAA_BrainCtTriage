"""
model.py — Competition Model API for IAAA 2026 Brain CT Triage.

Self-contained module (safe to ship inside the submission zip). It implements
the standard Model API required by the competition:

    predict(study_dir: str, models: dict) -> dict

which returns exactly 7 intermediate imaging quantities:

    V_EDH, V_SDH, V_IPH, V_SAH, V_IVH   hemorrhage volumes in mL
    fracture_prob                       probability of skull fracture in [0, 1]
    MLS_mm                              midline shift in mm

Model components bundled in ``models/``:

    models/monai/SegResNet_best.pth     ICH segmentation (MONAI SegResNet)
    models/yolo/best.pt                 fracture detection (Ultralytics YOLO)
    models/mls_heatmap/mls_heatmap_best.pth   MLS keypoints (HRNet heatmap)

MLS uses the **heatmap strategy only** (no legacy slice selector / keypoint
regression models): the HRNet heatmap model runs on every slice, keypoints
are decoded with DARK sub-pixel refinement, and only confident slices
(minimum heatmap peak >= ``mls_min_peak``) contribute to the per-study MLS,
which is aggregated with ``max`` to match the competition ground truth
(MLS = max over the annotated slices of the study).

Usage:
    from model import load_models, predict

    models = load_models("models", device="auto")
    intermediates = predict("/path/to/dicom/study", models)
    triage_class = triage_from_intermediates(intermediates)  # 0, 1, or 2
"""

import glob
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_INFERENCE_CONFIG_PATH = _PACKAGE_ROOT / "config" / "inference.yaml"
with _INFERENCE_CONFIG_PATH.open("r", encoding="utf-8") as _config_stream:
    INFERENCE_CONFIG = yaml.safe_load(_config_stream)

# ===========================================================================
# 1.  Constants
# ===========================================================================

# CT windowing settings (brain / subdural / bone)
WINDOWS = {
    "brain":    {"width": 80,   "level": 40},
    "subdural": {"width": 200,  "level": 80},
    "bone":     {"width": 1000, "level": 400},
}

# ICH label mapping (same as src/config.py)
ICH_LABELS = {"background": 0, "IVH": 1, "IPH": 2, "SDH": 3, "EDH": 4, "SAH": 5}
ICH_LABEL_NAMES = {v: k for k, v in ICH_LABELS.items()}
NUM_ICH_CLASSES = len(ICH_LABELS)

# MLS heatmap model input resolution (matches training: 512x512 images)
ML_INPUT_SIZE = 512

# Default MLS inference knobs (override via load_models(..., mls_*=...))
MLS_MIN_PEAK = float(INFERENCE_CONFIG["mls"]["min_peak"])
MLS_TOP_K = INFERENCE_CONFIG["mls"]["top_k"]
MLS_AGGREGATION = INFERENCE_CONFIG["mls"]["aggregation"]
MLS_BATCH_SIZE = int(INFERENCE_CONFIG["mls"]["batch_size"])


def _mls_clip_bounds() -> tuple[float, float]:
    """Read the one submission-owned MLS physical range fail-closed."""
    values = INFERENCE_CONFIG.get("outputs", {}).get("mls_clip_mm")
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("config.outputs.mls_clip_mm must be a two-value range")
    lower, upper = (float(values[0]), float(values[1]))
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
        raise ValueError("config.outputs.mls_clip_mm is not a finite nonnegative range")
    return lower, upper


def _remove_small_components(label_map: np.ndarray, voxel_vol_ml: float) -> np.ndarray:
    """Suppress disconnected ICH blobs below physical per-class volumes."""
    from scipy import ndimage

    cleaned = label_map.copy()
    structure = ndimage.generate_binary_structure(3, 2)
    for raw_label, minimum_ml in INFERENCE_CONFIG["ich"]["min_component_ml"].items():
        label_id = int(raw_label)
        components, count = ndimage.label(cleaned == label_id, structure=structure)
        if not count:
            continue
        counts = np.bincount(components.ravel())
        remove = np.flatnonzero(counts * voxel_vol_ml < float(minimum_ml))
        remove = remove[remove != 0]
        if len(remove):
            cleaned[np.isin(components, remove)] = 0
    return cleaned


def _load_calibration(models_path: Path) -> Optional[dict]:
    config = INFERENCE_CONFIG["calibration"]
    if not config.get("enabled", False):
        return None
    relative = Path(config["path"])
    if relative.is_absolute():
        path = relative
    elif relative.parts and relative.parts[0] == "models":
        path = models_path / Path(*relative.parts[1:])
    else:
        path = _PACKAGE_ROOT / relative
    if not path.exists():
        if config.get("optional", True):
            logger.info("No calibration bundle found at %s; using physical sanitization only", path)
            return None
        raise FileNotFoundError(f"Required calibration bundle not found: {path}")
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported calibration bundle schema")
    return payload


def _finalize_intermediates(
    values: Dict[str, float],
    calibration: Optional[dict],
    *,
    preserve_locked_mls: bool = False,
) -> Dict[str, float]:
    result = {key: float(value) for key, value in values.items()}
    if calibration:
        mappings = calibration.get("mappings", {})
        missing = set(result) - set(mappings)
        if missing:
            raise ValueError(f"Incomplete calibration bundle: {sorted(missing)}")
        result = {
            key: (
                float(value)
                if preserve_locked_mls and key == "MLS_mm"
                else float(np.interp(value, mappings[key]["x"], mappings[key]["y"], left=mappings[key]["y"][0], right=mappings[key]["y"][-1]))
            )
            for key, value in result.items()
        }
    floor = float(INFERENCE_CONFIG["outputs"]["volume_noise_floor_ml"])
    for key in ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"):
        value = max(0.0, result[key])
        result[key] = 0.0 if value < floor else value
    result["fracture_prob"] = float(np.clip(result["fracture_prob"], *INFERENCE_CONFIG["outputs"]["fracture_clip"]))
    result["MLS_mm"] = float(np.clip(result["MLS_mm"], *_mls_clip_bounds()))
    return result


# ===========================================================================
# 2.  Self-contained DICOM reader
# ===========================================================================

def apply_windowing(image_hu: np.ndarray, width: float, level: float) -> np.ndarray:
    """Apply CT windowing and normalise to [0, 1]."""
    low = level - width / 2.0
    high = level + width / 2.0
    return np.clip((image_hu - low) / (high - low), 0.0, 1.0)


class DicomReader:
    """Minimal DICOM reader for a single study."""

    def __init__(self, study_dir: str):
        self.study_dir = study_dir
        self.slices: List[pydicom.Dataset] = []
        self.metadata: Dict[str, Any] = {}
        self._hu_volume: Optional[np.ndarray] = None

    def load_and_sort(self) -> "DicomReader":
        """Load all DICOM slices and sort by Z-axis position."""
        dcm_files = sorted(glob.glob(os.path.join(self.study_dir, "*.dcm")))
        if not dcm_files:
            raise FileNotFoundError(
                f"No DICOM files (*.dcm) found in {self.study_dir}"
            )

        self.slices = []
        for fpath in dcm_files:
            try:
                ds = pydicom.dcmread(fpath, force=True)
                self.slices.append(ds)
            except Exception:
                continue

        if not self.slices:
            raise ValueError(f"Failed to read any DICOM files from {self.study_dir}")

        # Sort by Z position
        self.slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Extract metadata from first slice
        first = self.slices[0]
        spacing_xy = getattr(first, "PixelSpacing", [1.0, 1.0])

        if len(self.slices) > 1:
            z_spacing = abs(
                float(self.slices[1].ImagePositionPatient[2])
                - float(self.slices[0].ImagePositionPatient[2])
            )
        else:
            z_spacing = float(getattr(first, "SliceThickness", 1.0))

        self.metadata = {
            "patient_id": os.path.basename(self.study_dir),
            "spacing_x": float(spacing_xy[1]),
            "spacing_y": float(spacing_xy[0]),
            "spacing_z": float(z_spacing),
            "rows": int(getattr(first, "Rows", 512)),
            "columns": int(getattr(first, "Columns", 512)),
        }
        return self

    def get_3d_volume_hu(self) -> np.ndarray:
        """Return the full 3D volume in Hounsfield Units (H, W, D)."""
        if self._hu_volume is not None:
            return self._hu_volume

        slices_hu = []
        for ds in self.slices:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu = (ds.pixel_array.astype(np.float32) * slope) + intercept
            slices_hu.append(hu)

        self._hu_volume = np.stack(slices_hu, axis=-1)
        return self._hu_volume

    def save_as_nifti(self, output_path: str) -> str:
        """Export the 3D HU volume as a NIfTI file."""
        vol = self.get_3d_volume_hu()
        affine = np.diag([
            self.metadata["spacing_x"],
            self.metadata["spacing_y"],
            self.metadata["spacing_z"],
            1.0,
        ])
        nib.save(nib.Nifti1Image(vol, affine), output_path)
        return output_path

    def __len__(self) -> int:
        return len(self.slices)


# ===========================================================================
# 3.  MLS heatmap model (matches src/strategies/mls_heatmap/model.py)
# ===========================================================================

class _MLSHeatmapHead(torch.nn.Module):
    """Heatmap prediction head matching the training-time ``HeatmapHead``.

    Uses NAMED submodules (``conv1``/``bn1``/``relu``/``conv2``) exactly like
    ``src.strategies.mls_heatmap.model.HeatmapHead`` so that the state dict
    keys stored in ``mls_heatmap_best.pth`` (e.g. ``head.conv1.weight``)
    load correctly.
    """

    def __init__(self, in_channels: int, num_keypoints: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout2d(float(dropout))
        self.conv2 = torch.nn.Conv2d(64, num_keypoints, kernel_size=1)

        # Small-init the final conv (same as training) for safety on partial loads.
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.001)
        if self.conv2.bias is not None:
            torch.nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class _MLSHeatmapModel(torch.nn.Module):
    """HRNet heatmap model, including the optional locked slice-selector head.

    ``backbone`` and ``input_channels`` are read from the checkpoint's saved
    ``config`` so any trained variant (hrnet_w32 / hrnet_w18, 1 or 3 input
    channels) is reconstructed correctly. The first convolution is adapted
    when the checkpoint was trained with != 3 channels.
    """

    def __init__(
        self,
        backbone_name: str = "hrnet_w18",
        in_channels: int = 3,
        head_dropout: float = 0.0,
        use_selector: bool = False,
        selector_head_mode: str = "single",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_selector = bool(use_selector)
        self.selector_head_mode = str(selector_head_mode)
        self._use_timm = False
        if self.selector_head_mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported MLS selector head mode: {self.selector_head_mode}")

        try:
            import timm
            self.backbone = timm.create_model(
                backbone_name, pretrained=False,
                features_only=True, out_indices=(1,),  # 1/4 resolution
            )
            feat_dim = self.backbone.feature_info.channels()[0]
            self._use_timm = True
        except ImportError:
            logger.info("timm not available, using ResNet34 fallback for heatmap model")
            from torchvision.models import resnet34
            base = resnet34(weights=None, num_classes=1000)
            self.backbone = torch.nn.Sequential(*list(base.children())[:-2])
            feat_dim = 512

        # Adapt first conv if the trained model used != 3 input channels.
        if in_channels != 3:
            self._adapt_input_channels()

        if self._use_timm:
            self.head = _MLSHeatmapHead(feat_dim, num_keypoints=3, dropout=head_dropout)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(feat_dim, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
                torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 3, kernel_size=1),
            )
        self.selector_head = None
        if self.use_selector:
            if not self._use_timm:
                raise ValueError("Selector MLS runtime requires the timm HRNet backbone")
            self.selector_head = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(feat_dim, 64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=max(0.1, float(head_dropout))),
                torch.nn.Linear(64, 2 if self.selector_head_mode == "dual" else 1),
            )

    def _adapt_input_channels(self) -> None:
        """Replace the first convolution to match the trained ``in_channels``.

        Averages the pretrained RGB weights for 1-channel input; repeats them
        for more than 3 channels (mirrors the training code).
        """
        if not self._use_timm:
            raise ValueError("Input-channel adaptation is only supported for the timm backbone")

        old_conv = self.backbone.conv1
        new_conv = torch.nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        if self.in_channels == 1:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        elif self.in_channels < 3:
            new_conv.weight.data = old_conv.weight.data[:, :self.in_channels]
        else:
            repeats = self.in_channels // 3 + 1
            repeated = old_conv.weight.data.repeat(1, repeats, 1, 1)
            new_conv.weight.data = repeated[:, :self.in_channels]

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        self.backbone.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_timm:
            features = self.backbone(x)
            feat = features[0]
        else:
            feat = self.backbone(x)
        return self.head(feat)

    def forward_multitask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror the training-time heatmap + selector forward contract exactly."""
        if self.selector_head is None:
            raise RuntimeError("MLS checkpoint does not contain an explicit selector head")
        if not self._use_timm:
            raise RuntimeError("Selector MLS runtime requires the timm HRNet backbone")
        feat = self.backbone(x)[0]
        selector_logits = self.selector_head(feat)
        if self.selector_head_mode == "single":
            selector_logits = selector_logits.squeeze(1)
        return self.head(feat), selector_logits


# ===========================================================================
# 4.  DARK sub-pixel decoding + MLS geometry (standalone copies)
# ===========================================================================

def _decode_heatmap_dark(
    heatmap: torch.Tensor,
    heatmap_size: int,
    img_size: int,
) -> tuple[float, float]:
    """DARK sub-pixel decoding — standalone copy (no external deps)."""
    H, W = heatmap.shape
    scale = img_size / heatmap_size

    max_val = heatmap.max()
    if max_val < 1e-8:
        return (-1.0, -1.0)

    max_idx = heatmap.argmax()
    y0, x0 = int(max_idx // W), int(max_idx % W)

    if y0 == 0 or y0 == H - 1 or x0 == 0 or x0 == W - 1:
        return (float(x0) * scale, float(y0) * scale)

    g_x = (heatmap[y0, x0 + 1] - heatmap[y0, x0 - 1]) / 2.0
    g_y = (heatmap[y0 + 1, x0] - heatmap[y0 - 1, x0]) / 2.0

    H_xx = heatmap[y0, x0 + 1] - 2.0 * heatmap[y0, x0] + heatmap[y0, x0 - 1]
    H_yy = heatmap[y0 + 1, x0] - 2.0 * heatmap[y0, x0] + heatmap[y0 - 1, x0]
    H_xy = (heatmap[y0 + 1, x0 + 1] - heatmap[y0 + 1, x0 - 1]
            - heatmap[y0 - 1, x0 + 1] + heatmap[y0 - 1, x0 - 1]) / 4.0

    det = H_xx * H_yy - H_xy * H_xy
    if abs(det) < 1e-12:
        return (float(x0) * scale, float(y0) * scale)

    delta_x = -(H_yy * g_x - H_xy * g_y) / det
    delta_y = -(H_xx * g_y - H_xy * g_x) / det
    delta_x = max(-0.5, min(0.5, delta_x.item()))
    delta_y = max(-0.5, min(0.5, delta_y.item()))

    x_sub = max(0.0, min(img_size - 1, (float(x0) + delta_x) * scale))
    y_sub = max(0.0, min(img_size - 1, (float(y0) + delta_y) * scale))
    return (x_sub, y_sub)


def _decode_heatmap_dark_batch(
    heatmaps: torch.Tensor,
    heatmap_size: int,
    img_size: int,
) -> np.ndarray:
    """Batch DARK decoding. Returns (B, K, 2) array of sub-pixel coords."""
    B, K = heatmaps.shape[0], heatmaps.shape[1]
    coords = np.zeros((B, K, 2), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            x, y = _decode_heatmap_dark(heatmaps[b, k], heatmap_size, img_size)
            coords[b, k, 0] = x
            coords[b, k, 1] = y
    return coords


def _decode_heatmap_dark_batch_with_scores(
    heatmaps: torch.Tensor,
    heatmap_size: int,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact DARK coordinates and per-keypoint maxima used in training."""
    coords = _decode_heatmap_dark_batch(heatmaps, heatmap_size, img_size)
    scores = heatmaps.amax(dim=(2, 3)).cpu().numpy().astype(np.float32, copy=False)
    return coords, scores


@dataclass(frozen=True)
class _MLSSlicePrediction:
    """Minimal self-contained equivalent of the training MLS slice record."""

    index: int
    selector_probability: float
    mls_mm: float
    heatmap_peak: float
    peak_probability: Optional[float] = None


def _selector_rank_probability(item: _MLSSlicePrediction) -> float:
    return item.selector_probability if item.peak_probability is None else item.peak_probability


def _split_selector_logits(
    selector_logits: torch.Tensor,
    selector_head_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if selector_head_mode == "dual":
        if selector_logits.ndim != 2 or selector_logits.shape[1] != 2:
            raise ValueError(f"Dual selector expected [batch, 2], got {tuple(selector_logits.shape)}")
        return selector_logits[:, 0], selector_logits[:, 1]
    if selector_head_mode != "single" or selector_logits.ndim != 1:
        raise ValueError(f"Single selector expected [batch], got {tuple(selector_logits.shape)}")
    return selector_logits, selector_logits


def _aggregate_selector_study_mls(
    predictions: List[_MLSSlicePrediction],
    *,
    selector_threshold: float,
    top_k: int,
    aggregation: str,
    relative_ratio: float,
    aggregation_quantile: float,
    probability_weighted: bool,
    anchor_window_radius: int,
    min_active_slices: int,
    heatmap_guard_ratio: float,
    negative_value: float,
) -> float:
    """Self-contained mirror of training ``aggregate_study_mls``."""
    if not predictions:
        return float(negative_value)
    gate_ranked = sorted(predictions, key=lambda item: item.selector_probability, reverse=True)
    if gate_ranked[0].selector_probability < selector_threshold:
        return float(negative_value)
    if sum(item.selector_probability >= selector_threshold for item in predictions) < min_active_slices:
        return float(negative_value)

    if aggregation in {"relative_component", "anchor_window", "joint_component", "severity_window"}:
        ordered = sorted(predictions, key=lambda item: item.index)
        rank_probabilities = np.asarray([_selector_rank_probability(item) for item in ordered], dtype=float)
        heatmap_peaks = np.asarray([item.heatmap_peak for item in ordered], dtype=float)
        if aggregation == "joint_component":
            peak_scale = heatmap_peaks / max(float(heatmap_peaks.max()), 1e-8)
            component_scores = rank_probabilities * np.sqrt(np.maximum(peak_scale, 0.0))
        elif aggregation == "severity_window":
            peak_scale = heatmap_peaks / max(float(heatmap_peaks.max()), 1e-8)
            values = np.asarray([item.mls_mm for item in ordered], dtype=float)
            value_scale = np.clip(values, 0.0, 30.0) / max(float(np.clip(values, 0.0, 30.0).max()), 1e-8)
            component_scores = rank_probabilities * np.sqrt(np.maximum(peak_scale, 0.0)) * np.sqrt(np.maximum(value_scale, 0.0))
        else:
            component_scores = rank_probabilities
        anchor = int(np.argmax(component_scores))
        if aggregation in {"relative_component", "joint_component"}:
            active = component_scores >= component_scores[anchor] * relative_ratio
            left = anchor
            right = anchor
            while left > 0 and active[left - 1]:
                left -= 1
            while right + 1 < len(ordered) and active[right + 1]:
                right += 1
        else:
            left = max(0, anchor - anchor_window_radius)
            right = min(len(ordered) - 1, anchor + anchor_window_radius)
        selected = ordered[left:right + 1]
        if heatmap_guard_ratio > 0:
            selected_peaks = np.asarray([item.heatmap_peak for item in selected], dtype=float)
            peak_gate = float(selected_peaks.max()) * heatmap_guard_ratio
            guarded = [item for item in selected if item.heatmap_peak >= peak_gate]
            if guarded:
                selected = guarded
        values = np.asarray([item.mls_mm for item in selected], dtype=float)
        if probability_weighted:
            weights = np.asarray([_selector_rank_probability(item) for item in selected], dtype=float)
            order = np.argsort(values)
            ordered_values = values[order]
            ordered_weights = np.maximum(weights[order], 1e-8)
            cutoff = aggregation_quantile * float(ordered_weights.sum())
            index = min(int(np.searchsorted(np.cumsum(ordered_weights), cutoff)), len(ordered_values) - 1)
            return float(ordered_values[index])
        return float(np.quantile(values, aggregation_quantile))

    ranked = sorted(predictions, key=_selector_rank_probability, reverse=True)
    selected = ranked[:top_k]
    values = np.asarray([item.mls_mm for item in selected], dtype=float)
    if aggregation == "median":
        return float(np.median(values))
    if aggregation == "p90":
        return float(np.percentile(values, 90))
    if aggregation == "max":
        return float(values.max())
    if aggregation == "quantile":
        if probability_weighted:
            weights = np.asarray([_selector_rank_probability(item) for item in selected], dtype=float)
            order = np.argsort(values)
            ordered_values = values[order]
            ordered_weights = np.maximum(weights[order], 1e-8)
            cutoff = aggregation_quantile * float(ordered_weights.sum())
            index = min(int(np.searchsorted(np.cumsum(ordered_weights), cutoff)), len(ordered_values) - 1)
            return float(ordered_values[index])
        return float(np.quantile(values, aggregation_quantile))
    raise ValueError(f"Unsupported selector MLS aggregation: {aggregation}")


def _calculate_mls(coords_pixels: np.ndarray, spacing_x: float) -> float:
    """MLS = perpendicular distance from point 3 to the falx line (1-2)."""
    x1, y1, x2, y2, x3, y3 = coords_pixels
    num = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return (num / den if den > 0 else 0.0) * spacing_x


# ===========================================================================
# 5.  Windowed CT input (matches the training-time 3-channel PNG)
# ===========================================================================

def _create_windowed_input(hu_image: np.ndarray, in_channels: int = 3):
    """Window a HU slice into the number of channels the model expects.

    Training images were 3-channel PNGs with channels (brain, subdural, bone).
    For ``in_channels == 1`` only the brain window is used.
    """
    if in_channels == 1:
        ch1 = apply_windowing(hu_image, WINDOWS["brain"]["width"], WINDOWS["brain"]["level"])
        return ch1[None, ...]  # (1, H, W)

    ch1 = apply_windowing(hu_image, WINDOWS["brain"]["width"], WINDOWS["brain"]["level"])
    ch2 = apply_windowing(hu_image, WINDOWS["subdural"]["width"], WINDOWS["subdural"]["level"])
    ch3 = apply_windowing(hu_image, WINDOWS["bone"]["width"], WINDOWS["bone"]["level"])
    return np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)


# ===========================================================================
# 6.  MLS inference — heatmap strategy only (no slice selector)
# ===========================================================================

def _predict_mls_heatmap(
    vol_hu: np.ndarray,
    heatmap_model: torch.nn.Module,
    spacing_x: float,
    device: torch.device,
    batch_size: int = MLS_BATCH_SIZE,
    min_peak: float = MLS_MIN_PEAK,
    top_k: Optional[int] = MLS_TOP_K,
    aggregation: str = MLS_AGGREGATION,
    *,
    use_selector: bool = False,
    selector_head_mode: str = "single",
    selector_threshold: float = 0.5,
    selector_relative_ratio: float = 0.3,
    aggregation_quantile: float = 0.75,
    aggregation_probability_weighted: bool = False,
    anchor_window_radius: int = 2,
    min_active_slices: int = 1,
    heatmap_guard_ratio: float = 0.0,
    negative_value_mm: float = 0.1,
    return_trace: bool = False,
) -> Any:
    """Predict MLS using the checkpoint's exact declared inference contract.

    Legacy heatmap-only checkpoints retain their historical min-peak/max-or-p90
    path.  Selector checkpoints follow the training implementation exactly:
    spatial-softmax, DARK, slice selector, and component pooling.
    """
    in_channels = int(getattr(heatmap_model, "in_channels", 3))
    if in_channels not in {1, 3}:
        raise ValueError(f"Submission MLS runtime supports only 1/3 channels, got {in_channels}")
    if use_selector != bool(getattr(heatmap_model, "use_selector", False)):
        raise ValueError("MLS checkpoint/runtime selector contract mismatch")
    n_slices = vol_hu.shape[2]
    orig_h = int(vol_hu.shape[0])
    eff_spacing = float(spacing_x) * (orig_h / ML_INPUT_SIZE)
    legacy_candidates: List[tuple[float, float, float]] = []
    selector_candidates: List[_MLSSlicePrediction] = []

    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        batch_imgs = []
        for z in range(start, end):
            win = _create_windowed_input(vol_hu[:, :, z], in_channels)
            tensor = torch.from_numpy(win).float().unsqueeze(0)
            if tensor.shape[-2:] != (ML_INPUT_SIZE, ML_INPUT_SIZE):
                tensor = F.interpolate(
                    tensor, size=(ML_INPUT_SIZE, ML_INPUT_SIZE),
                    mode="bilinear", align_corners=False,
                )
            batch_imgs.append(tensor)
        inputs = torch.cat(batch_imgs, dim=0).to(device)

        with torch.inference_mode():
            if use_selector:
                heatmap_logits, selector_logits = heatmap_model.forward_multitask(inputs)
                if not torch.isfinite(heatmap_logits).all() or not torch.isfinite(selector_logits).all():
                    raise FloatingPointError("Non-finite selector MLS CUDA output")
                spatial_probabilities = torch.softmax(heatmap_logits.flatten(2), dim=-1).reshape_as(heatmap_logits)
                coords, peaks = _decode_heatmap_dark_batch_with_scores(
                    spatial_probabilities.cpu(), spatial_probabilities.shape[-1], ML_INPUT_SIZE,
                )
                target_logits, peak_logits = _split_selector_logits(selector_logits, selector_head_mode)
                selector_probabilities = torch.sigmoid(target_logits).cpu().numpy()
                peak_probabilities = torch.sigmoid(peak_logits).cpu().numpy()
            else:
                heatmaps = heatmap_model(inputs)
                peaks = heatmaps.amax(dim=(2, 3)).cpu().numpy()
                coords = _decode_heatmap_dark_batch(heatmaps.cpu(), heatmaps.shape[-1], ML_INPUT_SIZE)

        for offset, keypoints in enumerate(coords):
            mls = 0.0
            if (keypoints[:, 0] >= 0).all():
                mls = float(_calculate_mls(keypoints.ravel(), eff_spacing))
            if use_selector:
                selector_candidates.append(_MLSSlicePrediction(
                    index=start + offset,
                    selector_probability=float(selector_probabilities[offset]),
                    peak_probability=float(peak_probabilities[offset]),
                    mls_mm=mls,
                    heatmap_peak=float(np.min(peaks[offset])),
                ))
            elif (keypoints[:, 0] >= 0).all():
                legacy_candidates.append((mls, float(peaks[offset].min()), float(peaks[offset].mean())))

    if use_selector:
        result = _aggregate_selector_study_mls(
            selector_candidates,
            selector_threshold=float(selector_threshold),
            top_k=max(1, int(top_k or 1)),
            aggregation=str(aggregation),
            relative_ratio=float(selector_relative_ratio),
            aggregation_quantile=float(aggregation_quantile),
            probability_weighted=bool(aggregation_probability_weighted),
            anchor_window_radius=int(anchor_window_radius),
            min_active_slices=int(min_active_slices),
            heatmap_guard_ratio=float(heatmap_guard_ratio),
            negative_value=float(negative_value_mm),
        )
        clipped = float(np.clip(result, *_mls_clip_bounds()))
        return (clipped, tuple(selector_candidates)) if return_trace else clipped

    if not legacy_candidates:
        logger.warning("No valid MLS measurements for this study. Returning 0.0.")
        return (0.0, tuple()) if return_trace else 0.0
    if top_k is not None and top_k > 0:
        selected = sorted(legacy_candidates, key=lambda item: -item[1])[:top_k]
    else:
        selected = [item for item in legacy_candidates if item[1] >= min_peak]
        if not selected:
            selected = [max(legacy_candidates, key=lambda item: item[1])]
    mls_values = np.array([item[0] for item in selected])
    legacy_result = float(np.percentile(mls_values, 90) if aggregation == "p90" else mls_values.max())
    return (legacy_result, tuple()) if return_trace else legacy_result


# ===========================================================================
# 7.  MONAI ICH segmentation
# ===========================================================================

def _load_ich_monai(models_path: Path, device: torch.device):
    """Load the MONAI ICH model from ``models/monai/``.

    The network type is inferred from the checkpoint filename
    (``*segresnet*`` / ``*swin*`` / ``*dynunet*``, else UNETR).
    """
    from monai.networks.nets import SegResNet, SwinUNETR, DynUNet, UNETR

    ckpt_path = models_path / "monai" / "SegResNet_best.pth"
    if not ckpt_path.exists():
        candidates = list((models_path / "monai").glob("*.pth"))
        if candidates:
            ckpt_path = candidates[0]
        else:
            raise FileNotFoundError(f"No MONAI checkpoint found in {models_path / 'monai'}")

    fname = ckpt_path.stem.lower()
    if "segresnet" in fname:
        model = SegResNet(
            spatial_dims=3, in_channels=1, out_channels=NUM_ICH_CLASSES,
            init_filters=16, blocks_down=(1, 2, 2, 4), dropout_prob=0.1,
        )
    elif "swin" in fname:
        model = SwinUNETR(
            in_channels=1, out_channels=NUM_ICH_CLASSES,
            patch_size=(2, 2, 2), window_size=(7, 14, 14),
            feature_size=48, use_checkpoint=False,
        )
    elif "dynunet" in fname:
        model = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=NUM_ICH_CLASSES,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=[32, 64, 128, 256, 320],
            dropout=0.1,
            deep_supervision=True,
            deep_sup_num=3,
            res_block=True,
        )
    else:  # Default: UNETR
        model = UNETR(
            in_channels=1, out_channels=NUM_ICH_CLASSES,
            img_size=(128, 128, 128), feature_size=16,
            hidden_size=768, mlp_dim=3072, num_heads=12,
            pos_embed="perceptron", norm_name="instance",
            res_block=True, dropout_rate=0.0,
        )

    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _predict_ich_monai(model, reader: DicomReader, device: torch.device) -> dict:
    """Run MONAI 3D inference with training-matched preprocessing.

    Uses the same pipeline as training (Spacingd -> 1mm isotropic,
    CropForegroundd, ScaleIntensityRanged). Prediction is resized back to
    the original DICOM space for accurate volume calculation.
    """
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        Resize,
        ScaleIntensityRanged,
        Spacingd,
        ToTensord,
    )

    # Stride factor of the MONAI model (SegResNet blocks_down=(1,2,2,4) = 16)
    STRIDE = 16

    with tempfile.TemporaryDirectory() as tmp:
        nifti_path = os.path.join(tmp, "input.nii.gz")
        reader.save_as_nifti(nifti_path)

        # Preprocessing — matches the MONAI training pipeline.
        preproc = Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys="image", a_min=-200, a_max=300,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys="image", source_key="image"),
            ToTensord(keys="image"),
        ])

        data = preproc({"image": nifti_path})
        inp = data["image"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
        orig_spatial = inp.shape[2:]                 # (D, H, W) before padding
        n_voxels = inp[0, 0].numel()

        # Pad to multiples of STRIDE (required for U-Net skip connections).
        pad_d = (STRIDE - orig_spatial[0] % STRIDE) % STRIDE
        pad_h = (STRIDE - orig_spatial[1] % STRIDE) % STRIDE
        pad_w = (STRIDE - orig_spatial[2] % STRIDE) % STRIDE

        needs_pad = (pad_d + pad_h + pad_w) > 0
        if needs_pad:
            inp = torch.nn.functional.pad(
                inp,
                (0, pad_w, 0, pad_h, 0, pad_d),  # (W, H, D) right pads
                mode="replicate",
            )

        # Full-volume forward for typical studies; sliding window for very
        # large volumes to stay within GPU memory.
        MAX_VOXELS_FULL_VOLUME = 8_000_000  # ~200³ — fits any GPU / fast CPU
        if n_voxels > MAX_VOXELS_FULL_VOLUME:
            from monai.inferers import sliding_window_inference
            pred_logits = sliding_window_inference(
                inputs=inp,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
            )
        else:
            with torch.no_grad():
                pred_logits = model(inp)

        # Crop back to the original (pre-pad) spatial size.
        if needs_pad:
            ds, hs, ws = orig_spatial
            pred_logits = pred_logits[:, :, :ds, :hs, :ws]

        pred_labels = pred_logits.argmax(dim=1).cpu()  # (1, D, H, W) in 1mm space

        # Resize back to the original DICOM dimensions.
        H_orig = reader.metadata["rows"]
        W_orig = reader.metadata["columns"]
        D_orig = len(reader)

        pred_orig = Resize(
            spatial_size=(H_orig, W_orig, D_orig), mode="nearest",
        )(pred_labels).squeeze().numpy().astype(np.int64)

    # Volumes using the original voxel spacing.
    voxel_vol_ml = (
        reader.metadata["spacing_x"] *
        reader.metadata["spacing_y"] *
        reader.metadata["spacing_z"] / 1000.0
    )

    pred_orig = _remove_small_components(pred_orig, voxel_vol_ml)

    return {
        "V_IVH": float(np.sum(pred_orig == 1) * voxel_vol_ml),
        "V_IPH": float(np.sum(pred_orig == 2) * voxel_vol_ml),
        "V_SDH": float(np.sum(pred_orig == 3) * voxel_vol_ml),
        "V_EDH": float(np.sum(pred_orig == 4) * voxel_vol_ml),
        "V_SAH": float(np.sum(pred_orig == 5) * voxel_vol_ml),
    }


# ===========================================================================
# 8.  YOLO fracture detection
# ===========================================================================

def _predict_fracture(vol_hu: np.ndarray, yolo_model, device: torch.device) -> float:
    """Max YOLO box confidence over all slices on the bone window."""
    import cv2

    max_conf = 0.0
    for z in range(vol_hu.shape[2]):
        slice_hu = vol_hu[:, :, z]
        bone_img = apply_windowing(
            slice_hu, WINDOWS["bone"]["width"], WINDOWS["bone"]["level"],
        )
        img_rgb = cv2.cvtColor((bone_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        results = yolo_model.predict(img_rgb, device=device, verbose=False)
        for res in results:
            if res.boxes is not None and len(res.boxes) > 0:
                conf = float(res.boxes.conf.max())
                if conf > max_conf:
                    max_conf = conf
    return max_conf


# ===========================================================================
# 9.  Public API
# ===========================================================================

def _resolve_device(device: str) -> torch.device:
    """Resolve a device string; ``"auto"`` picks cuda when available."""
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_mls_checkpoint(
    models_path: Path,
    device_obj: torch.device,
) -> tuple[_MLSHeatmapModel, Dict[str, Any]]:
    """Load an MLS checkpoint without silently discarding trained heads."""
    heatmap_ckpt = models_path / "mls_heatmap" / "mls_heatmap_best.pth"
    if not heatmap_ckpt.exists():
        raise FileNotFoundError(f"MLS heatmap checkpoint not found at {heatmap_ckpt}")
    checkpoint = torch.load(str(heatmap_ckpt), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict):
        raise TypeError("MLS checkpoint config must be a mapping")
    if bool(config.get("use_ordinal_aux_head", False)) or bool(config.get("use_reference_refinement", False)):
        raise ValueError("Submission MLS runtime does not support a training-only ordinal/refinement head")
    input_channels = int(config.get("input_channels", 3))
    if input_channels not in {1, 3}:
        raise ValueError(f"Submission MLS runtime only supports 1/3 input channels, got {input_channels}")
    use_selector = bool(config.get("use_selector", False))
    if use_selector:
        if device_obj.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Locked selector MLS runtime requires CUDA; CPU fallback is forbidden")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError("Selector MLS checkpoint must contain model_state_dict")
        required = {
            "backbone", "input_channels", "head_dropout", "use_selector",
            "selector_head_mode", "selector_threshold", "top_k_slices",
            "aggregation", "selector_relative_ratio", "aggregation_quantile",
            "aggregation_probability_weighted", "anchor_window_radius",
            "min_active_slices", "heatmap_guard_ratio", "negative_value_mm",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"Selector MLS checkpoint lacks locked runtime fields: {missing}")
        backbone = str(config["backbone"])
        input_channels = int(config["input_channels"])
        head_dropout = float(config["head_dropout"])
        selector_head_mode = str(config["selector_head_mode"])
    else:
        backbone = str(config.get("backbone", "hrnet_w18"))
        head_dropout = float(config.get("head_dropout", 0.0))
        selector_head_mode = str(config.get("selector_head_mode", "single"))
    model = _MLSHeatmapModel(
        backbone_name=backbone,
        in_channels=input_channels,
        head_dropout=head_dropout,
        use_selector=use_selector,
        selector_head_mode=selector_head_mode,
    )
    # A selector checkpoint must never succeed while its selector weights are ignored.
    # Configured checkpoints are therefore fail-closed; raw legacy state dicts retain
    # the historical compatibility fallback.
    model.load_state_dict(state_dict, strict=bool(config))
    model = model.to(device_obj).eval()
    if use_selector and next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Locked selector MLS model did not load onto CUDA")
    return model, config


def _mls_runtime_options(
    config: Dict[str, Any],
    *,
    min_peak: float,
    top_k: Optional[int],
    batch_size: int,
    aggregation: str,
) -> Dict[str, Any]:
    """Use the locked checkpoint contract for selector models, not CLI surrogates."""
    use_selector = bool(config.get("use_selector", False))
    options: Dict[str, Any] = {
        "mls_min_peak": float(min_peak),
        "mls_top_k": top_k,
        "mls_batch_size": int(batch_size),
        "mls_aggregation": str(aggregation),
        "mls_use_selector": use_selector,
        "mls_selector_head_mode": str(config.get("selector_head_mode", "single")),
        "mls_selector_threshold": float(config.get("selector_threshold", 0.5)),
        "mls_selector_relative_ratio": float(config.get("selector_relative_ratio", 0.3)),
        "mls_aggregation_quantile": float(config.get("aggregation_quantile", 0.75)),
        "mls_aggregation_probability_weighted": bool(config.get("aggregation_probability_weighted", False)),
        "mls_anchor_window_radius": int(config.get("anchor_window_radius", 2)),
        "mls_min_active_slices": int(config.get("min_active_slices", 1)),
        "mls_heatmap_guard_ratio": float(config.get("heatmap_guard_ratio", 0.0)),
        "mls_negative_value_mm": float(config.get("negative_value_mm", 0.1)),
        "mls_locked_runtime": use_selector,
    }
    if use_selector:
        # Do not let command-line legacy pooling knobs mutate a locked selector
        # checkpoint. R1 was qualified with a fixed CUDA batch of eight.
        options.update({
            "mls_top_k": int(config["top_k_slices"]),
            "mls_batch_size": 8,
            "mls_aggregation": str(config["aggregation"]),
            "mls_selector_head_mode": str(config["selector_head_mode"]),
            "mls_selector_threshold": float(config["selector_threshold"]),
            "mls_selector_relative_ratio": float(config["selector_relative_ratio"]),
            "mls_aggregation_quantile": float(config["aggregation_quantile"]),
            "mls_aggregation_probability_weighted": bool(config["aggregation_probability_weighted"]),
            "mls_anchor_window_radius": int(config["anchor_window_radius"]),
            "mls_min_active_slices": int(config["min_active_slices"]),
            "mls_heatmap_guard_ratio": float(config["heatmap_guard_ratio"]),
            "mls_negative_value_mm": float(config["negative_value_mm"]),
        })
    return options


def load_models(
    models_dir: str = "models",
    device: str = "auto",
    mls_min_peak: float = MLS_MIN_PEAK,
    mls_top_k: Optional[int] = MLS_TOP_K,
    mls_batch_size: int = MLS_BATCH_SIZE,
    mls_aggregation: str = MLS_AGGREGATION,
) -> dict:
    """Load all models from the ``models`` directory.

    Expected layout::

        models/
        ├── monai/
        │   └── SegResNet_best.pth        # ICH segmentation
        ├── yolo/
        │   └── best.pt                   # fracture detection
        └── mls_heatmap/
            └── mls_heatmap_best.pth      # MLS keypoints (heatmap strategy)

    Args:
        models_dir: Path to the ``models`` directory.
        device: ``"auto"`` (default, cuda when available), ``"cuda"`` or ``"cpu"``.
        mls_min_peak: Minimum heatmap peak (all 3 keypoints) to trust a slice.
        mls_top_k: If set, keep the top-K most confident slices instead of
            the ``mls_min_peak`` threshold.
        mls_batch_size: Slices per heatmap forward pass.
        mls_aggregation: ``"max"`` (default) or ``"p90"`` across selected slices.

    Returns:
        Dict with keys ``"ich"``, ``"fracture"``, ``"mls_model"``,
        ``"device"``, ``"ich_strategy"`` and the MLS inference knobs.
    """
    models_path = Path(models_dir)
    device_obj = _resolve_device(device)

    logger.info("Loading models from %s ...", models_path.resolve())

    # --- ICH segmentation (MONAI) ------------------------------------------
    ich_model = _load_ich_monai(models_path, device_obj)
    logger.info("ICH model loaded (MONAI)")

    # --- Fracture detection (YOLO) -----------------------------------------
    from ultralytics import YOLO

    yolo_path = models_path / "yolo" / "best.pt"
    if not yolo_path.exists():
        raise FileNotFoundError(f"No YOLO model found at {yolo_path}")
    fracture_predictor = YOLO(str(yolo_path))

    # --- MLS (strict checkpoint-defined runtime) ---------------------------
    mls_model, mls_config = _load_mls_checkpoint(models_path, device_obj)
    logger.info(
        "MLS model: heatmap (backbone=%s, input_channels=%d, selector=%s)",
        mls_config.get("backbone", "hrnet_w18"),
        int(mls_config.get("input_channels", 3)),
        bool(mls_config.get("use_selector", False)),
    )
    return {
        "ich": ich_model,
        "fracture": fracture_predictor,
        "mls_model": mls_model,
        "device": device_obj,
        "ich_strategy": "monai",
        **_mls_runtime_options(
            mls_config, min_peak=mls_min_peak, top_k=mls_top_k,
            batch_size=mls_batch_size, aggregation=mls_aggregation,
        ),
        "calibration": _load_calibration(models_path),
    }


def predict(study_dir: str, models: dict = None) -> Dict[str, float]:
    """Run model inference on a single study.

    Args:
        study_dir: Path to a directory containing ``*.dcm`` files.
        models: Dict returned by :func:`load_models`. If ``None``, models are
            loaded from the default ``models/`` directory.

    Returns:
        Dict with exactly the 7 intermediate keys:
        ``V_EDH``, ``V_SDH``, ``V_IPH``, ``V_SAH``, ``V_IVH`` (mL),
        ``fracture_prob`` (0-1), ``MLS_mm`` (mm).
    """
    if models is None:
        models = load_models()

    device = models["device"]

    # 1. Read DICOM
    reader = DicomReader(study_dir).load_and_sort()
    vol_hu = reader.get_3d_volume_hu()  # (H, W, D)

    # 2. ICH volumes (MONAI)
    volumes = _predict_ich_monai(models["ich"], reader, device)

    # 3. Fracture probability (YOLO)
    fracture_prob = _predict_fracture(vol_hu, models["fracture"], device)

    # 4. MLS (heatmap strategy only)
    mls_mm = _predict_mls_heatmap(
        vol_hu=vol_hu,
        heatmap_model=models["mls_model"],
        spacing_x=reader.metadata["spacing_x"],
        device=device,
        batch_size=models.get("mls_batch_size", MLS_BATCH_SIZE),
        min_peak=models.get("mls_min_peak", MLS_MIN_PEAK),
        top_k=models.get("mls_top_k", MLS_TOP_K),
        aggregation=models.get("mls_aggregation", MLS_AGGREGATION),
        use_selector=models.get("mls_use_selector", False),
        selector_head_mode=models.get("mls_selector_head_mode", "single"),
        selector_threshold=models.get("mls_selector_threshold", 0.5),
        selector_relative_ratio=models.get("mls_selector_relative_ratio", 0.3),
        aggregation_quantile=models.get("mls_aggregation_quantile", 0.75),
        aggregation_probability_weighted=models.get("mls_aggregation_probability_weighted", False),
        anchor_window_radius=models.get("mls_anchor_window_radius", 2),
        min_active_slices=models.get("mls_min_active_slices", 1),
        heatmap_guard_ratio=models.get("mls_heatmap_guard_ratio", 0.0),
        negative_value_mm=models.get("mls_negative_value_mm", 0.1),
    )

    return _finalize_intermediates({
        **volumes,
        "fracture_prob": float(fracture_prob),
        "MLS_mm": mls_mm,
    }, models.get("calibration"),
        preserve_locked_mls=models.get("mls_locked_runtime", False),
    )


# ---------------------------------------------------------------------------
# MLS-only helpers (used by the task-specific leaderboards)
# ---------------------------------------------------------------------------

def load_mls_models(
    models_dir: str = "models",
    device: str = "auto",
    mls_min_peak: float = MLS_MIN_PEAK,
    mls_top_k: Optional[int] = MLS_TOP_K,
    mls_batch_size: int = MLS_BATCH_SIZE,
    mls_aggregation: str = MLS_AGGREGATION,
) -> dict:
    """Load ONLY the MLS heatmap model (no ICH / fracture models)."""
    models_path = Path(models_dir)
    device_obj = _resolve_device(device)

    mls_model, mls_config = _load_mls_checkpoint(models_path, device_obj)

    return {
        "mls_model": mls_model,
        "device": device_obj,
        **_mls_runtime_options(
            mls_config, min_peak=mls_min_peak, top_k=mls_top_k,
            batch_size=mls_batch_size, aggregation=mls_aggregation,
        ),
    }


def predict_mls_only(study_dir: str, mls_models: dict = None) -> float:
    """Run ONLY the MLS estimation for a single study (no ICH / fracture)."""
    if mls_models is None:
        mls_models = load_mls_models()

    device = mls_models["device"]
    reader = DicomReader(study_dir).load_and_sort()
    vol_hu = reader.get_3d_volume_hu()

    return _predict_mls_heatmap(
        vol_hu=vol_hu,
        heatmap_model=mls_models["mls_model"],
        spacing_x=reader.metadata["spacing_x"],
        device=device,
        batch_size=mls_models.get("mls_batch_size", MLS_BATCH_SIZE),
        min_peak=mls_models.get("mls_min_peak", MLS_MIN_PEAK),
        top_k=mls_models.get("mls_top_k", MLS_TOP_K),
        aggregation=mls_models.get("mls_aggregation", MLS_AGGREGATION),
        use_selector=mls_models.get("mls_use_selector", False),
        selector_head_mode=mls_models.get("mls_selector_head_mode", "single"),
        selector_threshold=mls_models.get("mls_selector_threshold", 0.5),
        selector_relative_ratio=mls_models.get("mls_selector_relative_ratio", 0.3),
        aggregation_quantile=mls_models.get("mls_aggregation_quantile", 0.75),
        aggregation_probability_weighted=mls_models.get("mls_aggregation_probability_weighted", False),
        anchor_window_radius=mls_models.get("mls_anchor_window_radius", 2),
        min_active_slices=mls_models.get("mls_min_active_slices", 1),
        heatmap_guard_ratio=mls_models.get("mls_heatmap_guard_ratio", 0.0),
        negative_value_mm=mls_models.get("mls_negative_value_mm", 0.1),
    )
