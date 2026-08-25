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
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
MLS_MIN_PEAK = 0.9        # minimum peak of all 3 keypoints to trust a slice
MLS_TOP_K = None          # if set, keep top-K confident slices instead of threshold
MLS_AGGREGATION = "max"   # "max" or "p90" across selected slices
MLS_BATCH_SIZE = 16       # slices per heatmap forward pass


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

    def __init__(self, in_channels: int, num_keypoints: int = 3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(64, num_keypoints, kernel_size=1)

        # Small-init the final conv (same as training) for safety on partial loads.
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.001)
        if self.conv2.bias is not None:
            torch.nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class _MLSHeatmapModel(torch.nn.Module):
    """HRNet (timm) backbone + heatmap head — matches the trained checkpoint.

    ``backbone`` and ``input_channels`` are read from the checkpoint's saved
    ``config`` so any trained variant (hrnet_w32 / hrnet_w18, 1 or 3 input
    channels) is reconstructed correctly. The first convolution is adapted
    when the checkpoint was trained with != 3 channels.
    """

    def __init__(self, backbone_name: str = "hrnet_w18", in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self._use_timm = False

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
            self.head = _MLSHeatmapHead(feat_dim, num_keypoints=3)
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
) -> float:
    """Predict MLS with the heatmap model only.

    Pipeline (no SliceSelector — it was removed with the legacy strategy):

    1. Window every slice and batch them through the HRNet heatmap model.
    2. DARK-decode the 3 keypoints per slice and compute per-slice MLS.
    3. Keep only confident slices — either the ``top_k`` slices with the
       highest minimum keypoint peak, or (default) every slice whose minimum
       peak is >= ``min_peak``. This emulates the removed slice selector:
       the model produces reliable keypoints on target-like slices and the
       per-slice MLS values there are trustworthy.
    4. Aggregate across the selected slices with ``max`` (competition ground
       truth is the max over annotated slices) or ``p90``.
    """
    in_channels = getattr(heatmap_model, "in_channels", 3)
    n_slices = vol_hu.shape[2]

    # If the native slice is not ML_INPUT_SIZE, resize to match training and
    # rescale the mm-per-pixel factor accordingly (aspect preserved, square CT).
    orig_h = int(vol_hu.shape[0])
    eff_spacing = spacing_x * (orig_h / ML_INPUT_SIZE)

    candidates: List[tuple[float, float, float]] = []  # (mls, min_peak, mean_peak)

    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        batch_imgs = []
        for z in range(start, end):
            win = _create_windowed_input(vol_hu[:, :, z], in_channels)
            t = torch.from_numpy(win).float().unsqueeze(0)  # (1, C, H, W)
            if win.shape[1:] != (ML_INPUT_SIZE, ML_INPUT_SIZE):
                t = F.interpolate(
                    t, size=(ML_INPUT_SIZE, ML_INPUT_SIZE),
                    mode="bilinear", align_corners=False,
                )
            batch_imgs.append(t)

        inp = torch.cat(batch_imgs, dim=0).to(device)  # (B, C, 512, 512)
        with torch.no_grad():
            heatmaps = heatmap_model(inp)

        peaks = heatmaps.amax(dim=(2, 3)).cpu().numpy()  # (B, 3)
        coords = _decode_heatmap_dark_batch(
            heatmaps.cpu(), heatmaps.shape[-1], ML_INPUT_SIZE,
        )  # (B, 3, 2)

        for i in range(len(coords)):
            if (coords[i, :, 0] >= 0).all():
                mls = _calculate_mls(coords[i].ravel(), eff_spacing)
                candidates.append(
                    (mls, float(peaks[i].min()), float(peaks[i].mean()))
                )

    if not candidates:
        logger.warning("No valid MLS measurements for this study. Returning 0.0.")
        return 0.0

    if top_k is not None and top_k > 0:
        # Keep the top-K most confident slices (all candidates if fewer).
        selected = sorted(candidates, key=lambda c: -c[1])[:top_k]
    else:
        selected = [c for c in candidates if c[1] >= min_peak]
        if not selected:
            # Nothing confident enough — fall back to the most confident slice.
            selected = [max(candidates, key=lambda c: c[1])]

    mls_values = np.array([c[0] for c in selected])
    if aggregation == "p90":
        return float(np.percentile(mls_values, 90))
    return float(mls_values.max())


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

    # --- MLS (heatmap strategy only) ---------------------------------------
    heatmap_ckpt = models_path / "mls_heatmap" / "mls_heatmap_best.pth"
    if not heatmap_ckpt.exists():
        raise FileNotFoundError(
            f"MLS heatmap checkpoint not found at {heatmap_ckpt}. "
            "This submission ships the heatmap MLS strategy only."
        )
    checkpoint = torch.load(str(heatmap_ckpt), map_location=device_obj, weights_only=False)
    sd = checkpoint.get("model_state_dict", checkpoint)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    backbone = cfg.get("backbone", "hrnet_w18")
    in_channels = cfg.get("input_channels", 3)
    logger.info(
        "MLS model: heatmap (backbone=%s, input_channels=%d)", backbone, in_channels,
    )
    mls_model = _MLSHeatmapModel(backbone_name=backbone, in_channels=in_channels)
    mls_model.load_state_dict(sd, strict=False)
    mls_model = mls_model.to(device_obj).eval()

    return {
        "ich": ich_model,
        "fracture": fracture_predictor,
        "mls_model": mls_model,
        "device": device_obj,
        "ich_strategy": "monai",
        "mls_min_peak": float(mls_min_peak),
        "mls_top_k": mls_top_k,
        "mls_batch_size": int(mls_batch_size),
        "mls_aggregation": mls_aggregation,
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
    )

    return {
        **volumes,
        "fracture_prob": float(fracture_prob),
        "MLS_mm": mls_mm,
    }


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

    heatmap_ckpt = models_path / "mls_heatmap" / "mls_heatmap_best.pth"
    if not heatmap_ckpt.exists():
        raise FileNotFoundError(f"MLS heatmap checkpoint not found at {heatmap_ckpt}")

    checkpoint = torch.load(str(heatmap_ckpt), map_location=device_obj, weights_only=False)
    sd = checkpoint.get("model_state_dict", checkpoint)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    mls_model = _MLSHeatmapModel(
        backbone_name=cfg.get("backbone", "hrnet_w18"),
        in_channels=cfg.get("input_channels", 3),
    )
    mls_model.load_state_dict(sd, strict=False)
    mls_model = mls_model.to(device_obj).eval()

    return {
        "mls_model": mls_model,
        "device": device_obj,
        "mls_min_peak": float(mls_min_peak),
        "mls_top_k": mls_top_k,
        "mls_batch_size": int(mls_batch_size),
        "mls_aggregation": mls_aggregation,
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
    )
