"""
model.py — Competition Model API for IAAA 2026 Brain CT Triage Challenge.

This module implements the standard Model API required by the competition.
It is self-contained so it can be safely included inside the submission zip.

Required API:
    predict(study_dir: str) -> dict
        Returns intermediates with exactly 7 keys:
            V_EDH, V_SDH, V_IPH, V_SAH, V_IVH (hemorrhage volumes in mL)
            fracture_prob (float 0-1)
            MLS_mm (midline shift in mm)

ICH Strategy Support:
    load_models(models_dir, ich_strategy="nnunet")
        Supported strategies: "nnunet", "smp", "monai", "yolo_seg"

Usage:
    from model import load_models, predict

    models = load_models("models", ich_strategy="smp")
    intermediates = predict("/path/to/dicom/study", models)
    triage_class = triage_from_intermediates(intermediates)  # 0, 1, or 2
"""

import os
import glob
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import pydicom
import nibabel as nib

logger = logging.getLogger(__name__)

# Supported ICH strategies
ICH_STRATEGIES = ("nnunet", "smp", "monai", "yolo_seg")

# ===========================================================================
# 1.  Self-contained DICOM Reader (no dependency on src/)
# ===========================================================================

WINDOWS = {
    "brain":   {"width": 80,  "level": 40},
    "subdural":{"width": 200, "level": 80},
    "bone":    {"width": 1000,"level": 400},
}

# ICH label mapping (same as src/config.py)
ICH_LABELS = {"background": 0, "IVH": 1, "IPH": 2, "SDH": 3, "EDH": 4, "SAH": 5}
ICH_LABEL_NAMES = {v: k for k, v in ICH_LABELS.items()}
NUM_ICH_CLASSES = len(ICH_LABELS)


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
        """Return full 3D volume in Hounsfield Units (H, W, D)."""
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
# 2.  MLS model architectures (lightweight copies from src/training/)
# ===========================================================================

class _SliceSelectorModel(torch.nn.Module):
    """Select best slice for MLS measurement (ResNet18-like)."""
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None, num_classes=1000)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


class _KeypointModel(torch.nn.Module):
    """Predict 3 keypoints (6 coords) for MLS calculation (ResNet34-like).

    Matches the training architecture:
        ResNet34 backbone → fc: Linear(512→256) → ReLU → Dropout → Linear(256→6)
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet34
        self.backbone = resnet34(weights=None, num_classes=1000)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 6),
        )

    def forward(self, x):
        return self.backbone(x)


# ===========================================================================
# 3.  ICH Strategy-specific model loaders
# ===========================================================================

def _clean_state_dict(state_dict):
    """Remove 'model.' prefix from Lightning checkpoint keys."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned[key[6:]] = value
        else:
            cleaned[key] = value
    return cleaned


def _create_3channel_window(hu_image: np.ndarray):
    """Stack brain + subdural + bone windows into a 3-channel array."""
    ch1 = apply_windowing(hu_image, WINDOWS["brain"]["width"], WINDOWS["brain"]["level"])
    ch2 = apply_windowing(hu_image, WINDOWS["subdural"]["width"], WINDOWS["subdural"]["level"])
    ch3 = apply_windowing(hu_image, WINDOWS["bone"]["width"], WINDOWS["bone"]["level"])
    return np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)


def _calculate_mls(coords_pixels, spacing_x):
    x1, y1, x2, y2, x3, y3 = coords_pixels
    num = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return (num / den if den > 0 else 0.0) * spacing_x


# ── ICH Loaders ───────────────────────────────────────────────────

def _load_ich_nnunet(models_path: Path, device: torch.device):
    """Load nnU-Net predictor for ICH segmentation."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
    )
    nnunet_model_path = str(models_path / "nnunet")
    predictor.initialize_from_trained_model_folder(
        nnunet_model_path,
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )
    return predictor


def _load_ich_smp(models_path: Path, device: torch.device):
    """Load SMP model for ICH segmentation."""
    import segmentation_models_pytorch as smp

    ckpt_path = models_path / "smp" / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = list((models_path / "smp").glob("*.ckpt"))
        if ckpt_path:
            ckpt_path = ckpt_path[0]
        else:
            raise FileNotFoundError(f"No SMP checkpoint found in {models_path / 'smp'}")

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    hparams = state.get("hyper_parameters", {})

    arch = hparams.get("architecture", "Unet")
    encoder = hparams.get("encoder", "resnet34")

    model = smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=1,
        classes=NUM_ICH_CLASSES,
    )
    if "state_dict" in state:
        state = state["state_dict"]
    # Use strict=False — Lightning checkpoints include keys from loss/metric
    # objects (dice_loss, ce_loss, train_iou, val_iou) that aren't part of
    # the bare SMP model. We only care about the model weights.
    model.load_state_dict(_clean_state_dict(state), strict=False)
    return model.to(device).eval()


def _load_ich_monai(models_path: Path, device: torch.device):
    """Load MONAI model for ICH segmentation."""
    from monai.networks.nets import UNETR, SwinUNETR, SegResNet, DynUNet

    ckpt_path = models_path / "monai" / "best.pth"
    if not ckpt_path.exists():
        candidates = list((models_path / "monai").glob("*.pth"))
        if candidates:
            ckpt_path = candidates[0]
        else:
            raise FileNotFoundError(f"No MONAI checkpoint found in {models_path / 'monai'}")

    # Infer model type from checkpoint filename
    fname = ckpt_path.stem.lower()
    if "swin" in fname:
        model = SwinUNETR(
            in_channels=1, out_channels=NUM_ICH_CLASSES,
            patch_size=(2, 2, 2),
            window_size=(7, 14, 14),
            feature_size=48, use_checkpoint=False,
        )
    elif "segresnet" in fname:
        model = SegResNet(
            spatial_dims=3, in_channels=1, out_channels=NUM_ICH_CLASSES,
            init_filters=16, blocks_down=(1, 2, 2, 4), dropout_prob=0.1,
        )
    elif "dynunet" in fname:
        model = DynUNet(
            spatial_dims=3,
            in_channels=1, out_channels=NUM_ICH_CLASSES,
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


def _load_ich_yolo_seg(models_path: Path, device: torch.device):
    """Load YOLO segmentation model for ICH."""
    from ultralytics import YOLO

    yolo_path = models_path / "yolo_seg" / "best.pt"
    if not yolo_path.exists():
        raise FileNotFoundError(f"No YOLO seg model found at {yolo_path}")
    return YOLO(str(yolo_path))


# ── ICH Inference Helpers ─────────────────────────────────────────

def _predict_ich_nnunet(model, reader: DicomReader) -> dict:
    """Run nnU-Net inference and return ICH volumes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pid = reader.metadata["patient_id"]
        nifti_input = str(tmp_dir / f"{pid}_0000.nii.gz")
        reader.save_as_nifti(nifti_input)
        nifti_output = str(tmp_dir / f"{pid}.nii.gz")
        model.predict_from_files(
            [[nifti_input]], [nifti_output],
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=1, num_processes_segmentation_export=1,
        )
        mask_nii = nib.load(nifti_output)
        mask_data = mask_nii.get_fdata()
        voxel_vol_ml = np.prod(mask_nii.header.get_zooms()) / 1000.0
        return {
            "V_IVH": float(np.sum(mask_data == 1) * voxel_vol_ml),
            "V_IPH": float(np.sum(mask_data == 2) * voxel_vol_ml),
            "V_SDH": float(np.sum(mask_data == 3) * voxel_vol_ml),
            "V_EDH": float(np.sum(mask_data == 4) * voxel_vol_ml),
            "V_SAH": float(np.sum(mask_data == 5) * voxel_vol_ml),
        }


def _predict_ich_smp(model, reader: DicomReader, device: torch.device) -> dict:
    """Run SMP 2D slice-level inference and aggregate volumes."""
    vol_hu = reader.get_3d_volume_hu()  # (H, W, D)
    voxel_vol_ml = (
        reader.metadata["spacing_x"] *
        reader.metadata["spacing_y"] *
        reader.metadata["spacing_z"] / 1000.0
    )

    volumes = {"V_IVH": 0.0, "V_IPH": 0.0, "V_SDH": 0.0, "V_EDH": 0.0, "V_SAH": 0.0}

    for z in range(vol_hu.shape[2]):
        slice_hu = vol_hu[:, :, z]
        p_low, p_high = np.percentile(slice_hu[slice_hu > -900], [0.5, 99.5])
        img_norm = np.clip((slice_hu - p_low) / max(p_high - p_low, 1e-6), 0.0, 1.0)

        from PIL import Image as PILImage
        img_pil = PILImage.fromarray((img_norm * 255).astype(np.uint8))
        img_512 = np.array(img_pil.resize((512, 512), PILImage.BILINEAR), dtype=np.float32) / 255.0

        inp = torch.from_numpy(img_512).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        with torch.no_grad():
            logits = model(inp)  # (1, C, H, W)
            pred = logits.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

        # Scale back to original size for voxel count
        pred_pil = PILImage.fromarray(pred.astype(np.uint8))
        pred_orig = np.array(pred_pil.resize(
            (vol_hu.shape[1], vol_hu.shape[0]), PILImage.NEAREST,
        ))

        for label_id, name in ICH_LABEL_NAMES.items():
            if label_id == 0:
                continue
            key = f"V_{name}"
            volumes[key] += float(np.sum(pred_orig == label_id))

    # Convert voxel counts to mL
    for key in volumes:
        volumes[key] *= voxel_vol_ml

    return volumes


def _predict_ich_monai(model, reader: DicomReader, device: torch.device) -> dict:
    """Run MONAI 3D inference with training-matched preprocessing.

    Uses the same pipeline as training (Spacingd→1mm isotropic,
    CropForegroundd, ScaleIntensityRanged).

    Optimisation:
        - If the preprocessed volume is small (< 8M voxels), does a SINGLE
          full-volume forward pass (much faster on both CPU and GPU).
        - Otherwise falls back to sliding-window inference.

    .. important::
        Some MONAI models (e.g. SegResNet with ``blocks_down=(1,2,2,4)``)
        have a stride factor of 16. The input is **automatically padded** to
        the next multiple of 16 so that U-Net skip connections don't
        misalign. The output is cropped back before resize.

    Prediction is resized back to original DICOM space for accurate
    volume calculation.
    """
    import os
    import tempfile

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

    # 1. Save as NIfTI (proper orientation metadata for MONAI)
    with tempfile.TemporaryDirectory() as tmp:
        nifti_path = os.path.join(tmp, "input.nii.gz")
        reader.save_as_nifti(nifti_path)

        # 2. Preprocessing — matches monai/dataset.py training pipeline
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

        # 3. Pad to multiples of STRIDE (required for U-Net skip connections)
        pad_d = (STRIDE - orig_spatial[0] % STRIDE) % STRIDE
        pad_h = (STRIDE - orig_spatial[1] % STRIDE) % STRIDE
        pad_w = (STRIDE - orig_spatial[2] % STRIDE) % STRIDE

        needs_pad = (pad_d + pad_h + pad_w) > 0
        if needs_pad:
            inp = torch.nn.functional.pad(
                inp,
                (0, pad_w, 0, pad_h, 0, pad_d),  # (W_left, W_right, H_left, H_right, D_left, D_right)
                mode="replicate",
            )

        # 4. Choose inference strategy based on volume size
        MAX_VOXELS_FULL_VOLUME = 8_000_000  # ~200³ — fits most GPUs / fast CPUs
        use_sw = n_voxels > MAX_VOXELS_FULL_VOLUME

        if use_sw:
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
            # Fast path: single full-volume forward pass
            with torch.no_grad():
                pred_logits = model(inp)

        # 5. Crop back to original spatial size (undo padding)
        if needs_pad:
            ds, hs, ws = orig_spatial
            pred_logits = pred_logits[
                :, :,
                :ds,       # D
                :hs,       # H
                :ws,       # W
            ]

        pred_labels = pred_logits.argmax(dim=1).cpu()  # (1, D, H, W) in 1mm space

        # 6. Resize prediction back to original DICOM dimensions
        H_orig = reader.metadata["rows"]
        W_orig = reader.metadata["columns"]
        D_orig = len(reader)

        pred_orig = Resize(
            spatial_size=(H_orig, W_orig, D_orig), mode="nearest",
        )(pred_labels).squeeze().numpy().astype(np.int64)

    # 7. Calculate volumes using original voxel spacing
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


def _predict_ich_yolo_seg(model, reader: DicomReader, device: torch.device) -> dict:
    """Run YOLO segmentation inference and return ICH volumes."""
    vol_hu = reader.get_3d_volume_hu()
    voxel_vol_ml = (
        reader.metadata["spacing_x"] *
        reader.metadata["spacing_y"] *
        reader.metadata["spacing_z"] / 1000.0
    )

    volumes = {"V_IVH": 0.0, "V_IPH": 0.0, "V_SDH": 0.0, "V_EDH": 0.0, "V_SAH": 0.0}
    yolo_to_ich = {0: "IVH", 1: "IPH", 2: "SDH", 3: "EDH", 4: "SAH"}

    import cv2

    for z in range(vol_hu.shape[2]):
        slice_hu = vol_hu[:, :, z]
        p_low, p_high = np.percentile(slice_hu[slice_hu > -900], [0.5, 99.5])
        img_uint8 = np.clip(
            (slice_hu - p_low) / max(p_high - p_low, 1e-6) * 255, 0, 255,
        ).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        results = model.predict(img_rgb, device=device, verbose=False)
        for res in results:
            if res.masks is not None:
                for seg_mask, cls_id in zip(res.masks.data, res.boxes.cls):
                    cls_name = yolo_to_ich.get(int(cls_id.item()))
                    if cls_name:
                        mask_np = seg_mask.cpu().numpy()
                        # mask_np is normalized to original image size
                        pixel_count = np.sum(mask_np > 0.5)
                        volumes[f"V_{cls_name}"] += float(pixel_count)

    for key in volumes:
        volumes[key] *= voxel_vol_ml

    return volumes


# ===========================================================================
# 3b.  MLS Heatmap model + DARK decoding (new strategy — auto-detected)
# ===========================================================================

class _MLSHeatmapModel(torch.nn.Module):
    """Lightweight heatmap model using timm HRNet backbone.

    Falls back to ResNet34 if timm is not available (competition Docker).
    Predicts 3 Gaussian heatmap channels at 1/4 input resolution.
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

        if self._use_timm:
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(feat_dim, 64, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 3, kernel_size=1),
            )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_timm:
            features = self.backbone(x)
            feat = features[0]
        else:
            feat = self.backbone(x)
        return self.head(feat)


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


def _predict_mls_heatmap_pipeline(
    vol_hu: np.ndarray,
    slice_model: torch.nn.Module,
    heatmap_model: torch.nn.Module,
    spacing_x: float,
    device: torch.device,
    top_k: int = 3,
    aggregation: str = "max",
) -> float:
    """MLS prediction using heatmap model with Top-K slice aggregation.

    1. SliceSelector روی همه اسلایس‌ها → انتخاب top-K
    2. Heatmap مدل روی K اسلایس (batch)
    3. DARK decode → مختصات sub-pixel
    4. محاسبه MLS برای هر اسلایس
    5. Aggregation (max یا p90)
    """
    n_slices = vol_hu.shape[2]

    # 1. Slice selector on all slices
    slices_256 = []
    for z in range(n_slices):
        ch3 = _create_3channel_window(vol_hu[:, :, z])
        t = torch.tensor(ch3, dtype=torch.float32).unsqueeze(0)
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
        slices_256.append(t)

    batch = torch.cat(slices_256, dim=0).to(device)
    with torch.no_grad():
        slice_logits = slice_model(batch)

    k = min(top_k, n_slices)
    top_indices = torch.topk(slice_logits.squeeze(), k=k).indices.cpu().numpy()
    if top_indices.ndim == 0:
        top_indices = np.array([top_indices.item()])

    # 2. Prepare top-K slices for heatmap model
    batch_3ch = []
    for z in top_indices:
        ch3 = _create_3channel_window(vol_hu[:, :, z])
        batch_3ch.append(ch3)

    inp = torch.from_numpy(np.stack(batch_3ch, axis=0)).float().to(device)

    # 3. Forward and DARK decode
    with torch.no_grad():
        heatmap_pred = heatmap_model(inp)

    heatmap_size = heatmap_pred.shape[-1]
    img_size = vol_hu.shape[0]
    coords = _decode_heatmap_dark_batch(heatmap_pred.cpu(), heatmap_size, img_size)

    # 4. Compute MLS per slice
    mls_values: list[float] = []
    for i in range(len(coords)):
        kps = coords[i]
        if (kps[:, 0] >= 0).all():
            mls_values.append(float(_calculate_mls(kps.ravel(), spacing_x)))

    if not mls_values:
        return 0.0

    mls_arr = np.array(mls_values)
    if aggregation == "p90":
        return float(np.percentile(mls_arr, 90))
    return float(mls_arr.max())  # default: max (conservative)


# ===========================================================================
# 4.  Public API
# ===========================================================================

_loaded_models = None


def load_models(
    models_dir: str = "models",
    device: str = "cuda",
    ich_strategy: str = "nnunet",
) -> dict:
    """Load all trained models from the model directory.

    Auto-detects the MLS estimation method:
    - If ``models/mls_heatmap/mls_heatmap_best.pth`` exists → uses HRNet
      heatmap pipeline (DARK decoding + Top-K aggregation).
    - Otherwise → falls back to legacy keypoint regression pipeline.

    Expected directory layout (relative to *models_dir*)::

        models/
        ├── nnunet/                  # nnU-Net ICH model (for ich_strategy="nnunet")
        │   ├── checkpoint_best.pth
        │   ├── dataset.json
        │   ├── plans.json
        │   └── dataset_fingerprint.json
        ├── smp/                     # SMP ICH model (for ich_strategy="smp")
        │   └── best.ckpt
        ├── monai/                   # MONAI ICH model (for ich_strategy="monai")
        │   └── best.pth
        ├── yolo_seg/                # YOLO Seg ICH model (for ich_strategy="yolo_seg")
        │   └── best.pt
        ├── yolo/                    # Fracture detection model
        │   └── best.pt
        ├── mls/                     # Legacy MLS pipeline (always needed for slice selector)
        │   ├── slice_selector_best.ckpt
        │   └── keypoint_best.ckpt   # Only used if mls_heatmap/ not present
        └── mls_heatmap/             # [NEW] Heatmap-based MLS pipeline
            └── mls_heatmap_best.pth

    Args:
        models_dir: Path to the ``models`` directory.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        ich_strategy: ICH model strategy to use
            (``"nnunet"``, ``"smp"``, ``"monai"``, ``"yolo_seg"``).

    Returns:
        A dict with keys ``"ich"``, ``"fracture"``, ``"mls_slice"``,
        ``"mls_model"``, ``"mls_mode"``, and metadata ``"ich_strategy"``, ``"device"``.
    """
    global _loaded_models

    if ich_strategy not in ICH_STRATEGIES:
        raise ValueError(
            f"Unknown ICH strategy: '{ich_strategy}'. "
            f"Choose from: {ICH_STRATEGIES}"
        )

    models_path = Path(models_dir)
    device_obj = torch.device(device)

    logger.info(
        "Loading models from %s [ICH strategy: %s]",
        models_path.resolve(), ich_strategy,
    )

    # --- ICH model (strategy-dependent) -----------------------------------
    ich_loaders = {
        "nnunet": _load_ich_nnunet,
        "smp": _load_ich_smp,
        "monai": _load_ich_monai,
        "yolo_seg": _load_ich_yolo_seg,
    }
    ich_model = ich_loaders[ich_strategy](models_path, device_obj)
    logger.info("ICH model loaded via '%s' strategy", ich_strategy)

    # --- YOLO (fracture detection) -----------------------------------------
    from ultralytics import YOLO

    yolo_path = str(models_path / "yolo" / "best.pt")
    fracture_predictor = YOLO(yolo_path)

    # --- MLS (midline shift) -----------------------------------------------
    # Slice selector (always needed — even for heatmap pipeline)
    slice_ckpt = models_path / "mls" / "slice_selector_best.ckpt"
    slice_model = _SliceSelectorModel()
    state = torch.load(str(slice_ckpt), map_location=device_obj, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    slice_model.load_state_dict(_clean_state_dict(state))
    slice_model = slice_model.to(device_obj).eval()

    # Auto-detect: new heatmap model OR old keypoint model
    heatmap_ckpt = models_path / "mls_heatmap" / "mls_heatmap_best.pth"
    mls_mode = "heatmap"

    if heatmap_ckpt.exists():
        # ── New heatmap pipeline ────────────────────────────────────────
        logger.info("MLS mode: heatmap (HRNet + DARK) — found %s", heatmap_ckpt)
        checkpoint = torch.load(str(heatmap_ckpt), map_location=device_obj, weights_only=False)
        sd = checkpoint.get("model_state_dict", checkpoint)
        # Try to load with timm backbone first
        hm_model = _MLSHeatmapModel(backbone_name="hrnet_w18", in_channels=3)
        try:
            hm_model.load_state_dict(sd, strict=False)
        except Exception:
            # Fallback: load without strict matching
            hm_model.load_state_dict(sd, strict=False)
        hm_model = hm_model.to(device_obj).eval()
        mls_model = hm_model
        mls_mode = "heatmap"
    else:
        # ── Legacy keypoint pipeline ────────────────────────────────────
        logger.info("MLS mode: legacy (ResNet keypoint regression) — %s not found", heatmap_ckpt)
        kp_ckpt = models_path / "mls" / "keypoint_best.ckpt"
        kp_model = _KeypointModel()
        state = torch.load(str(kp_ckpt), map_location=device_obj, weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        kp_model.load_state_dict(_clean_state_dict(state))
        kp_model = kp_model.to(device_obj).eval()
        mls_model = kp_model
        mls_mode = "legacy"

    _loaded_models = {
        "ich": ich_model,
        "fracture": fracture_predictor,
        "mls_slice": slice_model,
        "mls_model": mls_model,
        "mls_mode": mls_mode,
        "ich_strategy": ich_strategy,
        "device": device_obj,
    }
    logger.info("All models loaded successfully.")
    return _loaded_models


def predict(study_dir: str, models: dict = None) -> Dict[str, float]:
    """Run model inference on a single study.

    Args:
        study_dir: Path to directory containing ``*.dcm`` files for one study.
        models: Dict returned by :func:`load_models`. If ``None``, loads
                models from the default ``models/`` directory.

    Returns:
        Dictionary with exactly 7 intermediate keys:

        - ``V_EDH``, ``V_SDH``, ``V_IPH``, ``V_SAH``, ``V_IVH``: hemorrhage
          volumes in mL.
        - ``fracture_prob``: probability of skull fracture in [0, 1].
        - ``MLS_mm``: midline shift magnitude in mm.

    Raises:
        Various exceptions on invalid input or model failure.
    """
    if models is None:
        models = load_models()

    device = models.get("device", torch.device("cuda"))
    ich_strategy = models.get("ich_strategy", "nnunet")

    # --- 1. Read DICOM ----------------------------------------------------
    reader = DicomReader(study_dir).load_and_sort()
    vol_hu = reader.get_3d_volume_hu()  # (H, W, D)

    # --- 2. ICH volumes (strategy-dependent) ------------------------------
    ich_predictors = {
        "nnunet": _predict_ich_nnunet,
        "smp": _predict_ich_smp,
        "monai": _predict_ich_monai,
        "yolo_seg": _predict_ich_yolo_seg,
    }
    predictor_fn = ich_predictors.get(ich_strategy)
    if predictor_fn is None:
        raise ValueError(f"Missing ICH predictor for strategy '{ich_strategy}'")

    if ich_strategy == "nnunet":
        volumes = predictor_fn(models["ich"], reader)
    else:
        volumes = predictor_fn(models["ich"], reader, device)

    # --- 3. Fracture probability via YOLO ----------------------------------
    max_fracture_conf = 0.0
    for z in range(vol_hu.shape[2]):
        slice_hu = vol_hu[:, :, z]
        bone_img = apply_windowing(slice_hu,
                                   WINDOWS["bone"]["width"],
                                   WINDOWS["bone"]["level"])
        bone_img_8bit = (bone_img * 255).astype(np.uint8)

        import cv2
        img_rgb = cv2.cvtColor(bone_img_8bit, cv2.COLOR_GRAY2RGB)
        results = models["fracture"].predict(img_rgb, device=device, verbose=False)
        for res in results:
            if res.boxes is not None and len(res.boxes) > 0:
                conf = float(res.boxes.conf.max())
                if conf > max_fracture_conf:
                    max_fracture_conf = conf

    # --- 4. MLS estimation --------------------------------------------------
    mls_mode = models.get("mls_mode", "legacy")
    spacing_x = reader.metadata["spacing_x"]

    if mls_mode == "heatmap":
        # ── New: heatmap pipeline with Top-K aggregation ─────────────────
        mls_mm = _predict_mls_heatmap_pipeline(
            vol_hu=vol_hu,
            slice_model=models["mls_slice"],
            heatmap_model=models["mls_model"],
            spacing_x=spacing_x,
            device=device,
            top_k=3,
            aggregation="max",
        )
    else:
        # ── Legacy: single-slice keypoint regression ─────────────────────
        slices_256 = []
        for z in range(vol_hu.shape[2]):
            ch3 = _create_3channel_window(vol_hu[:, :, z])
            t = torch.tensor(ch3, dtype=torch.float32).unsqueeze(0)
            t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
            slices_256.append(t)
        batch = torch.cat(slices_256, dim=0).to(device)
        with torch.no_grad():
            slice_logits = models["mls_slice"](batch)
        best_z = int(torch.argmax(slice_logits).item())

        kp_input_3ch = _create_3channel_window(vol_hu[:, :, best_z])
        kp_input_t = (
            torch.tensor(kp_input_3ch, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            coords_norm = models["mls_model"](kp_input_t).squeeze().cpu().numpy()
        coords_pixels = coords_norm * 512.0
        mls_mm = float(_calculate_mls(coords_pixels, spacing_x))

    # --- 5. Assemble result ------------------------------------------------
    result = {
        **volumes,
        "fracture_prob": max_fracture_conf,
        "MLS_mm": mls_mm,
    }
    return result

