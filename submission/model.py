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
            patch_size=(2, 4, 4),
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
    CropForegroundd, ScaleIntensityRanged) and sliding-window inference.
    Prediction is resized back to original DICOM space for accurate
    volume calculation.
    """
    import os
    import tempfile

    from monai.inferers import sliding_window_inference
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

        # 3. Sliding-window inference (handles any volume size)
        roi_size = (128, 128, 128)
        with torch.no_grad():
            pred_logits = sliding_window_inference(
                inputs=inp,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
        pred_labels = pred_logits.argmax(dim=1).cpu()  # (1, D, H, W) in 1mm space

        # 4. Resize prediction back to original DICOM dimensions
        H_orig = reader.metadata["rows"]
        W_orig = reader.metadata["columns"]
        D_orig = len(reader)

        pred_orig = Resize(
            spatial_size=(H_orig, W_orig, D_orig), mode="nearest",
        )(pred_labels).squeeze().numpy().astype(np.int64)

    # 5. Calculate volumes using original voxel spacing
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
# 4.  Public API
# ===========================================================================

_loaded_models = None


def load_models(
    models_dir: str = "models",
    device: str = "cuda",
    ich_strategy: str = "nnunet",
) -> dict:
    """Load all trained models from the model directory.

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
        └── mls/
            ├── slice_selector_best.ckpt
            └── keypoint_best.ckpt

    Args:
        models_dir: Path to the ``models`` directory.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        ich_strategy: ICH model strategy to use
            (``"nnunet"``, ``"smp"``, ``"monai"``, ``"yolo_seg"``).

    Returns:
        A dict with keys ``"ich"``, ``"fracture"``, ``"mls"``,
        and metadata ``"ich_strategy"``, ``"device"``.
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
    # Slice selector
    slice_ckpt = models_path / "mls" / "slice_selector_best.ckpt"
    slice_model = _SliceSelectorModel()
    state = torch.load(str(slice_ckpt), map_location=device_obj, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    slice_model.load_state_dict(_clean_state_dict(state))
    slice_model = slice_model.to(device_obj).eval()

    # Keypoint detector
    kp_ckpt = models_path / "mls" / "keypoint_best.ckpt"
    kp_model = _KeypointModel()
    state = torch.load(str(kp_ckpt), map_location=device_obj, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    kp_model.load_state_dict(_clean_state_dict(state))
    kp_model = kp_model.to(device_obj).eval()

    _loaded_models = {
        "ich": ich_model,
        "fracture": fracture_predictor,
        "mls_slice": slice_model,
        "mls_kp": kp_model,
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

    # --- 4. MLS via custom CNN --------------------------------------------
    # 4a. Slice selector: find best slice
    slices_256 = []
    for z in range(vol_hu.shape[2]):
        ch3 = _create_3channel_window(vol_hu[:, :, z])
        t = torch.tensor(ch3, dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
        slices_256.append(t)
    batch = torch.cat(slices_256, dim=0).to(device)
    with torch.no_grad():
        slice_logits = models["mls_slice"](batch)
    best_z = int(torch.argmax(slice_logits).item())

    # 4b. Keypoint detection on best slice
    kp_input_3ch = _create_3channel_window(vol_hu[:, :, best_z])
    kp_input_t = (
        torch.tensor(kp_input_3ch, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        coords_norm = models["mls_kp"](kp_input_t).squeeze().cpu().numpy()
    coords_pixels = coords_norm * 512.0

    # 4c. Calculate MLS in mm
    mls_mm = float(_calculate_mls(coords_pixels, reader.metadata["spacing_x"]))

    # --- 5. Assemble result ------------------------------------------------
    result = {
        **volumes,
        "fracture_prob": max_fracture_conf,
        "MLS_mm": mls_mm,
    }
    return result

