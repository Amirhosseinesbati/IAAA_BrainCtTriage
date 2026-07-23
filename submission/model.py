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

Usage:
    from model import load_models, predict

    models = load_models("models")
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

# ===========================================================================
# 1.  Self-contained DICOM Reader (no dependency on src/)
# ===========================================================================

WINDOWS = {
    "brain":   {"width": 80,  "level": 40},
    "subdural":{"width": 200, "level": 80},
    "bone":    {"width": 1000,"level": 400},
}


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
# 3.  Model loaders
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


# ===========================================================================
# 4.  Public API
# ===========================================================================

_loaded_models = None


def load_models(models_dir: str = "models", device: str = "cuda") -> dict:
    """Load all three trained models from the model directory.

    Expected directory layout (relative to *models_dir*)::

        models/
        ├── nnunet/
        │   └── checkpoint_best.pth
        ├── yolo/
        │   └── best.pt
        └── mls/
            ├── slice_selector_best.ckpt
            └── keypoint_best.ckpt

    Args:
        models_dir: Path to the ``models`` directory.
        device: Torch device string (``"cuda"`` or ``"cpu"``).

    Returns:
        A dict with keys ``"ich"``, ``"fracture"``, ``"mls"``.
    """
    global _loaded_models

    models_path = Path(models_dir)
    device_obj = torch.device(device)

    logger.info("Loading models from %s", models_path.resolve())

    # --- nnU-Net (ICH segmentation) ---------------------------------------
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    ich_predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device_obj,
        verbose=False,
    )
    nnunet_model_path = str(models_path / "nnunet")
    ich_predictor.initialize_from_trained_model_folder(
        nnunet_model_path,
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )

    # --- YOLO (fracture detection) -----------------------------------------
    from ultralytics import YOLO

    yolo_path = str(models_path / "yolo" / "best.pt")
    fracture_predictor = YOLO(yolo_path)

    # --- MLS (midline shift) -----------------------------------------------
    # Slice selector
    slice_ckpt = models_path / "mls" / "slice_selector_best.ckpt"
    slice_model = _SliceSelectorModel()
    state = torch.load(str(slice_ckpt), map_location=device_obj)
    if "state_dict" in state:
        state = state["state_dict"]
    slice_model.load_state_dict(_clean_state_dict(state))
    slice_model = slice_model.to(device_obj).eval()

    # Keypoint detector
    kp_ckpt = models_path / "mls" / "keypoint_best.ckpt"
    kp_model = _KeypointModel()
    state = torch.load(str(kp_ckpt), map_location=device_obj)
    if "state_dict" in state:
        state = state["state_dict"]
    kp_model.load_state_dict(_clean_state_dict(state))
    kp_model = kp_model.to(device_obj).eval()

    _loaded_models = {
        "ich": ich_predictor,
        "fracture": fracture_predictor,
        "mls_slice": slice_model,
        "mls_kp": kp_model,
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

    # --- 1. Read DICOM ----------------------------------------------------
    reader = DicomReader(study_dir).load_and_sort()
    vol_hu = reader.get_3d_volume_hu()  # (H, W, D)

    # --- 2. ICH volumes via nnU-Net ---------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pid = reader.metadata["patient_id"]

        nifti_input = str(tmp_dir / f"{pid}_0000.nii.gz")
        reader.save_as_nifti(nifti_input)

        nifti_output = str(tmp_dir / f"{pid}.nii.gz")
        models["ich"].predict_from_files(
            [[nifti_input]],
            [nifti_output],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        mask_nii = nib.load(nifti_output)
        mask_data = mask_nii.get_fdata()
        voxel_vol_ml = np.prod(mask_nii.header.get_zooms()) / 1000.0

        volumes = {
            "V_IVH": float(np.sum(mask_data == 1) * voxel_vol_ml),
            "V_IPH": float(np.sum(mask_data == 2) * voxel_vol_ml),
            "V_SDH": float(np.sum(mask_data == 3) * voxel_vol_ml),
            "V_EDH": float(np.sum(mask_data == 4) * voxel_vol_ml),
            "V_SAH": float(np.sum(mask_data == 5) * voxel_vol_ml),
        }

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
