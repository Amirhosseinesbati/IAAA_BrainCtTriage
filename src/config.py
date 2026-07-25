"""
config.py — Centralized configuration for IAAA Brain CT Triage.

All paths, constants, and label mappings used across the project
should be defined here and imported from this module.
"""

import os
import shutil
import tempfile
from pathlib import Path

# ==========================================
# Directory Paths
# ==========================================

# Project root (3 levels up from src/config.py -> src/ -> project_root/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# Raw data subdirectories
RAW_TRAINING_DIR = RAW_DIR / "training"
RAW_ANNOTATIONS_DIR = RAW_DIR / "annotations"

# Processed data subdirectories
NNUNET_RAW_DIR = PROCESSED_DIR / "nnUNet" / "nnUNet_raw"
NNUNET_RESULTS_DIR = PROCESSED_DIR / "nnUNet" / "brain_ct_model_fold0"
ICH_NIFTI_DIR = PROCESSED_DIR / "ich_nifti"          # Generic NIfTI storage (strategy-agnostic)
YOLO_DIR = PROCESSED_DIR / "yolo_fracture"
MLS_DIR = PROCESSED_DIR / "mls_dataset"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_WEIGHTS_DIR = MODELS_DIR / "yolo_weights"
MLS_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Metadata files
TRAINING_CSV_PATH = METADATA_DIR / "training_df.csv"
TRAINING_PKL_PATH = RAW_DIR / "training_df.pkl"

# ==========================================
# Image & Window Settings
# ==========================================

IMG_SIZE = 512
IMG_SIZE_MLS_SELECTOR = 256  # SliceSelector uses 256x256
RANDOM_SEED = 42

# Windowing parameters (width, level) for CT Hounsfield Units
# Reference: https://radiopaedia.org/articles/windowing-ct
WINDOWS = {
    "brain": {"width": 80, "level": 40},
    "subdural": {"width": 200, "level": 80},
    "bone": {"width": 1000, "level": 400},
    "stroke": {"width": 40, "level": 40},      # For acute ischemia
    "soft_tissue": {"width": 350, "level": 50},  # General soft tissue
}

# Default window (used for nnUNet NIfTI export — full dynamic range)
NIFTI_WINDOW = None  # None means apply no windowing, keep full HU range

# ==========================================
# Label Mappings
# ==========================================

# ICH label mapping (must match nnUNet dataset.json)
# 0 = background (no pathology)
ICH_LABELS = {
    "background": 0,
    "IVH": 1,   # Intraventricular Hemorrhage
    "IPH": 2,   # Intraparenchymal Hemorrhage
    "SDH": 3,   # Subdural Hemorrhage
    "EDH": 4,   # Epidural Hemorrhage
    "SAH": 5,   # Subarachnoid Hemorrhage
}

# Reverse mapping: integer → name
ICH_LABEL_NAMES = {v: k for k, v in ICH_LABELS.items()}

# List of ICH types in label order (excluding background)
ICH_TYPES = ["IVH", "IPH", "SDH", "EDH", "SAH"]

# MLS keypoint names (as they appear in annotation JSONs)
MLS_KEYPOINT_NAMES = [
    "AnteriorFalxAttachment",
    "PosteriorFalxAttachment",
    "OutermostPointOfTheFalx",
]

# ==========================================
# Competition Triage Constants
# ==========================================

# Required intermediate keys for the official triage function
TRIAGE_REQUIRED_KEYS = {
    "V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
    "fracture_prob", "MLS_mm",
}

# Triage thresholds (from competition PDF)
TRIAGE_THRESHOLDS = {
    "EPS_VOLUME": 0.1,          # mL: ignore tiny volumes as noise
    "EPS_MLS": 1.0,             # mm: ignore <1mm as no meaningful shift
    "MLS_CRITICAL": 5.0,        # mm: high MLS threshold
    "MLS_URGENT_LOW": 3.0,      # mm: moderate MLS range lower bound
    "EDH_CRIT": 30.0,           # mL
    "SDH_CRIT": 70.0,           # mL
    "IPH_CRIT": 70.0,           # mL
    "TOTAL_VOL_CRIT": 60.0,     # mL
    "COMBO_MLS": 3.0,           # mm: for MLS + volume combo rule
    "COMBO_VOL": 40.0,          # mL
    "FRAC_VOL_CRIT": 15.0,      # mL: fracture + hemorrhage for critical
    "FRACTURE_PRESENCE_THRESHOLD": 0.5,
}

# ==========================================
# MLflow Experiment Naming (unified prefix per task)
# ==========================================

MLFLOW_EXPERIMENT_PREFIX = "IAAA_BrainCT"
MLFLOW_EXP_YOLO      = f"{MLFLOW_EXPERIMENT_PREFIX}_YOLO"
MLFLOW_EXP_NNUNET    = f"{MLFLOW_EXPERIMENT_PREFIX}_nnUNet"
MLFLOW_EXP_MLS_SELECTOR = f"{MLFLOW_EXPERIMENT_PREFIX}_MLS_Selector"
MLFLOW_EXP_MLS_KEYPOINT = f"{MLFLOW_EXPERIMENT_PREFIX}_MLS_Keypoint"

# ── MLS Heatmap Strategy experiment name ──
MLFLOW_EXP_MLS_HEATMAP = f"{MLFLOW_EXPERIMENT_PREFIX}_MLS_Heatmap"

# ── ICH Strategy-specific experiment names ──
MLFLOW_EXP_ICH_PREFIX = f"{MLFLOW_EXPERIMENT_PREFIX}_ICH"
MLFLOW_EXP_ICH_NNUNET   = f"{MLFLOW_EXP_ICH_PREFIX}_nnunet"
MLFLOW_EXP_ICH_SMP      = f"{MLFLOW_EXP_ICH_PREFIX}_smp"
MLFLOW_EXP_ICH_MONAI    = f"{MLFLOW_EXP_ICH_PREFIX}_monai"
MLFLOW_EXP_ICH_YOLO_SEG = f"{MLFLOW_EXP_ICH_PREFIX}_yolo_seg"

# Default ICH strategy (used when none is specified)
ICH_DEFAULT_STRATEGY = "nnunet"


def _mlflow_log_dict_param(prefix: str, d: dict) -> None:
    """Helper: log a nested config dict as flat MLflow params with a prefix."""
    import mlflow
    for key, value in d.items():
        param_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _mlflow_log_dict_param(param_name, value)
        elif isinstance(value, (list, tuple)):
            mlflow.log_param(param_name, str(value))
        else:
            mlflow.log_param(param_name, value)


def log_src_snapshot():
    """
    Zip the entire src/ directory and log it to MLflow as a code snapshot artifact.
    Should be called inside an active MLflow run.
    """
    src_dir = PROJECT_ROOT / "src"
    if not src_dir.exists():
        print("⚠️  Source snapshot: src/ directory not found, skipping.")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = shutil.make_archive(
            os.path.join(tmpdir, "src_snapshot"),
            "zip",
            str(src_dir),
        )
        try:
            import mlflow
            mlflow.log_artifact(zip_path, artifact_path="code_snapshot")
            print(f"✅ Source code snapshot logged to MLflow (artifact_path='code_snapshot')")
        except Exception as e:
            print(f"⚠️  Could not log code snapshot to MLflow: {e}")


# ==========================================
# Training Defaults
# ==========================================

YOLO_DEFAULTS = {
    "image_size": IMG_SIZE,
    "epochs": 150,
    "batch_size": 8,
    "patience": 100,
    "optimizer": "AdamW",
    "lr": 0.001,
    "pretrained": "yolov8s.pt",
}

NNUNET_DEFAULTS = {
    "dataset_id": 501,
    "dataset_name": "BrainICH",
    "fold": 0,
    "num_folds": 5,
    # انتخاب معماری: "2d" یا "3d_fullres" یا "3d_lowres"
    "configuration": "2d",
    # Early stopping: اگر val_loss بهبود نداشت، بعد از این تعداد epoch متوقف کن
    "early_stopping_patience": 100,
    # ذخیره checkpoint هر چند epoch یکبار (پیش‌فرض nnU-Net=50، کمتر = آپلود بیشتر در MLflow)
    "save_every": 20,
}

MLS_DEFAULTS = {
    "selector_epochs": 20,
    "selector_batch_size": 32,
    "keypoint_epochs": 40,
    "keypoint_batch_size": 16,
    "selector_img_size": IMG_SIZE_MLS_SELECTOR,
}

# MLS Heatmap Strategy (new) defaults
MLS_HEATMAP_DEFAULTS = {
    "backbone": "hrnet_w32",
    "heatmap_sigma": 2.0,
    "heatmap_size": 128,
    "top_k_slices": 3,
    "aggregation": "max",
}

# Default MLS strategy (used when none is specified)
MLS_DEFAULT_STRATEGY = "mls_heatmap"

# ==========================================
# YOLO / nnUNet Data Splits
# ==========================================

YOLO_TRAIN_RATIO = 0.8
YOLO_VAL_RATIO = 0.2

# ==========================================
# Utility Functions
# ==========================================

def get_patient_ids(raw_dir: Path = None) -> list:
    """Get sorted list of patient directory IDs from raw training directory."""
    if raw_dir is None:
        raw_dir = RAW_TRAINING_DIR
    if not raw_dir.exists():
        return []
    return sorted(
        d.name for d in raw_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )


def get_annotated_patient_ids(anno_dir: Path = None) -> list:
    """Get sorted list of patient IDs that have annotation folders."""
    if anno_dir is None:
        anno_dir = RAW_ANNOTATIONS_DIR
    if not anno_dir.exists():
        return []
    return sorted(
        d.name for d in anno_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )


def get_volume_from_area(area_pixels: float, spacing_x: float,
                         spacing_y: float, thickness: float) -> float:
    """
    Convert pixel area to volume in mL.

    Formula: volume_mL = area_pixels * spacing_x * spacing_y * thickness / 1000
    (1 mm³ = 0.001 mL)
    """
    return area_pixels * spacing_x * spacing_y * thickness / 1000.0
