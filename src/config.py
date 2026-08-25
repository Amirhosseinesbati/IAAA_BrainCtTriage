"""Configuration facade backed by ``config/project.yaml``.

The YAML document is the single source of truth. Existing constants remain
available so older modules keep working while new code consumes structured
sections through :func:`config_section`.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project.yaml"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_project_config(
    path: str | os.PathLike[str] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = Path(path or os.getenv("IAAA_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    if not selected.is_file():
        raise FileNotFoundError(f"Project config not found: {selected}")
    with selected.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise TypeError("Project YAML must contain a top-level mapping")
    return _deep_merge(payload, overrides or {})


PROJECT_CONFIG = load_project_config()


def config_section(*keys: str) -> Any:
    value: Any = PROJECT_CONFIG
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"Missing project config key: {'.'.join(walked)}")
        value = value[key]
    return deepcopy(value)


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_experiment_name(task_key: str) -> str:
    cfg = config_section("mlflow")
    return f"{cfg['experiment_prefix']}-{cfg['experiments'][task_key]}"


_PATHS = config_section("paths")
DATA_DIR = resolve_project_path(_PATHS["data"])
RAW_DIR = resolve_project_path(_PATHS["raw"])
PROCESSED_DIR = resolve_project_path(_PATHS["processed"])
METADATA_DIR = resolve_project_path(_PATHS["metadata"])
RAW_TRAINING_DIR = resolve_project_path(_PATHS["raw_training"])
RAW_ANNOTATIONS_DIR = resolve_project_path(_PATHS["raw_annotations"])
NNUNET_RAW_DIR = resolve_project_path(_PATHS["nnunet_raw"])
NNUNET_RESULTS_DIR = resolve_project_path(_PATHS["nnunet_results"])
ICH_NIFTI_DIR = resolve_project_path(_PATHS["ich_nifti"])
YOLO_DIR = resolve_project_path(_PATHS["yolo_fracture"])
MLS_DIR = resolve_project_path(_PATHS["mls_dataset"])
MODELS_DIR = resolve_project_path(_PATHS["models"])
YOLO_WEIGHTS_DIR = resolve_project_path(_PATHS["yolo_weights"])
MLS_CHECKPOINTS_DIR = resolve_project_path(_PATHS["mls_checkpoints"])
REPORTS_DIR = resolve_project_path(_PATHS["reports"])
TRAINING_CSV_PATH = resolve_project_path(_PATHS["training_csv"])
TRAINING_PKL_PATH = resolve_project_path(_PATHS["training_pickle"])

_IMAGING = config_section("imaging")
IMG_SIZE = int(_IMAGING["image_size"])
IMG_SIZE_MLS_SELECTOR = int(_IMAGING["mls_selector_image_size"])
WINDOWS = _IMAGING["windows"]
NIFTI_WINDOW = None
RANDOM_SEED = int(config_section("project", "random_seed"))

ICH_LABELS = config_section("labels", "ich")
ICH_LABEL_NAMES = {value: key for key, value in ICH_LABELS.items()}
ICH_TYPES = [name for name, value in sorted(ICH_LABELS.items(), key=lambda item: item[1]) if value]
MLS_KEYPOINT_NAMES = config_section("labels", "mls_keypoints")

TRIAGE_REQUIRED_KEYS = set(config_section("competition", "triage_required_keys"))
TRIAGE_THRESHOLDS = config_section("competition", "triage_thresholds")
SUBMISSION_MAX_BYTES = int(config_section("competition", "submission_max_bytes"))

MLFLOW_EXPERIMENT_PREFIX = config_section("mlflow", "experiment_prefix")
MLFLOW_EXP_YOLO = get_experiment_name("fracture")
MLFLOW_EXP_NNUNET = get_experiment_name("ich_nnunet")
MLFLOW_EXP_MLS_SELECTOR = get_experiment_name("mls_selector")
MLFLOW_EXP_MLS_KEYPOINT = get_experiment_name("mls_keypoint")
MLFLOW_EXP_MLS_HEATMAP = get_experiment_name("mls_heatmap")
MLFLOW_EXP_ICH_PREFIX = f"{MLFLOW_EXPERIMENT_PREFIX}-ich"
MLFLOW_EXP_ICH_NNUNET = get_experiment_name("ich_nnunet")
MLFLOW_EXP_ICH_SMP = get_experiment_name("ich_smp")
MLFLOW_EXP_ICH_MONAI = get_experiment_name("ich_monai")
MLFLOW_EXP_ICH_YOLO_SEG = get_experiment_name("ich_yolo_seg")

_TRAINING = config_section("training")
YOLO_DEFAULTS = _TRAINING["yolo"]
NNUNET_DEFAULTS = _TRAINING["nnunet"]
MLS_DEFAULTS = _TRAINING["mls_legacy"]
MLS_HEATMAP_DEFAULTS = _TRAINING["mls_heatmap"]
ICH_DEFAULT_STRATEGY = _TRAINING["defaults"]["ich_strategy"]
MLS_DEFAULT_STRATEGY = _TRAINING["defaults"]["mls_strategy"]
YOLO_TRAIN_RATIO = float(_TRAINING["splits"]["yolo_train_ratio"])
YOLO_VAL_RATIO = float(_TRAINING["splits"]["yolo_val_ratio"])


def _mlflow_log_dict_param(prefix: str, values: Mapping[str, Any]) -> None:
    from src.mlops.tracking import log_flat_params
    log_flat_params(values, prefix=prefix)


def log_src_snapshot() -> None:
    from src.mlops.tracking import log_source_snapshot
    log_source_snapshot()


def get_patient_ids(raw_dir: Path | None = None) -> list[str]:
    root = raw_dir or RAW_TRAINING_DIR
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def get_annotated_patient_ids(anno_dir: Path | None = None) -> list[str]:
    root = anno_dir or RAW_ANNOTATIONS_DIR
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def get_volume_from_area(area_pixels: float, spacing_x: float, spacing_y: float, thickness: float) -> float:
    return area_pixels * spacing_x * spacing_y * thickness / 1000.0
