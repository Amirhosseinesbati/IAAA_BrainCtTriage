import json
import os
from pathlib import Path
from zenml import step

from src.preprocessing.builders.nifti_builder import NiftiDatasetBuilder
from src.preprocessing.builders.nnunet_builder import NNUnetDatasetBuilder
from src.preprocessing.builders.yolo_builder import YoloDatasetBuilder
from src.preprocessing.builders.mls_builder import MlsDatasetBuilder
from src.training.train_nnunet import train_nnunet_pipeline
from src.training.train_yolo import train_fracture_detector
from src.training.train_mls import train_slice_selector, train_keypoint_detector
from src.config import NNUNET_DEFAULTS

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ==========================================
# Data Preparation Steps (No Cache - Saves S3 Space)
# ==========================================
@step(enable_cache=False)
def prepare_nifti_data() -> bool:
    """Prepare generic NIfTI data (strategy-agnostic, no nnUNet naming)."""
    print("=== Preparing Generic NIfTI Data (ICH_NIFTI_DIR) ===")
    builder = NiftiDatasetBuilder()
    builder.build()
    return True


@step(enable_cache=False)
def prepare_nnunet_data() -> bool:
    print("=== Preparing nnU-Net Data ===")
    out_dir = str(BASE_DIR / "Data/processed/nnUNet/nnUNet_raw")
    builder = NNUnetDatasetBuilder(str(BASE_DIR/"Data/raw/training"), str(BASE_DIR/"Data/raw/annotations"), out_dir)
    builder.build()
    return True

@step(enable_cache=False)
def prepare_yolo_data(should_prepare: bool = True) -> bool:
    if not should_prepare:
        print("=== Reusing existing YOLO data (preparation disabled by manifest) ===")
        return True
    print("=== Preparing YOLO Data ===")
    out_dir = str(BASE_DIR / "Data/processed/yolo_fracture")
    builder = YoloDatasetBuilder(str(BASE_DIR/"Data/raw/training"), str(BASE_DIR/"Data/raw/annotations"), out_dir)
    builder.build()
    return True

@step(enable_cache=False)
def prepare_mls_data() -> bool:
    print("=== Preparing MLS Data ===")
    out_dir = str(BASE_DIR / "Data/processed/mls_dataset")
    builder = MlsDatasetBuilder(str(BASE_DIR/"Data/raw/training"), str(BASE_DIR/"Data/raw/annotations"), out_dir)
    builder.build()
    return True

# ==========================================
# Training Steps (Tracked via MLflow/DagsHub)
# ==========================================
@step
def train_nnunet_step(data_ready: bool) -> bool:
    if data_ready:
        config = NNUNET_DEFAULTS.get("configuration", "2d")
        train_nnunet_pipeline(dataset_id="501", fold=0, configuration=config)
    return True

@step
def train_yolo_step(data_ready: bool, config_json: str = "{}") -> bool:
    if data_ready:
        train_fracture_detector(json.loads(config_json or "{}"))
    return True

@step
def train_mls_step(data_ready: bool) -> bool:
    if data_ready:
        ckpt_dir = str(BASE_DIR / "models" / "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        train_slice_selector(str(BASE_DIR/"Data/processed/mls_dataset/mls_labels.csv"), 
                             str(BASE_DIR/"Data/processed/mls_dataset/images"), ckpt_dir)
        train_keypoint_detector(str(BASE_DIR/"Data/processed/mls_dataset/mls_labels.csv"), 
                                str(BASE_DIR/"Data/processed/mls_dataset/images"), ckpt_dir)
    return True


# ==========================================
# Generic ICH Steps (Strategy Pattern)
# ==========================================

@step(enable_cache=False)
def prepare_ich_data(strategy_name: str = "nnunet", should_prepare: bool = True) -> bool:
    """
    Prepare data for the selected ICH segmentation strategy.

    Delegates to the strategy's ``prepare_data()`` method. Cache is
    disabled to ensure fresh data on every run (saves S3 space).
    """
    from src.strategies import get_strategy

    if not should_prepare:
        print(f"=== Reusing existing ICH data for '{strategy_name}' ===")
        return True
    print(f"=== Preparing Data for ICH strategy: '{strategy_name}' ===")
    strategy = get_strategy(strategy_name)
    return strategy.prepare_data()


@step
def train_ich_step(
    data_ready: bool,
    strategy_name: str = "nnunet",
    config_json: str = "{}",
) -> bool:
    """
    Train the selected ICH segmentation strategy with the given config.

    The ``config_json`` is validated against the strategy's Pydantic
    config model before training begins. All metrics and artifacts are
    logged to MLflow automatically by each strategy.
    """
    if not data_ready:
        print(f"⚠️  Data preparation failed — skipping training for '{strategy_name}'")
        return False

    from src.strategies import get_strategy

    print(f"=== Training ICH strategy: '{strategy_name}' ===")

    strategy = get_strategy(strategy_name)
    config_dict = json.loads(config_json) if config_json else {}

    # Validate config via Pydantic
    config = strategy.validate_config(config_dict)

    print(f"   Config: {config.model_dump_json(indent=2)}")
    return strategy.train(config)


# ==========================================
# Generic MLS Steps (Strategy Pattern)
# Mirrors the ICH strategy-agnostic steps (prepare_ich_data / train_ich_step)
# but dispatches through the MLSStrategyRegistry.
# ==========================================

@step(enable_cache=False)
def prepare_mls_strategy_step(strategy_name: str = "mls_heatmap", should_prepare: bool = True) -> bool:
    """
    Prepare data for the selected MLS estimation strategy.

    Delegates to the strategy's ``prepare_data()`` method. Cache is
    disabled to ensure fresh data on every run (saves S3 space).
    """
    from src.strategies import get_mls_strategy

    if not should_prepare:
        print(f"=== Reusing existing MLS data for '{strategy_name}' ===")
        return True
    print(f"=== Preparing Data for MLS strategy: '{strategy_name}' ===")
    strategy = get_mls_strategy(strategy_name)
    return strategy.prepare_data()


@step
def train_mls_strategy_step(
    data_ready: bool,
    strategy_name: str = "mls_heatmap",
    config_json: str = "{}",
) -> bool:
    """
    Train the selected MLS estimation strategy with the given config.

    The ``config_json`` is validated against the strategy's Pydantic
    config model before training begins. All metrics and artifacts are
    logged to MLflow automatically by each strategy.
    """
    if not data_ready:
        print(f"⚠️  Data preparation failed — skipping training for '{strategy_name}'")
        return False

    from src.strategies import get_mls_strategy

    print(f"=== Training MLS strategy: '{strategy_name}' ===")

    strategy = get_mls_strategy(strategy_name)
    config_dict = json.loads(config_json) if config_json else {}

    # Validate config via Pydantic
    config = strategy.validate_config(config_dict)

    print(f"   Config: {config.model_dump_json(indent=2)}")
    return strategy.train(config)
