import json
from pathlib import Path
from zenml import step

from src.preprocessing.builders.nifti_builder import NiftiDatasetBuilder
from src.preprocessing.builders.yolo_builder import YoloDatasetBuilder
from src.training.train_yolo import train_fracture_detector
from src.strategies.config_models import FractureYOLOConfig
from src.config import RAW_ANNOTATIONS_DIR, RAW_TRAINING_DIR, YOLO_DIR

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
def prepare_yolo_data(should_prepare: bool = True, config_json: str = "{}") -> bool:
    if not should_prepare:
        print("=== Reusing existing YOLO data (preparation disabled by manifest) ===")
        return True
    print("=== Preparing YOLO Data ===")
    resolved = FractureYOLOConfig.model_validate(json.loads(config_json or "{}"))
    fold = resolved.fold
    out_dir = str(YOLO_DIR / f"fold_{fold}")
    builder = YoloDatasetBuilder(
        str(RAW_TRAINING_DIR), str(RAW_ANNOTATIONS_DIR), out_dir,
        fold=fold,
        use_competition_folds=resolved.use_competition_folds,
    )
    builder.build()
    return True

# ==========================================
# Training Steps (Tracked via MLflow/DagsHub)
# ==========================================
@step
def train_yolo_step(data_ready: bool, config_json: str = "{}") -> bool:
    if data_ready:
        config = FractureYOLOConfig.model_validate(json.loads(config_json or "{}"))
        train_fracture_detector(config.model_dump())
    return True


# ==========================================
# Generic ICH Steps (Strategy Pattern)
# ==========================================

@step(enable_cache=False)
def prepare_ich_data(
    strategy_name: str = "monai", should_prepare: bool = True, config_json: str = "{}",
) -> bool:
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
    config = strategy.validate_config(json.loads(config_json or "{}"))
    return strategy.prepare_data(config)


@step
def train_ich_step(
    data_ready: bool,
    strategy_name: str = "monai",
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
def prepare_mls_strategy_step(
    strategy_name: str = "mls_heatmap", should_prepare: bool = True, config_json: str = "{}",
) -> bool:
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
    config = strategy.validate_config(json.loads(config_json or "{}"))
    return strategy.prepare_data(config)


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
