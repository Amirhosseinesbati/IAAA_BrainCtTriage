from zenml import pipeline
from src.pipelines.steps import (
    prepare_yolo_data, train_yolo_step,
    prepare_ich_data, train_ich_step,
    prepare_mls_strategy_step, train_mls_strategy_step,
)

@pipeline
def yolo_pipeline(config_json: str = "{}", prepare_data: bool = True):
    """پایپ‌لاین اختصاصی تشخیص شکستگی"""
    data_ready = prepare_yolo_data(prepare_data, config_json)
    train_yolo_step(data_ready, config_json)

@pipeline
def ich_pipeline(strategy_name: str = "monai", config_json: str = "{}", prepare_data: bool = True):
    """
    Generic ICH segmentation pipeline — strategy-agnostic.

    Competition ICH pipeline. ``monai`` is intentionally the only active
    strategy so training and submission cannot silently diverge.

    Parameters
    ----------
    strategy_name : str
        Registered strategy name (see src/strategies/registry.py).
    config_json : str
        JSON-serialized strategy configuration. Validated against the
        strategy's Pydantic config model before training.
    """
    data_ready = prepare_ich_data(strategy_name, prepare_data, config_json)
    train_ich_step(data_ready, strategy_name, config_json)


@pipeline
def mls_strategy_pipeline(strategy_name: str = "mls_heatmap", config_json: str = "{}", prepare_data: bool = True):
    """
    Generic MLS estimation pipeline — strategy-agnostic (mirror of ich_pipeline).

    Selects the appropriate data preparation and training logic based on
    ``strategy_name`` (currently 'mls_heatmap').

    Parameters
    ----------
    strategy_name : str
        Registered MLS strategy name (see src/strategies/mls_registry.py).
    config_json : str
        JSON-serialized strategy configuration. Validated against the
        strategy's Pydantic config model before training.
    """
    data_ready = prepare_mls_strategy_step(strategy_name, prepare_data, config_json)
    train_mls_strategy_step(data_ready, strategy_name, config_json)
