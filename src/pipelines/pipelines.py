from zenml import pipeline
from src.pipelines.steps import (
    prepare_nnunet_data, train_nnunet_step,
    prepare_yolo_data, train_yolo_step,
    prepare_mls_data, train_mls_step,
    prepare_ich_data, train_ich_step,
)

@pipeline
def nnunet_pipeline():
    """پایپ‌لاین اختصاصی تشخیص خونریزی (legacy — use ich_pipeline instead)"""
    data_ready = prepare_nnunet_data()
    train_nnunet_step(data_ready)

@pipeline
def yolo_pipeline():
    """پایپ‌لاین اختصاصی تشخیص شکستگی"""
    data_ready = prepare_yolo_data()
    train_yolo_step(data_ready)

@pipeline
def mls_pipeline():
    """پایپ‌لاین اختصاصی تشخیص شیفت خط میانی"""
    data_ready = prepare_mls_data()
    train_mls_step(data_ready)


@pipeline
def ich_pipeline(strategy_name: str = "nnunet", config_json: str = "{}"):
    """
    Generic ICH segmentation pipeline — strategy-agnostic.

    Selects the appropriate data preparation and training logic based
    on ``strategy_name`` ('nnunet', 'smp', 'monai', 'yolo_seg').

    Parameters
    ----------
    strategy_name : str
        Registered strategy name (see src/strategies/registry.py).
    config_json : str
        JSON-serialized strategy configuration. Validated against the
        strategy's Pydantic config model before training.
    """
    data_ready = prepare_ich_data(strategy_name)
    train_ich_step(data_ready, strategy_name, config_json)