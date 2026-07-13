from zenml import pipeline
from src.pipelines.steps import (
    prepare_nnunet_data, train_nnunet_step,
    prepare_yolo_data, train_yolo_step,
    prepare_mls_data, train_mls_step
)

@pipeline
def nnunet_pipeline():
    """پایپ‌لاین اختصاصی تشخیص خونریزی"""
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