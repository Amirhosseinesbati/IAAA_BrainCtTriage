import os
from pathlib import Path
from zenml import step

from src.preprocessing.builders.nnunet_builder import NNUnetDatasetBuilder
from src.preprocessing.builders.yolo_builder import YoloDatasetBuilder
from src.preprocessing.builders.mls_builder import MlsDatasetBuilder
from src.training.train_nnunet import train_nnunet_pipeline
from src.training.train_yolo import train_fracture_detector
from src.training.train_mls import train_slice_selector, train_keypoint_detector

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ==========================================
# Data Preparation Steps (No Cache - Saves S3 Space)
# ==========================================
@step(enable_cache=False)
def prepare_nnunet_data() -> bool:
    print("=== Preparing nnU-Net Data ===")
    out_dir = str(BASE_DIR / "Data/processed/nnUNet/nnUNet_raw")
    builder = NNUnetDatasetBuilder(str(BASE_DIR/"Data/raw/training"), str(BASE_DIR/"Data/raw/annotations"), out_dir)
    builder.build()
    return True

@step(enable_cache=False)
def prepare_yolo_data() -> bool:
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
    if data_ready: train_nnunet_pipeline(dataset_id="501", fold=0)
    return True

@step
def train_yolo_step(data_ready: bool) -> bool:
    if data_ready: train_fracture_detector()
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