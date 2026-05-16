import os
from pathlib import Path
from src.training.train_nnunet import train_nnunet_pipeline
from src.training.train_yolo import train_fracture_detector
from src.training.train_mls import train_slice_selector, train_keypoint_detector

def main():
    print("🚀 === MLOps Auto-Training Pipeline Started === 🚀")
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # ---------------------------------------------------------
    # 1. آموزش nnU-Net (خونریزی‌ها)
    # (هشدار: این مرحله ممکن است روزها طول بکشد. اگر قبلا آموزش دیده‌اید، کامنت کنید)
    # ---------------------------------------------------------
    # train_nnunet_pipeline(dataset_id="501")
    
    # ---------------------------------------------------------
    # 2. آموزش YOLO (شکستگی‌ها)
    # ---------------------------------------------------------
    # train_fracture_detector()
    
    # ---------------------------------------------------------
    # 3. آموزش مدل‌های انحراف خط میانی (MLS)
    # ---------------------------------------------------------
    mls_csv = str(BASE_DIR / "Data" / "processed" / "mls_dataset" / "mls_labels.csv")
    mls_img = str(BASE_DIR / "Data" / "processed" / "mls_dataset" / "images")
    ckpt_dir = str(BASE_DIR / "models" / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # آموزش مدل A (پیدا کردن بهترین اسلایس)
    train_slice_selector(mls_csv, mls_img, ckpt_dir)
    
    # آموزش مدل B (پیدا کردن کی‌پوینت‌ها)
    train_keypoint_detector(mls_csv, mls_img, ckpt_dir)

    print("\n🎉 === All Training Tasks Completed Successfully! === 🎉")

if __name__ == "__main__":
    main()