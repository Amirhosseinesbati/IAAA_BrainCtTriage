from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import warnings
import os
warnings.filterwarnings("ignore")

# افزودن مسیر پروژه به Python Path برای پیدا کردن ماژول‌ها
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.core.dicom_reader import BrainDicomReader
from inference.predictors import ICHPredictor, FracturePredictor, MLSPredictor
from inference.triage_rules import apply_triage_rules

# این متغیر، مسیر ریشه پروژه شما را مشخص می‌کند
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_all_models(device='cuda'):
    """
    بارگذاری تمام مدل‌های آموزش دیده در حافظه.
    این تابع فقط یک بار در ابتدای برنامه اجرا می‌شود.
    """
    # 🚨🚨 مسیر فایل‌های وزن مدل‌های خود را اینجا قرار دهید 🚨🚨
    BASE_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
    
    ich_predictor = ICHPredictor(device=device)
    
    fracture_model_path = str(BASE_MODEL_DIR / "yolo_weights" / "best.pt") # مثال
    fracture_predictor = FracturePredictor(fracture_model_path, device=device)
    
    slice_model_path = str(BASE_MODEL_DIR / "checkpoints" / "slice_selector_best.ckpt") # مثال
    kp_model_path = str(BASE_MODEL_DIR / "checkpoints" / "keypoint_best.ckpt") # مثال
    mls_predictor = MLSPredictor(slice_model_path, kp_model_path, device=device)
    
    return {
        "ich": ich_predictor,
        "fracture": fracture_predictor,
        "mls": mls_predictor
    }

def predict(study_dir, models):
    """
    این همان تابعی است که مسابقه از شما می‌خواهد.
    
    ورودی:
    study_dir (str): مسیر پوشه دایکام‌های یک بیمار
    models (dict): دیکشنری مدل‌های لود شده
    
    خروجی:
    str: برچسب نهایی تریاژ
    """
    try:
        # 1. خواندن دایکام با ابزار هسته‌ای که ساختیم
        reader = BrainDicomReader(study_dir).load_and_sort()
        
        # 2. اجرای پیش‌بینی‌ها
        print(f"Analyzing patient {reader.metadata['patient_id']}...")
        ich_volumes = models["ich"].predict(reader)
        has_fracture = models["fracture"].predict(reader)
        mls_mm = models["mls"].predict(reader)
        
        # چاپ نتایج خام
        print(f"  - ICH Volumes (ml): {ich_volumes}")
        print(f"  - Fracture Detected: {has_fracture}")
        print(f"  - Midline Shift (mm): {mls_mm:.2f}")

        # 3. اعمال قوانین و تصمیم‌گیری نهایی
        final_label = apply_triage_rules(ich_volumes, has_fracture, mls_mm)
        print(f"  - FINAL TRIAGE DECISION: {final_label}")
        
        return final_label

    except Exception as e:
        print(f"Error processing {study_dir}: {e}")
        # در صورت بروز خطا، یک لیبل پیش‌فرض (مثلا نرمال) برمی‌گردانیم
        return "Normal"

if __name__ == "__main__":
    # --- شبیه‌سازی اجرای مسابقه ---
    
    # 1. مدل‌ها یک بار در ابتدا لود می‌شوند
    print("Loading all models into memory... This may take a minute.")
    all_models = load_all_models()
    print("All models loaded successfully!")
    
    # 2. مسیر پوشه دایکام‌های تست را اینجا بدهید
    # برای مثال، روی یکی از بیماران پوشه training تست می‌کنیم
    TEST_PATIENT_DIR = str(BASE_DIR / "Data" / "raw" / "training" / "1952") # مثال
    
    # 3. تابع predict را برای بیمار تست اجرا کنید
    if os.path.isdir(TEST_PATIENT_DIR):
        result = predict(TEST_PATIENT_DIR, all_models)
        print(f"\nFinal result for patient {os.path.basename(TEST_PATIENT_DIR)}: {result}")
    else:
        print(f"Test patient directory not found: {TEST_PATIENT_DIR}")