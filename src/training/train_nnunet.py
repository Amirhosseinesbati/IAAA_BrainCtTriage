import os
import subprocess
from pathlib import Path

def train_nnunet_pipeline(dataset_id="501"):
    print("=== Starting Automated nnU-Net Pipeline ===")
    
    # 1. تنظیم مسیرهای پایه به صورت مطلق (Absolute Paths)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    NNUNET_DIR = BASE_DIR / "Data" / "processed" / "nnUNet"
    
    os.makedirs(NNUNET_DIR / "nnUNet_raw", exist_ok=True)
    os.makedirs(NNUNET_DIR / "nnUNet_preprocessed", exist_ok=True)
    os.makedirs(NNUNET_DIR / "nnUNet_results", exist_ok=True)
    
    # 2. تزریق متغیرهای محیطی به خود پایتون
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_DIR / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(NNUNET_DIR / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(NNUNET_DIR / "nnUNet_results")

    # 3. اجرای دستور Preprocessing (استخراج Fingerprint دیتاست)
    print("\n--- Running nnUNetv2_plan_and_preprocess ---")
    preprocess_cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"]
    subprocess.run(preprocess_cmd, env=env, check=True)

    # 4. اجرای دستور Training (آموزش روی Fold 0)
    # نکته: برای مسابقه معمولا اجرای Fold 0 کافی است، اما اگر خواستی میتوانی از 0 تا 4 را در حلقه بگذاری
    print("\n--- Running nnUNetv2_train (Fold 0) ---")
    # 3d_fullres یعنی آموزش سه بعدی با رزولوشن اصلی
    train_cmd = ["nnUNetv2_train", str(dataset_id), "3d_fullres", "0"] 
    subprocess.run(train_cmd, env=env, check=True)
    
    print("=== nnU-Net Training Completed Successfully! ===")

if __name__ == "__main__":
    train_nnunet_pipeline()