import os
import json
import subprocess
from pathlib import Path
import mlflow
from glob import glob

def get_dataset_name(nnunet_raw_dir: Path, dataset_id: str) -> str:
    """
    نام واقعی دیتاست را از فایل dataset.json آن استخراج می‌کند.
    """
    # ابتدا پوشه دیتاست را پیدا می‌کنیم (مثلاً Dataset501_Hemorrhage)
    dataset_folder_pattern = nnunet_raw_dir / f"Dataset{int(dataset_id):03d}_*"
    try:
        dataset_folder = next(Path(p) for p in glob(str(dataset_folder_pattern)))
        dataset_json_path = dataset_folder / "dataset.json"
        with open(dataset_json_path, 'r') as f:
            data = json.load(f)
        return data["name"] # نام دیتاست را از جیسان برمیگردانیم
    except (StopIteration, FileNotFoundError):
        print(f"Error: Could not find dataset folder or dataset.json for ID {dataset_id}")
        return None


def train_nnunet_pipeline(dataset_id="501", fold=0):
    print(f"=== Starting Automated nnU-Net Pipeline for Dataset {dataset_id}, Fold {fold} ===")
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    NNUNET_DIR = BASE_DIR / "Data" / "processed" / "nnUNet"
    MLFLOW_DIR = BASE_DIR / "logs" / "mlflow_runs"
    
    # اطمینان از وجود دایرکتوری‌ها
    (NNUNET_DIR / "nnUNet_raw").mkdir(parents=True, exist_ok=True)
    (NNUNET_DIR / "nnUNet_preprocessed").mkdir(parents=True, exist_ok=True)
    (NNUNET_DIR / "nnUNet_results").mkdir(parents=True, exist_ok=True)
    
    # تنظیم MLflow
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("Hemorrhage_nnUNet_Exp")
    
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_DIR / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(NNUNET_DIR / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(NNUNET_DIR / "nnUNet_results")

    with mlflow.start_run(run_name=f"Dataset_{dataset_id}_Fold_{fold}"):
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("fold", fold)
        mlflow.log_param("network", "3d_fullres")

        print("\n--- Running nnUNetv2_plan_and_preprocess ---")
        preprocess_cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"]
        subprocess.run(preprocess_cmd, env=env, check=True)

        print(f"\n--- Running nnUNetv2_train (Fold {fold}) ---")
        train_cmd = ["nnUNetv2_train", str(dataset_id), "3d_fullres", str(fold)] 
        subprocess.run(train_cmd, env=env, check=True)
        
        # ======================= بخش اصلاح شده و داینامیک =======================
        print("\n--- Logging artifacts to MLflow ---")
        
        # ۱. نام واقعی دیتاست را به صورت داینامیک پیدا می‌کنیم
        dataset_name = get_dataset_name(NNUNET_DIR / "nnUNet_raw", dataset_id)
        if not dataset_name:
            print("Could not log artifacts because dataset name was not found. Exiting.")
            return

        # ۲. مسیر پایه نتایج برای این دیتاست
        dataset_results_dir = NNUNET_DIR / "nnUNet_results" / f"Dataset{int(dataset_id):03d}_{dataset_name}"
        
        # ۳. پوشه ترینر را به جای هاردکد کردن، پیدا می‌کنیم
        try:
            # الگو: هر چیزی که با nnUNetTrainer شروع شود و شامل 3d_fullres باشد
            trainer_folder_pattern = str(dataset_results_dir / "nnUNetTrainer*__*__3d_fullres")
            trainer_folder = Path(glob(trainer_folder_pattern)[0])
            fold_folder = trainer_folder / f"fold_{fold}"
            print(f"Found results folder: {fold_folder}")

            # ۴. لاگ کردن فایل progress.png
            progress_png = fold_folder / "progress.png"
            if progress_png.exists():
                mlflow.log_artifact(str(progress_png), artifact_path="training_plots")
                print("Logged progress.png to MLflow.")

            # ۵. پیدا کردن جدیدترین فایل لاگ و ذخیره آن
            log_files = list(fold_folder.glob("training_log_*.txt"))
            if log_files:
                latest_log_file = max(log_files, key=os.path.getctime)
                mlflow.log_artifact(str(latest_log_file), artifact_path="training_logs")
                print(f"Logged latest training log: {latest_log_file.name}")
            
            # ۶. لاگ کردن بهترین مدل
            best_model_file = fold_folder / "checkpoint_best.pth"
            if best_model_file.exists():
                mlflow.log_artifact(str(best_model_file), artifact_path="models")
                print("Logged best model (checkpoint_best.pth) to MLflow.")

        except IndexError:
            print(f"Error: Could not find a matching trainer results folder in {dataset_results_dir}")
        except Exception as e:
            print(f"An error occurred during artifact logging: {e}")

    print(f"=== nnU-Net Training for Fold {fold} Completed Successfully! ===")


if __name__ == "__main__":
    # می‌توانیم برای همه fold ها به صورت خودکار اجرا کنیم
    for i in range(0): # nnU-Net به طور پیش‌فرض 5-fold است
        train_nnunet_pipeline(dataset_id="501", fold=i)