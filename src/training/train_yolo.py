import os
from pathlib import Path
import mlflow
from ultralytics import YOLO, settings

def train_fracture_detector():
    print("=== Starting YOLO Fracture Detection Training ===")
    
    # مطمئن می‌شویم لاگر MLflow یولو روشن است
    settings.update({'mlflow': True})
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CUSTOM_OUTPUT_DIR = BASE_DIR / "experiments" / "yolo_results"
    # MLFLOW_DIR = BASE_DIR /  "logs" / "mlflow_runs"
    # MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    # # --- تنظیمات MLflow برای هدایت YOLO ---
    # os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_DIR.as_uri()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Fracture_Detection_Exp" # نام آزمایش را به یولو می‌دهیم
    os.environ["MLFLOW_RUN"] = "yolo_v8s_fracture" # نام ران را به یولو می‌دهیم
    
    dataset_yaml = str(BASE_DIR / "Data" / "processed" / "yolo_fracture" / "dataset.yaml")
    
    weights_dir = BASE_DIR / "models" / "pretrained"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights_dir / "yolov8s.pt")

    # دقت کنید: بلاک with mlflow.start_run() حذف شد. YOLO خودش هندل می‌کند
    results = model.train(
        data=dataset_yaml,
        epochs=150,           
        imgsz=512,
        batch=8, 
        project=str(CUSTOM_OUTPUT_DIR), 
        name="medical_fracture_v1",
        device=0,
        workers=4,
        # تنظیمات مدیکال
        mosaic=0.0,       
        mixup=0.0,        
        degrees=10.0,     
        translate=0.1,    
        fliplr=0.5,       
        optimizer='AdamW',
        lr0=0.001,         
        patience=100        
    )

    print(f"=== YOLO Training Completed! Results saved to: {results.save_dir} ===")
    print("Check MLflow UI. YOLO has automatically logged metrics, parameters, and the best model!")

if __name__ == "__main__":
    train_fracture_detector()