import os
from pathlib import Path
import mlflow
from ultralytics import YOLO

def train_fracture_detector():
    print("=== Starting YOLO Fracture Detection Training ===")
    
    # ۱. تعریف مسیرهای دلخواه و تمیز
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # مسیر دلخواه برای ذخیره وزن‌های مدل و نتایج YOLO (مثلاً در پوشه مدل‌ها)
    CUSTOM_OUTPUT_DIR = BASE_DIR / "experiments" / "yolo_results"

    # مسیر دلخواه برای لاگ‌های MLflow (برای جلوگیری از ایجاد پوشه در Root)
    MLFLOW_DIR = BASE_DIR /  "logs" / "mlflow_runs"
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    # --- تنظیمات مدیریت شده MLflow ---
    mlflow_uri = MLFLOW_DIR.as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    mlflow.set_tracking_uri(mlflow_uri)
    # نام آزمایش در MLflow
    mlflow.set_experiment("Fracture_Detection_Exp") 
    
    # مسیر فایل YAML که در فاز قبلی ساختیم
    dataset_yaml = str(BASE_DIR / "Data" / "processed" / "yolo_fracture" / "dataset.yaml")
    

    weights_dir = BASE_DIR / "models" / "pretrained"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights_dir / "yolov8s.pt")
    
    results = model.train(
        data=dataset_yaml,
        epochs=150,           
        imgsz=512,
        batch=8, # در صورت داشتن VRAM بالا، این را بیشتر کن
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
    print(f"=== YOLO Training Completed! Results saved to: {CUSTOM_OUTPUT_DIR}/medical_fracture_v1 ===")

if __name__ == "__main__":
    train_fracture_detector()