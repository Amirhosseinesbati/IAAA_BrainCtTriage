import os
from pathlib import Path
import mlflow

# --- حل مشکل MLflow در ویندوز ---
# پیدا کردن مسیر اصلی پروژه
BASE_DIR = Path(__file__).resolve().parent.parent.parent
mlflow_path = BASE_DIR / "runs" / "mlflow"
mlflow_path.mkdir(parents=True, exist_ok=True) # ساخت پوشه در صورت نبودن

# تبدیل مسیر ویندوزی (D:\...) به مسیر استاندارد (file:///D:/...)
correct_uri = mlflow_path.as_uri()

# اجبار پایتون و YOLO به استفاده از مسیر استاندارد
os.environ["MLFLOW_TRACKING_URI"] = correct_uri
mlflow.set_tracking_uri(correct_uri)
# ---------------------------------

from ultralytics import YOLO

def train_fracture_detector():
    model = YOLO("yolov8s.pt")  
    
    results = model.train(
        data="data/processed/yolo_fracture/dataset.yaml",
        epochs=150,           # ۱۵۰ کافی است، دیدیم که در ایپوک 76 تمام شد
        imgsz=512,
        batch=4, 
        project="runs/detect/fracture_detection", 
        name="yolo_model_v3_medical_tuned",
        device=0,
        workers=0,
        
        # 🔴 خاموش کردن آگمنتیشن‌های مخرب برای پزشکی
        mosaic=0.0,       # خاموش کردن کلاژ کردن عکس‌ها (بسیار مهم)
        mixup=0.0,        # خاموش کردن ترکیب شفاف دو عکس
        
        # 🟢 روشن گذاشتن آگمنتیشن‌های مفید
        degrees=10.0,     # چرخش ملایم 10 درجه
        translate=0.1,    # جابجایی جزئی
        fliplr=0.5,       # قرینه چپ و راست
        
        # 🔵 تغییر الگوریتم یادگیری (بهترین حالت برای دیتای کم)
        optimizer='AdamW', # الگوریتم AdamW معمولا در دیتای کم پزشکی بهتر از SGD جواب می‌دهد
        lr0=0.001,         # نرخ یادگیری اولیه
        
        # تنظیم توقف زودهنگام
        patience=40        # اگر 40 ایپوک پیشرفت نکرد قطع کن (نیازی به 100 ایپوک صبر نیست)
    )

if __name__ == "__main__":
    train_fracture_detector()

if __name__ == "__main__":
    train_fracture_detector()