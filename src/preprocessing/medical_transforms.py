import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd
)

class ClinicalPreprocessor:
    def __init__(self, interim_img_dir, interim_mask_dir, output_dir, metadata_csv):
        self.interim_img_dir = interim_img_dir
        self.interim_mask_dir = interim_mask_dir
        self.output_dir = output_dir
        
        self.out_img_dir = os.path.join(output_dir, "images")
        self.out_mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(self.out_img_dir, exist_ok=True)
        os.makedirs(self.out_mask_dir, exist_ok=True)
        
        # لود کردن متادیتا برای استخراج Slope و Intercept (فاز ۱)
        self.df_meta = pd.read_csv(metadata_csv)
        self.df_meta.set_index("patient_id", inplace=True)
        
        # تعریف پایپ‌لاین MONAI (فقط برای تغییر سایز و کراپ)
        self.monai_pipeline = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # استانداردسازی به فضای 1 در 1 در 1 میلی‌متر
            Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), 
                    mode=("bilinear", "nearest")),
            # حذف فضای خالی اطراف سر. شرط: مقادیری که بالاتر از پس‌زمینه هستند نگه داشته شوند
            CropForegroundd(keys=["image", "mask"], source_key="image", margin=2)
        ])

    def apply_hu_and_windowing(self, img_array, patient_id):
        """
        تبدیل مقادیر خام به HU و ساخت تصویر ۳ کاناله (RGB پزشکی)
        """
        # 1. تبدیل به HU
        try:
            slope = self.df_meta.loc[int(patient_id), "rescale_slope"]
            intercept = self.df_meta.loc[int(patient_id), "rescale_intercept"]
        except KeyError:
            slope, intercept = 1.0, 0.0 # مقادیر پیش‌فرض در صورت نبود دیتا
            
        hu_img = (img_array * slope) + intercept

        def apply_window(img, w, l):
            min_val = l - (w / 2)
            max_val = l + (w / 2)
            # محدود کردن مقادیر و نرمال‌سازی بین 0 و 1
            windowed = np.clip(img, min_val, max_val)
            windowed = (windowed - min_val) / (max_val - min_val)
            return windowed

        # 2. اعمال ۳ ویندوی مختلف
        # کانال 1: Brain Window (W:80, L:40) -> بهترین برای بافت مغز و خونریزی
        ch1 = apply_window(hu_img, w=80, l=40)
        
        # کانال 2: Subdural Window (W:200, L:80) -> بهترین برای خونریزی‌های چسبیده به جمجمه
        ch2 = apply_window(hu_img, w=200, l=80)
        
        # کانال 3: Bone Window (W:1000, L:400) -> بهترین برای استخوان و شکستگی
        ch3 = apply_window(hu_img, w=1000, l=400)

        # 3. روی هم قرار دادن کانال‌ها (Channel, Height, Width, Depth)
        multi_channel_img = np.stack([ch1, ch2, ch3], axis=0)
        
        return multi_channel_img

    def process_dataset(self):
        print("Starting MONAI Clinical Preprocessing...")
        
        # پیدا کردن تمام تصاویر در پوشه interim
        img_files = glob.glob(os.path.join(self.interim_img_dir, "*.nii.gz"))
        
        updated_meta = []

        for img_path in tqdm(img_files, desc="Processing Volumes"):
            patient_id = os.path.basename(img_path).replace(".nii.gz", "")
            mask_path = os.path.join(self.interim_mask_dir, f"{patient_id}_mask.nii.gz")
            
            if not os.path.exists(mask_path):
                print(f"Skipping {patient_id}: No mask found.")
                continue

            # 1. اجرای پایپ‌لاین MONAI (لود، Resample، Crop)
            data_dict = {"image": img_path, "mask": mask_path}
            transformed_data = self.monai_pipeline(data_dict)
            
            # استخراج آرایه‌ها (تبدیل تانسورهای MONAI به Numpy)
            # تصویر در این مرحله یک کانال دارد (1, H, W, D)، ما کانال اول را برمیداریم
            img_array = transformed_data["image"].numpy()[0] 
            mask_array = transformed_data["mask"].numpy()[0]
            
            # 2. اعمال ویندوینگ و تبدیل به 3 کانال
            final_img = self.apply_hu_and_windowing(img_array, patient_id)
            
            # 3. ذخیره با فرمت .npy برای حداکثر سرعت در زمان آموزش (فاز 4)
            out_img_path = os.path.join(self.out_img_dir, f"{patient_id}_img.npy")
            out_mask_path = os.path.join(self.out_mask_dir, f"{patient_id}_mask.npy")
            
            # فرمت ذخیره: تصویر=(3, H, W, D) ، ماسک=(H, W, D) (یا حفظ فرمت دلخواه)
            np.save(out_img_path, final_img.astype(np.float32))
            np.save(out_mask_path, mask_array.astype(np.uint8))
            
            # 4. نگهداری اطلاعات سایز جدید برای فاز بعدی
            updated_meta.append({
                "patient_id": patient_id,
                "final_shape": final_img.shape, # (3, H, W, D)
                "processed_img_path": out_img_path,
                "processed_mask_path": out_mask_path
            })

        # ذخیره لیست نهایی فایل‌های آماده شده
        df_processed = pd.DataFrame(updated_meta)
        df_processed.to_csv(os.path.join(self.output_dir, "processed_dataset.csv"), index=False)
        print("Preprocessing Complete! Data is ready for deep learning.")

# ==========================================
# نحوه اجرای کد
# ==========================================
if __name__ == "__main__":
    INTERIM_IMG = "data/interim/images"     # خروجی فاز 1
    INTERIM_MASK = "data/interim/masks"     # خروجی فاز 2
    OUT_DIR = "data/processed"              # دیتای نهایی آموزش
    META_CSV = "data/interim/images/metadata_phase1.csv"
    
    preprocessor = ClinicalPreprocessor(INTERIM_IMG, INTERIM_MASK, OUT_DIR, META_CSV)
    preprocessor.process_dataset()