import os
import glob
import pydicom
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

class DicomToNiftiConverter:
    def __init__(self, raw_data_dir, output_dir):
        """
        raw_data_dir: مسیر پوشه training که داخلش پوشه بیماران (مثلا 272179) است.
        output_dir: مسیر خروجی برای ذخیره فایل‌های .nii.gz (مثلا data/interim/images)
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        
        # ساخت پوشه خروجی در صورت عدم وجود
        os.makedirs(self.output_dir, exist_ok=True)

    def process_patient(self, patient_id):
        """
        خواندن، مرتب‌سازی و تبدیل دایکام‌های یک بیمار به NIfTI
        """
        patient_folder = os.path.join(self.raw_data_dir, str(patient_id))
        dicom_files = glob.glob(os.path.join(patient_folder, "*.dcm"))
        
        if not dicom_files:
            print(f"Warning: No DICOM files found for patient {patient_id}")
            return None

        # 1. خواندن تمام فایل‌های دایکام
        slices = [pydicom.dcmread(f) for f in dicom_files]
        
        # 2. مرتب‌سازی اسلایس‌ها بر اساس محور Z (ImagePositionPatient[2])
        # این کار حیاتی است تا مغز از پایین به بالا درست چیده شود
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # 3. استخراج متادیتا از اولین اسلایس برای تبدیل HU در فاز بعدی
        first_slice = slices[0]
        slope = getattr(first_slice, 'RescaleSlope', 1.0)
        intercept = getattr(first_slice, 'RescaleIntercept', 0.0)
        
        # فاصله پیکسل‌ها (Pixel Spacing در محور x و y)
        spacing_xy = getattr(first_slice, 'PixelSpacing', [1.0, 1.0])
        
        # محاسبه ضخامت واقعی اسلایس (Z-spacing)
        # به جای SliceThickness از فاصله هندسی اسلایس 0 و 1 استفاده میکنیم که دقیقتر است
        if len(slices) > 1:
            z_spacing = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
        else:
            z_spacing = getattr(first_slice, 'SliceThickness', 1.0)

        # 4. روی هم قرار دادن ماتریس‌های 2D برای ساخت حجم 3D (Z-Stacking)
        # shape نهایی: (Height, Width, Depth)
        image_3d = np.stack([s.pixel_array for s in slices], axis=-1)
        
        # 5. ساخت ماتریس Affine (برای اینکه فرمت NIfTI مقیاس فیزیکی را بفهمد)
        affine = np.diag([float(spacing_xy[1]), float(spacing_xy[0]), float(z_spacing), 1.0])
        
        # 6. ذخیره به فرمت .nii.gz
        nifti_img = nib.Nifti1Image(image_3d, affine)
        output_filepath = os.path.join(self.output_dir, f"{patient_id}.nii.gz")
        nib.save(nifti_img, output_filepath)
        
        # بازگرداندن متادیتا برای ذخیره در CSV
        return {
            "patient_id": patient_id,
            "original_z_slices": len(slices),
            "rescale_slope": slope,
            "rescale_intercept": intercept,
            "spacing_x": float(spacing_xy[1]),
            "spacing_y": float(spacing_xy[0]),
            "spacing_z": float(z_spacing)
        }

    def run_pipeline(self):
        """
        اجرای تبدیل برای تمام پوشه‌های بیماران
        """
        print("Starting DICOM to NIfTI conversion...")
        
        # پیدا کردن نام تمام پوشه‌ها (که همان ID بیماران است)
        patient_ids = [d for d in os.listdir(self.raw_data_dir) 
                    if os.path.isdir(os.path.join(self.raw_data_dir, d))]
        
        metadata_list = []
        
        # استفاده از tqdm برای نمایش نوار پیشرفت (Progress Bar)
        for pid in tqdm(patient_ids, desc="Processing Patients"):
            meta = self.process_patient(pid)
            if meta:
                metadata_list.append(meta)
                
        # ذخیره متادیتا در یک فایل CSV
        df_meta = pd.DataFrame(metadata_list)
        csv_path = os.path.join(self.output_dir, "metadata_phase1.csv")
        df_meta.to_csv(csv_path, index=False)
        print(f"\nConversion Complete! Metadata saved to {csv_path}")

# ==========================================
# نحوه اجرای کد
# ==========================================
if __name__ == "__main__":
    # مسیرهای خود را اینجا تنظیم کنید
    RAW_DIR = "Data/raw/training"                 # مسیر دایکام‌های خام شما
    OUT_DIR = "data/interim/images"           # پوشه‌ای که فایل‌های nii.gz آنجا می‌روند
    
    converter = DicomToNiftiConverter(raw_data_dir=RAW_DIR, output_dir=OUT_DIR)
    
    # پیشنهاد: برای تست، ابتدا فقط یک یا دو پوشه را در RAW_DIR قرار دهید 
    # تا از درستی کار مطمئن شوید، سپس روی کل دیتا اجرا کنید.
    converter.run_pipeline()