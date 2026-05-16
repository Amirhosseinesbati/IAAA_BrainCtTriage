import os
import glob
import pydicom
import numpy as np
import nibabel as nib

class BrainDicomReader:
    def __init__(self, patient_dir):
        """
        مدیریت خواندن دایکام‌های یک بیمار
        patient_dir: مسیر پوشه بیمار (مثلا Data/raw/training/272179)
        """
        self.patient_dir = patient_dir
        self.dicom_files = glob.glob(os.path.join(self.patient_dir, "*.dcm"))
        self.slices = []
        self.metadata = {}
        
        if not self.dicom_files:
            raise ValueError(f"No DICOM files found in {self.patient_dir}")

    def load_and_sort(self):
        """خواندن دایکام‌ها و مرتب‌سازی دقیق از پایین به بالای سر"""
        self.slices = [pydicom.dcmread(f, force=True) for f in self.dicom_files]
        
        # مرتب‌سازی بر اساس محور Z (موقعیت اسلایس)
        self.slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # استخراج متادیتا از اولین اسلایس و کل حجم
        first_slice = self.slices[0]
        spacing_xy = getattr(first_slice, 'PixelSpacing', [1.0, 1.0])
        
        if len(self.slices) > 1:
            z_spacing = abs(float(self.slices[1].ImagePositionPatient[2]) - float(self.slices[0].ImagePositionPatient[2]))
        else:
            z_spacing = getattr(first_slice, 'SliceThickness', 1.0)
            
        self.metadata = {
            "patient_id": os.path.basename(self.patient_dir),
            "original_z_slices": len(self.slices),
            "rescale_slope": getattr(first_slice, 'RescaleSlope', 1.0),
            "rescale_intercept": getattr(first_slice, 'RescaleIntercept', 0.0),
            "spacing_x": float(spacing_xy[1]),
            "spacing_y": float(spacing_xy[0]),
            "spacing_z": float(z_spacing)
        }
        return self

    def get_3d_volume_hu(self):
        """تولید ماتریس سه‌بعدی با مقادیر Hounsfield Unit (HU)"""
        if not self.slices:
            self.load_and_sort()
            
        # Z-Stacking -> Shape: (H, W, D)
        image_3d_raw = np.stack([s.pixel_array for s in self.slices], axis=-1)
        
        # تبدیل به HU
        slope = self.metadata["rescale_slope"]
        intercept = self.metadata["rescale_intercept"]
        image_3d_hu = (image_3d_raw * slope) + intercept
        
        return image_3d_hu

    @staticmethod
    def apply_windowing(image_hu, window_width, window_level):
        """متد کمکی برای اعمال ویندوینگ (مناسب برای YOLO و MLS)"""
        min_val = window_level - (window_width / 2)
        max_val = window_level + (window_width / 2)
        windowed = np.clip(image_hu, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        return windowed
    
    def save_as_nifti(self, output_path):
        if not self.slices:
            self.load_and_sort()
        
        image_3d = self.get_3d_volume_hu()
        meta = self.metadata
        affine = np.diag([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"], 1.0])
        nifti_img = nib.Nifti1Image(image_3d, affine)
        nib.save(nifti_img, output_path)