import os
import json
import glob
import pydicom
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

class JsonToNiftiLabeler:
    def __init__(self, raw_dicom_dir, raw_json_dir, interim_img_dir, output_mask_dir):
        self.raw_dicom_dir = raw_dicom_dir
        self.raw_json_dir = raw_json_dir
        self.interim_img_dir = interim_img_dir
        self.output_mask_dir = output_mask_dir
        
        os.makedirs(self.output_mask_dir, exist_ok=True)
        self.master_labels = [] # برای ذخیره نقاط کلیدی و باکس‌ها

    def decode_rle(self, counts, shape=(512, 512)):
        """
        تبدیل آرایه RLE [Value, Count, Value, Count] به ماتریس دو بعدی ماسک
        """
        if not counts:
            return np.zeros(shape, dtype=np.uint8)
            
        values = counts[0::2]  # اندیس‌های زوج (کلاس‌ها)
        lengths = counts[1::2] # اندیس‌های فرد (تعداد تکرار)
        
        # ساخت یک آرایه مسطح (1D) از کل پیکسل‌ها
        # np.repeat این کار را به شدت سریع انجام می‌دهد
        mask_flat = np.repeat(values, lengths).astype(np.uint8)
        
        # تبدیل به شکل 2D (512x512)
        # نکته: معمولا تصاویر پزشکی Row-major هستند. اگر در Sanity Check ماسک چرخیده بود، order='F' را تست می‌کنیم.
        mask_2d = mask_flat.reshape(shape, order='C') 
        return mask_2d

    def process_patient(self, patient_id):
        dicom_folder = os.path.join(self.raw_dicom_dir, str(patient_id))
        json_folder = os.path.join(self.raw_json_dir, str(patient_id))
        
        dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
        if not dicom_files:
            return
        
        # 1. خواندن دایکام‌ها و مرتب‌سازی دقیقاً مشابه فاز ۱ (برای همگام‌سازی محور Z)
        slices_info = []
        for f in dicom_files:
            ds = pydicom.dcmread(f, stop_before_pixels=True) # فقط هدر را می‌خوانیم تا سریع باشد
            filename = os.path.basename(f)
            z_pos = float(ds.ImagePositionPatient[2])
            slices_info.append((z_pos, filename))
            
        slices_info.sort(key=lambda x: x[0]) # مرتب‌سازی از پایین به بالا
        
        masks_3d_list = []
        
        # 2. پیمایش اسلایس‌ها و خواندن JSON متناظر
        for z_index, (_, dcm_filename) in enumerate(slices_info):
            json_filename = dcm_filename.replace('.dcm', '.json')
            json_path = os.path.join(json_folder, json_filename)
            
            mask_2d = np.zeros((512, 512), dtype=np.uint8)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as jf:
                    data = json.load(jf)
                    
                    # الف) ساخت ماسک خونریزی
                    if "segmentation_rle" in data and "counts" in data["segmentation_rle"]:
                        mask_2d = self.decode_rle(data["segmentation_rle"]["counts"], shape=data["segmentation_rle"]["shape"])
                    
                    # ب) استخراج نقاط کلیدی (Midline Shift)
                    if "keypoints" in data and data["keypoints"]:
                        for kp_name, coords in data["keypoints"].items():
                            if coords: # اگر لیست خالی نبود
                                self.master_labels.append({
                                    "patient_id": patient_id,
                                    "z_index": z_index,  # بسیار مهم! این نقطه در کدام اسلایس است
                                    "label_type": "keypoint",
                                    "name": kp_name,
                                    "x": coords[0],
                                    "y": coords[1]
                                })
                                
                    # ج) استخراج باکس شکستگی جمجمه
                    if "boxes_xywh" in data and data["boxes_xywh"]:
                        for box in data["boxes_xywh"]:
                            self.master_labels.append({
                                "patient_id": patient_id,
                                "z_index": z_index,
                                "label_type": "bbox",
                                "name": "fracture",
                                "box_data": box # [x_center, y_center, w, h]
                            })
                            
            masks_3d_list.append(mask_2d)
            
        # 3. روی هم قرار دادن ماسک‌های 2D برای ساخت ماسک 3D
        mask_3d_volume = np.stack(masks_3d_list, axis=-1)
        
        # 4. خواندن Affine تصویر اصلی از فاز ۱ (تا ماسک و تصویر دقیقاً روی هم منطبق شوند)
        img_nifti_path = os.path.join(self.interim_img_dir, f"{patient_id}.nii.gz")
        if os.path.exists(img_nifti_path):
            img_nifti = nib.load(img_nifti_path)
            affine = img_nifti.affine
        else:
            affine = np.eye(4)
            print(f"Warning: Base NIfTI not found for {patient_id}. Using default Affine.")

        # 5. ذخیره ماسک به عنوان .nii.gz
        mask_nifti = nib.Nifti1Image(mask_3d_volume, affine)
        output_mask_path = os.path.join(self.output_mask_dir, f"{patient_id}_mask.nii.gz")
        nib.save(mask_nifti, output_mask_path)

    def run_pipeline(self):
        print("Starting JSON parsing and 3D Mask generation...")
        patient_ids = [d for d in os.listdir(self.raw_dicom_dir) 
                    if os.path.isdir(os.path.join(self.raw_dicom_dir, d))]
        
        for pid in tqdm(patient_ids, desc="Processing Labels"):
            self.process_patient(pid)
            
        # ذخیره نقاط کلیدی و باکس‌ها در یک CSV تمیز
        if self.master_labels:
            df_labels = pd.DataFrame(self.master_labels)
            csv_path = os.path.join(self.output_mask_dir, "master_labels_phase2.csv")
            df_labels.to_csv(csv_path, index=False)
            print(f"\nLabels extracted and saved to {csv_path}")

# ==========================================
# نحوه اجرای کد
# ==========================================
if __name__ == "__main__":
    RAW_DICOM = "Data/raw/training"
    RAW_JSON = "Data/raw/annotations"
    INTERIM_IMG = "data/interim/images" # خروجی فاز 1
    OUT_MASK = "data/interim/masks"     # اینجا ماسک‌ها ذخیره می‌شوند
    
    labeler = JsonToNiftiLabeler(RAW_DICOM, RAW_JSON, INTERIM_IMG, OUT_MASK)
    labeler.run_pipeline()