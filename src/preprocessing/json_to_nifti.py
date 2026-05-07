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
        mask_flat = np.repeat(values, lengths).astype(np.uint8)
        
        # تبدیل به شکل 2D (512x512)
        mask_2d = mask_flat.reshape(shape, order='C') 
        return mask_2d

    def process_patient(self, patient_id):
        dicom_folder = os.path.join(self.raw_dicom_dir, str(patient_id))
        json_folder = os.path.join(self.raw_json_dir, str(patient_id))
        
        # ---------------------------------------------------------
        # تغییر مهم 1: اگر پوشه JSON وجود ندارد یا خالی است، کلا رد شو
        # ---------------------------------------------------------
        if not os.path.exists(json_folder) or len(glob.glob(os.path.join(json_folder, "*.json"))) == 0:
            return # بدون انجام هیچ کاری از این بیمار عبور میکند

        dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
        if not dicom_files:
            return
        
        # خواندن دایکام‌ها و مرتب‌سازی
        slices_info = []
        for f in dicom_files:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            filename = os.path.basename(f)
            z_pos = float(ds.ImagePositionPatient[2])
            slices_info.append((z_pos, filename))
            
        slices_info.sort(key=lambda x: x[0]) 
        
        masks_3d_list = []
        
        # ---------------------------------------------------------
        # تغییر مهم 2: یک پرچم (Flag) برای اطمینان از خوانده شدن حداقل یک لیبل
        # ---------------------------------------------------------
        has_any_label = False 
        
        for z_index, (_, dcm_filename) in enumerate(slices_info):
            json_filename = dcm_filename.replace('.dcm', '.json')
            json_path = os.path.join(json_folder, json_filename)
            
            mask_2d = np.zeros((512, 512), dtype=np.uint8)
            
            if os.path.exists(json_path):
                has_any_label = True # حداقل یک JSON پیدا شد!
                
                with open(json_path, 'r') as jf:
                    data = json.load(jf)
                    
                    if "segmentation_rle" in data and "counts" in data["segmentation_rle"]:
                        mask_2d = self.decode_rle(data["segmentation_rle"]["counts"], shape=data["segmentation_rle"]["shape"])
                    
                    if "keypoints" in data and data["keypoints"]:
                        for kp_name, coords in data["keypoints"].items():
                            if coords: 
                                self.master_labels.append({
                                    "patient_id": patient_id,
                                    "z_index": z_index,  
                                    "label_type": "keypoint",
                                    "name": kp_name,
                                    "x": coords[0],
                                    "y": coords[1]
                                })
                                
                    if "boxes_xywh" in data and data["boxes_xywh"]:
                        for box in data["boxes_xywh"]:
                            self.master_labels.append({
                                "patient_id": patient_id,
                                "z_index": z_index,
                                "label_type": "bbox",
                                "name": "fracture",
                                "box_data": box
                            })
                            
            masks_3d_list.append(mask_2d)
            
        # ---------------------------------------------------------
        # تغییر مهم 3: اگر هیچ جیسونی برای اسلایس‌های این بیمار پیدا نشد، ذخیره نکن
        # ---------------------------------------------------------
        if not has_any_label:
            return

        # ساخت ماسک 3D و ذخیره
        mask_3d_volume = np.stack(masks_3d_list, axis=-1)
        
        img_nifti_path = os.path.join(self.interim_img_dir, f"{patient_id}.nii.gz")
        if os.path.exists(img_nifti_path):
            img_nifti = nib.load(img_nifti_path)
            affine = img_nifti.affine
        else:
            affine = np.eye(4)

        mask_nifti = nib.Nifti1Image(mask_3d_volume, affine)
        output_mask_path = os.path.join(self.output_mask_dir, f"{patient_id}_mask.nii.gz")
        nib.save(mask_nifti, output_mask_path)

    def run_pipeline(self):
        print("Starting JSON parsing and 3D Mask generation...")
        patient_ids = [d for d in os.listdir(self.raw_dicom_dir) 
                    if os.path.isdir(os.path.join(self.raw_dicom_dir, d))]
        
        # یک شمارنده برای اینکه ببینیم چند بیمار واقعا پردازش شدند
        processed_count = 0 
        
        for pid in tqdm(patient_ids, desc="Processing Labels"):
            # چک میکنیم آیا ماسک ساخته شد یا نه
            expected_mask_path = os.path.join(self.output_mask_dir, f"{pid}_mask.nii.gz")
            
            self.process_patient(pid)
            
            if os.path.exists(expected_mask_path):
                processed_count += 1
                
        print(f"\nSuccessfully created masks for {processed_count} out of {len(patient_ids)} patients.")
            
        if self.master_labels:
            df_labels = pd.DataFrame(self.master_labels)
            csv_path = os.path.join(self.output_mask_dir, "master_labels_phase2.csv")
            df_labels.to_csv(csv_path, index=False)
            print(f"Labels extracted and saved to {csv_path}")

# ==========================================
# نحوه اجرای کد
# ==========================================
if __name__ == "__main__":
    RAW_DICOM = "Data/raw/training"
    RAW_JSON = "Data/raw/annotations"
    INTERIM_IMG = "data/interim/images" 
    OUT_MASK = "data/interim/masks"     
    
    labeler = JsonToNiftiLabeler(RAW_DICOM, RAW_JSON, INTERIM_IMG, OUT_MASK)
    labeler.run_pipeline()