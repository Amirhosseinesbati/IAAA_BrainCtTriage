import os
import random
import pandas as pd
import numpy as np
import cv2
import ast
import nibabel as nib
from tqdm import tqdm

class YoloDatasetBuilder:
    def __init__(self, interim_img_dir, master_csv, metadata_csv, output_dir, split_ratio=0.8):
        self.interim_img_dir = interim_img_dir
        self.master_csv = master_csv
        self.metadata_csv = metadata_csv
        self.output_dir = output_dir 
        self.split_ratio = split_ratio
        
        self.df_meta = pd.read_csv(self.metadata_csv)
        self.df_meta.set_index("patient_id", inplace=True)

    def setup_directories(self):
        dirs = ["images/train", "images/val", "labels/train", "labels/val"]
        for d in dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def apply_bone_window(self, slice_array, patient_id):
        try:
            slope = self.df_meta.loc[int(patient_id), "rescale_slope"]
            intercept = self.df_meta.loc[int(patient_id), "rescale_intercept"]
        except KeyError:
            slope, intercept = 1.0, 0.0
            
        hu_img = (slice_array * slope) + intercept
        w, l = 1000, 400
        min_val = l - (w / 2)
        max_val = l + (w / 2)
        windowed = np.clip(hu_img, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val) 
        return (windowed * 255).astype(np.uint8)

    def build(self):
        print("Building Dataset with Hard Negatives & Pure Negatives...")
        self.setup_directories()
        
        # 1. خواندن بیماران دارای شکستگی
        df_labels = pd.read_csv(self.master_csv)
        fractures = df_labels[df_labels['name'] == 'fracture']
        
        pos_dict = {}
        for _, row in fractures.iterrows():
            pid = str(row['patient_id'])
            z = int(row['z_index'])
            box = ast.literal_eval(row['box_data']) if isinstance(row['box_data'], str) else row['box_data']
            if pid not in pos_dict: pos_dict[pid] = {}
            if z not in pos_dict[pid]: pos_dict[pid][z] = []
            pos_dict[pid][z].append(box)

        fractured_patients = list(pos_dict.keys())
        
        # 2. پیدا کردن تمام بیماران و استخراج بیماران کاملا سالم
        all_nifti_files = [f for f in os.listdir(self.interim_img_dir) if f.endswith(".nii.gz") and not f.endswith("_mask.nii.gz")]
        all_patient_ids = [f.replace(".nii.gz", "") for f in all_nifti_files]
        healthy_patients = [pid for pid in all_patient_ids if pid not in fractured_patients]

        # 3. تقسیم‌بندی (Split) جداگانه برای حفظ تعادل
        random.seed(42)
        
        random.shuffle(fractured_patients)
        f_split = int(len(fractured_patients) * self.split_ratio)
        train_fractured = set(fractured_patients[:f_split])
        val_fractured = set(fractured_patients[f_split:])

        random.shuffle(healthy_patients)
        h_split = int(len(healthy_patients) * self.split_ratio)
        train_healthy = set(healthy_patients[:h_split])
        val_healthy = set(healthy_patients[h_split:])
        
        train_patients = train_fractured.union(train_healthy)
        
        print(f"Fractured Patients - Train: {len(train_fractured)} | Val: {len(val_fractured)}")
        print(f"Healthy Patients   - Train: {len(train_healthy)} | Val: {len(val_healthy)}")

        for nifti_file in tqdm(all_nifti_files, desc="Extracting Slices"):
            patient_id = nifti_file.replace(".nii.gz", "")
            subset = "train" if patient_id in train_patients else "val"
            
            img_out_dir = os.path.join(self.output_dir, "images", subset)
            lbl_out_dir = os.path.join(self.output_dir, "labels", subset)

            nifti_path = os.path.join(self.interim_img_dir, nifti_file)
            img_nifti = nib.load(nifti_path)
            volume = img_nifti.get_fdata()
            depth = volume.shape[2]
            
            slices_to_process = []
            is_fractured = patient_id in fractured_patients
            
            if is_fractured:
                patient_fractures = pos_dict.get(patient_id, {})
                positive_slices = list(patient_fractures.keys())
                
                # به جای ۵ اسلایس سالم، فقط ۱ یا نهایت ۲ اسلایس سالم از بیمار شکسته برمی‌داریم
                all_slices = list(range(depth))
                negative_slices = [z for z in all_slices if z not in positive_slices]
                selected_negatives = random.sample(negative_slices, min(1, len(negative_slices))) # تغییر به 1
                slices_to_process = positive_slices + selected_negatives
                
            else:
                # برای بیماران کاملاً سالم:
                # با احتمال ۲۰ درصد اصلاً از این بیمار عکسی برنمی‌داریم تا دیتاسِت پر از عکس سالم نشود
                if random.random() > 0.2: 
                    # اگر انتخاب شد، فقط ۱ عکس تصادفی از او برمی‌داریم
                    all_slices = list(range(depth))
                    slices_to_process = random.sample(all_slices, min(1, len(all_slices))) # تغییر به 1
                else:
                    slices_to_process = []

            for z in slices_to_process:
                slice_2d = volume[:, :, z]
                bone_img_8bit = self.apply_bone_window(slice_2d, patient_id)
                slice_rgb = cv2.cvtColor(bone_img_8bit, cv2.COLOR_GRAY2RGB)
                
                filename = f"{patient_id}_z{z}"
                img_save_path = os.path.join(img_out_dir, f"{filename}.jpg")
                txt_save_path = os.path.join(lbl_out_dir, f"{filename}.txt")
                
                cv2.imwrite(img_save_path, slice_rgb)
                
                with open(txt_save_path, 'w') as f:
                    if is_fractured and z in positive_slices:
                        boxes = pos_dict[patient_id][z]
                        for box in boxes:
                            x_min, y_min, w, h = box
                            x_center = x_min + (w / 2.0)
                            y_center = y_min + (h / 2.0)
                            x_c_norm, y_c_norm = x_center / 512, y_center / 512
                            w_norm, h_norm = w / 512, h / 512
                            
                            if 0 < x_c_norm < 1 and 0 < y_c_norm < 1 and 0 < w_norm < 1 and 0 < h_norm < 1:
                                f.write(f"0 {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # ساخت dataset.yaml
        yaml_content = f"""
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

names:
    0: fracture
"""
        with open(os.path.join(self.output_dir, "dataset.yaml"), "w" , encoding="utf-8") as f:
            f.write(yaml_content)
            
        print("Dataset generated successfully!")

if __name__ == "__main__":
    INTERIM_DIR = "data/interim/images"
    MASTER_CSV = "data/interim/masks/master_labels_phase2.csv"
    META_CSV = "data/interim/images/metadata_phase1.csv"
    OUT_YOLO_DIR = "data/processed/yolo_fracture"
    
    builder = YoloDatasetBuilder(INTERIM_DIR, MASTER_CSV, META_CSV, OUT_YOLO_DIR)
    builder.build()