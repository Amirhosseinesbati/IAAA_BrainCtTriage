import os
import json
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

class NNUnetDatasetBuilder:
    def __init__(self, raw_dicom_dir, raw_json_dir, nnunet_raw_dir, dataset_id=501, dataset_name="BrainICH"):
        self.raw_dicom_dir = raw_dicom_dir
        self.raw_json_dir = raw_json_dir
        
        # ساخت مسیرهای استاندارد nnU-Net
        self.dataset_name = f"Dataset{dataset_id}_{dataset_name}"
        self.out_dir = os.path.join(nnunet_raw_dir, self.dataset_name)
        self.imagesTr = os.path.join(self.out_dir, "imagesTr")
        self.labelsTr = os.path.join(self.out_dir, "labelsTr")
        
        os.makedirs(self.imagesTr, exist_ok=True)
        os.makedirs(self.labelsTr, exist_ok=True)

    def build(self):
        print("Building nnU-Net Dataset directly from DICOM & JSON...")
        patient_ids = [d for d in os.listdir(self.raw_dicom_dir) 
                    if os.path.isdir(os.path.join(self.raw_dicom_dir, d))]
        
        processed_count = 0
        
        for pid in tqdm(patient_ids, desc="Processing Patients for nnU-Net"):
            dicom_patient_dir = os.path.join(self.raw_dicom_dir, pid)
            json_patient_dir = os.path.join(self.raw_json_dir, pid)
            
            # اگر پوشه جیسون وجود ندارد کلا رد شو
            if not os.path.exists(json_patient_dir):
                continue
                
            # 1. خواندن دایکام با ابزار هسته
            try:
                reader = BrainDicomReader(dicom_patient_dir).load_and_sort()
            except ValueError:
                continue # پوشه دایکام خالی است
                
            # 2. خواندن لیبل‌ها با ابزار هسته
            parser = AnnotationParser(json_patient_dir)
            masks_3d_list = []
            has_any_label = False
            
            for dcm_slice in reader.slices:
                dcm_filename = os.path.basename(dcm_slice.filename)
                slice_data = parser.parse_slice(dcm_filename)
                
                masks_3d_list.append(slice_data["mask_2d"])
                if slice_data["has_label"] :
                    has_any_label = True
                    
            # 3. اگر هیچ ماسکی برای این بیمار نبود، رد می‌شویم
            if not has_any_label:
                continue
                
            # 4. استخراج حجم تصویر و ساخت ماسک سه بعدی
            image_3d = reader.get_3d_volume_hu()
            mask_3d = np.stack(masks_3d_list, axis=-1)
            
            # 5. ساخت Affine Matrix (برای مقیاس فیزیکی)
            meta = reader.metadata
            affine = np.diag([meta["spacing_x"], meta["spacing_y"], meta["spacing_z"], 1.0])
            
            # 6. ذخیره فایل‌ها به فرمت nnU-Net
            img_nifti = nib.Nifti1Image(image_3d, affine)
            mask_nifti = nib.Nifti1Image(mask_3d, affine)
            
            nib.save(img_nifti, os.path.join(self.imagesTr, f"BRN_{pid}_0000.nii.gz"))
            nib.save(mask_nifti, os.path.join(self.labelsTr, f"BRN_{pid}.nii.gz"))
            processed_count += 1
            
        self._generate_dataset_json(processed_count)
        print(f"\nSuccessfully built nnU-Net dataset for {processed_count} patients.")

    def _generate_dataset_json(self, num_training):
        dataset_info = {
            "channel_names": {"0": "CT"},
            "labels": {
                "background": 0, "IVH": 1, "IPH": 2, "SDH": 3, "EDH": 4, "SAH": 5
            },
            "numTraining": num_training,
            "file_ending": ".nii.gz"
        }
        with open(os.path.join(self.out_dir, "dataset.json"), 'w') as f:
            json.dump(dataset_info, f, indent=4)