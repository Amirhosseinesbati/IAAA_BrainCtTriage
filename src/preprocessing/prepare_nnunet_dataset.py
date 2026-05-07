import os
import shutil
import json
from tqdm import tqdm

class NNUnetDatasetBuilder:
    def __init__(self, interim_img_dir, interim_mask_dir, nnunet_raw_dir, dataset_id=501, dataset_name="BrainICH"):
        self.interim_img_dir = interim_img_dir
        self.interim_mask_dir = interim_mask_dir
        
        # ساخت مسیر استاندارد nnU-Net
        self.dataset_name = f"Dataset{dataset_id}_{dataset_name}"
        self.out_dir = os.path.join(nnunet_raw_dir, self.dataset_name)
        
        self.imagesTr = os.path.join(self.out_dir, "imagesTr")
        self.labelsTr = os.path.join(self.out_dir, "labelsTr")
        
        os.makedirs(self.imagesTr, exist_ok=True)
        os.makedirs(self.labelsTr, exist_ok=True)

    def copy_and_rename_files(self):
        print("Copying and renaming files to nnU-Net format...")
        # خواندن فایل‌های تصویر از فاز 1
        img_files = [f for f in os.listdir(self.interim_img_dir) if f.endswith(".nii.gz") and not f.endswith("_mask.nii.gz")]
        
        for img_file in tqdm(img_files):
            patient_id = img_file.replace(".nii.gz", "")
            mask_file = f"{patient_id}_mask.nii.gz"
            mask_path = os.path.join(self.interim_mask_dir, mask_file)
            
            if not os.path.exists(mask_path):
                continue # اگر بیماری ماسک نداشت، رد می‌شویم
                
            # نام‌گذاری جدید طبق استاندارد nnUNet
            # تصویر: BRN_272179_0000.nii.gz
            # ماسک: BRN_272179.nii.gz
            new_img_name = f"BRN_{patient_id}_0000.nii.gz"
            new_mask_name = f"BRN_{patient_id}.nii.gz"
            
            shutil.copy(os.path.join(self.interim_img_dir, img_file), 
                        os.path.join(self.imagesTr, new_img_name))
            
            shutil.copy(mask_path, 
                        os.path.join(self.labelsTr, new_mask_name))

    def generate_dataset_json(self):
        """ساخت فایل dataset.json که برای nnU-Net الزامی است"""
        # نکته: اعداد لیبل‌ها باید دقیقا مطابق کدهای شما در فاز 2 باشد
        # در اینجا فرض کردم: 1=EDH, 2=SDH, 3=IPH, 4=SAH, 5=IVH
        dataset_info = {
            "channel_names": {
                "0": "CT"  # نوع تصویر
            },
            "labels": {
                "background": 0,
                "IVH": 1,
                "IPH": 2,
                "SDH": 3,
                "EDH": 4,
                "SAH": 5
            },
            "numTraining": len(os.listdir(self.imagesTr)),
            "file_ending": ".nii.gz"
        }
        
        json_path = os.path.join(self.out_dir, "dataset.json")
        with open(json_path, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        print(f"dataset.json generated at {json_path}")

if __name__ == "__main__":
    # مسیرها را با سیستم خود تطبیق دهید
    INTERIM_IMAGES = "Data/interim/images"
    INTERIM_MASKS = "Data/interim/masks"
    NNUNET_RAW = "Data/nn-unet/nnUNet_raw" # همان مسیری که در Environment Variable تنظیم کردید
    
    builder = NNUnetDatasetBuilder(INTERIM_IMAGES, INTERIM_MASKS, NNUNET_RAW)
    builder.copy_and_rename_files()
    builder.generate_dataset_json()