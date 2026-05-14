import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

class KeypointDataset(Dataset):
    def __init__(self, interim_img_dir, labels_csv, meta_csv, transform=None):
        self.interim_img_dir = interim_img_dir
        self.transform = transform
        
        # لود کردن متادیتا (برای استخراج Slope و Intercept)
        self.df_meta = pd.read_csv(meta_csv)
        self.df_meta.set_index("patient_id", inplace=True)
        
        # لود کردن لیبل‌ها
        df_labels = pd.read_csv(labels_csv)
        df_keypoints = df_labels[df_labels['label_type'] == 'keypoint']
        
        # گروه بندی بر اساس بیمار و اسلایس
        grouped = df_keypoints.groupby(['patient_id', 'z_index'])
        
        self.samples = []
        for (pid, z), group in grouped:
            # بررسی اینکه آیا هر 3 نقطه در این اسلایس وجود دارند؟
            if len(group) == 3:
                # استخراج مختصات
                points = {}
                for _, row in group.iterrows():
                    points[row['name']] = (row['x'], row['y'])
                
                # ترتیب نقاط برای خروجی شبکه (بسیار مهم که همیشه ثابت باشد)
                # [x1, y1, x2, y2, x3, y3]
                try:
                    target_coords = [
                        points['AnteriorFalxAttachment'][0],   points['AnteriorFalxAttachment'][1],
                        points['PosteriorFalxAttachment'][0],  points['PosteriorFalxAttachment'][1],
                        points['OutermostPointOfTheFalx'][0],  points['OutermostPointOfTheFalx'][1]
                    ]
                    
                    self.samples.append({
                        "patient_id": str(pid),
                        "z_index": int(z),
                        "coords": target_coords
                    })
                except KeyError:
                    # اگر اسم نقاط فرق داشت یا ناقص بود، اسکیپ می‌کنیم
                    continue
                    
        print(f"Keypoint Dataset created with {len(self.samples)} valid slices (having all 3 points).")

    def _apply_windowing(self, img_array, patient_id):
        # این همان منطق فاز 3 شماست، اما فقط روی یک اسلایس 2D اعمال می‌شود
        try:
            slope = self.df_meta.loc[int(patient_id), "rescale_slope"]
            intercept = self.df_meta.loc[int(patient_id), "rescale_intercept"]
        except:
            slope, intercept = 1.0, 0.0
            
        hu_img = (img_array * slope) + intercept

        def window(img, w, l):
            min_val = l - (w / 2)
            max_val = l + (w / 2)
            windowed = np.clip(img, min_val, max_val)
            return (windowed - min_val) / (max_val - min_val)

        # 3 کانال (Brain, Subdural, Bone)
        ch1 = window(hu_img, w=80, l=40)
        ch2 = window(hu_img, w=200, l=80)
        ch3 = window(hu_img, w=1000, l=400)
        return np.stack([ch1, ch2, ch3], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["patient_id"]
        z = sample["z_index"]
        coords = np.array(sample["coords"], dtype=np.float32)
        
        # لود کردن فایل NIfTI
        nii_path = os.path.join(self.interim_img_dir, f"{pid}.nii.gz")
        
        # ترفند سرعت: به جای لود کردن کل حجم با nibabel، از dataobj استفاده میکنیم 
        # تا فقط همان اسلایس Z مورد نظر در RAM لود شود
        img_obj = nib.load(nii_path)
        slice_2d = img_obj.dataobj[:, :, z] # Shape: (512, 512)
        
        # اعمال ویندوینگ (تبدیل به 3 کانال)
        multi_channel_img = self._apply_windowing(slice_2d, pid)
        
        # --- نرمال‌سازی طلایی مختصات ---
        # شبکه عصبی با اعداد بین 0 و 1 خیلی بهتر کار می‌کند تا اعداد بزرگ مثل 350 یا 480
        # چون تصویر ما 512x512 است، مختصات را تقسیم بر 512 می‌کنیم.
        # در زمان Inference، خروجی مدل را ضرب در 512 خواهیم کرد!
        coords_normalized = coords / 512.0
        
        img_tensor = torch.tensor(multi_channel_img, dtype=torch.float32)
        coords_tensor = torch.tensor(coords_normalized, dtype=torch.float32)
        
        return img_tensor, coords_tensor

# ==========================================
# معماری مدل Keypoint Regression
# ==========================================
class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        # برای Keypoint، از ResNet34 استفاده می‌کنیم چون برای رگرسیونِ پیوسته پایداری بهتری دارد
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # جایگزین کردن لایه آخر
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            # خروجی دقیقاً 6 عدد است (X1, Y1, X2, Y2, X3, Y3)
            nn.Linear(256, 6),
            # استفاده از Sigmoid حیاتی است! چون مختصات را بین 0 و 1 نرمال کردیم، 
            # Sigmoid مدل را مجبور می‌کند همیشه عددی در همین بازه تولید کند.
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# تست سریع (Sanity Check)
# ==========================================
if __name__ == "__main__":
    INTERIM_IMG = "Data/interim/images"
    LABELS_CSV = "Data/interim/masks/master_labels_phase2.csv"
    META_CSV = "Data/interim/images/metadata_phase1.csv"
    
    dataset = KeypointDataset(INTERIM_IMG, LABELS_CSV, META_CSV)
    
    if len(dataset) > 0:
        img, coords = dataset[0]
        print(f"Input Image Shape: {img.shape}") # باید (3, 512, 512) باشد
        print(f"Target Coords (Normalized): {coords}") # باید 6 عدد بین 0 و 1 باشد
        
        # تست عبور از مدل
        model = KeypointModel()
        # اضافه کردن بُعد Batch Size مجازی (1, 3, 512, 512)
        dummy_out = model(img.unsqueeze(0))
        print(f"Model Output Shape: {dummy_out.shape}") # باید (1, 6) باشد
    else:
        print("No valid keypoint samples found. Please check paths.")