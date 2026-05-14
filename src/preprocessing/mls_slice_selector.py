import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from monai.transforms import Compose, RandRotate90, RandFlip, NormalizeIntensity, Resize
from tqdm import tqdm

class SliceSelectorDataset(Dataset):
    def __init__(self, processed_csv, labels_csv, mode='train', transform=None, neg_pos_ratio=3):
        """
        processed_csv: فایل خروجی فاز 3 (لیست تصاویر .npy)
        labels_csv: فایل master_labels_phase2.csv که شامل z_index نقاط است
        mode: 'train' یا 'val'
        neg_pos_ratio: برای هر اسلایس مثبت (دارای نقطه)، چند اسلایس منفی (بدون نقطه) برداریم؟
                    چون اکثر اسلایس‌های سر نقطه کلیدی ندارند، باید داده‌ها را بالانس کنیم.
        """
        self.mode = mode
        self.transform = transform
        
        df_processed = pd.read_csv(processed_csv)
        df_labels = pd.read_csv(labels_csv)
        
        # فیلتر کردن فقط لیبل‌های نقاط کلیدی
        df_keypoints = df_labels[df_labels['label_type'] == 'keypoint']
        
        # پیدا کردن اسلایس‌های مثبت (اسلایس‌هایی که دارای نقطه کلیدی هستند)
        # خروجی: دیکشنری {patient_id: [z_index1, z_index2, ...]}
        positive_slices = {}
        for _, row in df_keypoints.iterrows():
            pid = str(row['patient_id'])
            z = int(row['z_index'])
            if pid not in positive_slices:
                positive_slices[pid] = set()
            positive_slices[pid].add(z)
            
        self.samples = []
        
        # ساخت لیست نهایی داده‌ها (متشکل از اسلایس‌های 2D)
        for _, row in df_processed.iterrows():
            pid = str(row['patient_id'])
            img_path = row['processed_img_path']
            
            # خواندن ابعاد تصویر بدون لود کردن دیتای آن (سریع)
            shape = np.load(img_path, mmap_mode='r').shape
            depth = shape[3] # ابعاد: (3, H, W, D)
            
            pos_z_list = list(positive_slices.get(pid, []))
            
            # 1. اضافه کردن تمام اسلایس‌های مثبت
            for z in pos_z_list:
                # اگر در مرحله کراپ MONAI بُعد Z تغییر کرده باشه، باید مطمئن بشیم z از depth بیرون نزنه
                if z < depth: 
                    self.samples.append({"img_path": img_path, "z_index": z, "label": 1.0})
            
            # 2. اضافه کردن اسلایس‌های منفی (رندوم)
            if pos_z_list:
                # انتخاب رندوم اسلایس‌هایی که مثبت نیستند
                neg_candidates = [z for z in range(depth) if z not in pos_z_list]
                num_negatives = min(len(neg_candidates), len(pos_z_list) * neg_pos_ratio)
                
                # برای جلوگیری از تقلب مدل، منفی‌ها را ترجیحا از نزدیک اسلایس‌های مثبت برمی‌داریم 
                # (اینجا فعلا کاملا رندوم برداشتیم، برای سادگی)
                chosen_negs = np.random.choice(neg_candidates, num_negatives, replace=False)
                for z in chosen_negs:
                    self.samples.append({"img_path": img_path, "z_index": z, "label": 0.0})

        print(f"[{mode.upper()}] Dataset created with {len(self.samples)} 2D slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        z = sample["z_index"]
        label = sample["label"]
        
        # ترفند طلایی: استفاده از mmap_mode='c'
        volume = np.load(img_path, mmap_mode='c')
        
        # --- تغییر اینجاست ---
        # اضافه کردن .copy() برای جلوگیری از تداخل حافظه (Memory Contiguity)
        slice_2d = volume[:, :, :, z].copy() 
        # ---------------------
        
        slice_tensor = torch.tensor(slice_2d, dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
            
        return slice_tensor, label_tensor

# ==========================================
# معماری مدل (استفاده از Transfer Learning)
# ==========================================
class SliceSelectorModel(nn.Module):
    def __init__(self):
        super(SliceSelectorModel, self).__init__()
        
        # استفاده از EfficientNet-B0 به عنوان یک استخراج‌گر ویژگی بسیار قوی و سبک
        # چون در فاز ۳ تصاویر را ۳ کاناله (۳ ویندوی پزشکی) کردیم، می‌توانیم از وزن‌های ImageNet استفاده کنیم!
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # تغییر لایه آخر برای دسته‌بندی باینری (اسلایس هدف هست / نیست)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1) 
            # نکته: خروجی بدون Sigmoid است، چون از BCEWithLogitsLoss در آموزش استفاده خواهیم کرد 
            # (از نظر پایداری ریاضی در PyTorch بهتر است)
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# نحوه ساخت دیتالودر (برای تست)
# ==========================================
def test_pipeline():
    PROCESSED_CSV = "data/processed/processed_dataset.csv"
    LABELS_CSV = "data/interim/masks/master_labels_phase2.csv"
    
    # Augmentation ساده برای تصاویر 2D
    transforms = Compose([
        # همه تصاویر را دقیقاً به 256 در 256 پیکسل تغییر سایز می‌دهیم
        Resize(spatial_size=(256, 256)), 
        # تصاویر پزشکی معمولا باید Normalize شوند
        NormalizeIntensity(nonzero=True) 
    ])
    
    dataset = SliceSelectorDataset(PROCESSED_CSV, LABELS_CSV, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    model = SliceSelectorModel()
    
    # گرفتن یک Batch برای تست
    for images, labels in dataloader:
        print(f"Batch Image Shape: {images.shape}") # باید (16, 3, H, W) باشد
        print(f"Batch Label Shape: {labels.shape}") # باید (16, 1) باشد
        
        # عبور دادن از مدل
        outputs = model(images)
        print(f"Model Output Shape: {outputs.shape}") # باید (16, 1) باشد
        break
        
if __name__ == "__main__":
    # اگر فایل‌های CSV در مسیر درست باشند، این تابع بدون خطا اجرا می‌شود.
    test_pipeline()
    pass