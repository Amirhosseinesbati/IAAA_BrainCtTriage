import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, RandAffined, RandFlipd, RandGaussianNoised

class BrainCTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        csv_file: مسیر فایل processed_dataset.csv که در فاز 3 ساخته شد
        transform: توابع Augmentation (افزایش داده) که فقط در زمان Train اعمال می‌شوند
        """
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 1. خواندن مسیر فایل‌ها از روی CSV
        img_path = self.data_info.iloc[idx]['processed_img_path']
        mask_path = self.data_info.iloc[idx]['processed_mask_path']
        
        # 2. بارگذاری فوق‌سریع فایل‌های .npy در RAM
        # numpy خیلی سریع این کار را انجام می‌دهد
        image = np.load(img_path)  # ابعاد: (3, H, W, D)
        mask = np.load(mask_path)  # ابعاد: (H, W, D)
        
        # ماسک باید یک بعد Channel داشته باشد (برای MONAI و PyTorch) -> (1, H, W, D)
        mask = np.expand_dims(mask, axis=0)
        
        # 3. تبدیل به فرمت دیکشنری (استاندارد شبکه‌های پزشکی)
        data_dict = {
            "image": image.astype(np.float32),
            "mask": mask.astype(np.float32) # معمولا Loss functionها با float کار میکنند
        }
        
        # 4. اعمال Data Augmentation (در صورت وجود)
        if self.transform:
            data_dict = self.transform(data_dict)
            
        # 5. تبدیل نهایی به Torch Tensor
        data_dict["image"] = torch.as_tensor(data_dict["image"])
        data_dict["mask"] = torch.as_tensor(data_dict["mask"])
        
        return data_dict

# ==========================================
# نحوه استفاده از این Dataset در اسکریپت آموزش (Train)
# ==========================================
def get_dataloaders(csv_path, batch_size=2):
    """
    این تابع دیتالودرهای قدرتمند را برای شما می‌سازد
    """
    
    # تعریف Augmentation برای جلوگیری از Overfitting (فقط برای داده‌های آموزشی)
    # MONAI به صورت هوشمند چرخش و نویز را همزمان روی Image و Mask اعمال می‌کند!
    train_transforms = Compose([
        RandFlipd(keys=["image", "mask"], spatial_axis=[0], prob=0.5), # قرینه کردن رندوم
        RandAffined(keys=["image", "mask"], prob=0.5, rotate_range=(0.1, 0.1, 0.1)), # چرخش کم
        RandGaussianNoised(keys=["image"], prob=0.2) # نویز فقط روی تصویر اعمال می‌شود، نه ماسک!
    ])
    
    # ساخت نمونه از Dataset
    train_ds = BrainCTDataset(csv_file=csv_path, transform=train_transforms)
    
    # ساخت DataLoader با تنظیمات طلایی MLOps
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,        # تعداد هسته‌های CPU که دیتا را موازی لود می‌کنند (بسته به سیستم خودتون تغییر بدید)
        pin_memory=True,      # انتقال سریع‌تر دیتا از RAM به VRAM گرافیک
        prefetch_factor=2     # همیشه 2 بچ (Batch) بعدی را در صف آماده نگه دار!
    )
    
    return train_loader

if __name__ == "__main__":
    # تست دیتالودر
    CSV_PATH = "data/processed/processed_dataset.csv"
    
    loader = get_dataloaders(CSV_PATH, batch_size=1)
    
    print("Starting DataLoader Test...")
    for batch_data in loader:
        imgs = batch_data["image"]
        masks = batch_data["mask"]
        
        print(f"Batch Image Shape: {imgs.shape}") # باید (Batch, 3, H, W, D) باشد
        print(f"Batch Mask Shape: {masks.shape}")   # باید (Batch, 1, H, W, D) باشد
        print(f"Image Data Type: {imgs.dtype}")     # باید torch.float32 باشد
        print("DataLoader works perfectly! GPU is ready to eat data!")
        break # یک Batch برای تست کافیست