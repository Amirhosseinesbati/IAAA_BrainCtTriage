import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from monai.transforms import Compose, Resize, NormalizeIntensity

# وارد کردن کلاس‌هایی که در فایل قبلی نوشتیم
# نکته: مطمئن شوید نام فایل قبلی شما mls_slice_selector.py است
from src.preprocessing.mls_slice_selector import SliceSelectorDataset, SliceSelectorModel

# ==========================================
# 1. تعریف ماژول لایتنینگ (مدیریت هوشمند آموزش)
# ==========================================
class SliceSelectorLitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
        # تابع Loss: چون مسئله 0 و 1 است (اسلایس هدف هست یا نیست)
        # استفاده از BCEWithLogitsLoss از نظر ریاضی پایدارترین گزینه در PyTorch است
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # استفاده از AdamW برای جلوگیری از Overfitting و همگرایی بهتر
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # کاهش نرخ یادگیری در صورت عدم پیشرفت (بسیار مهم برای مسابقات)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # محاسبه دقت (Accuracy) در حین آموزش
        # خروجی بیش از 0 در BCEWithLogits یعنی احتمال بالای 50%
        preds = (outputs > 0).float()
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        preds = (outputs > 0).float()
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

# ==========================================
# 2. آماده‌سازی داده‌ها و اجرای حلقه آموزش
# ==========================================
def run_training():
    print("Preparing Dataset and Dataloaders...")
    
    # مسیرها (تنظیم کنید تا با سیستم شما مطابقت داشته باشد)
    PROCESSED_CSV = "Data/processed/processed_dataset.csv"
    LABELS_CSV = "Data/interim/masks/master_labels_phase2.csv"
    
    # Transforms مشابه تستی که انجام دادیم
    transforms = Compose([
        Resize(spatial_size=(256, 256)), 
        NormalizeIntensity(nonzero=True) 
    ])
    
    # بارگذاری کل دیتاست
    full_dataset = SliceSelectorDataset(PROCESSED_CSV, LABELS_CSV, transform=transforms)
    
    # تقسیم دیتاست به 80% آموزش و 20% ارزیابی
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    # استفاده از Seed برای اینکه تقسیم‌بندی همیشه ثابت بماند
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # ساخت DataLoader ها
    # نکته: num_workers را بسته به قدرت پردازنده خود تنظیم کنید (2 تا 4 معمولا مناسب است)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} slices, Validating on {len(val_dataset)} slices.")

    # -------------------------
    # تنظیمات مدل و مربی (Trainer)
    # -------------------------
    base_model = SliceSelectorModel()
    lit_model = SliceSelectorLitModel(model=base_model, learning_rate=3e-4)
    
    # ایجاد پوشه برای ذخیره مدل
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # ذخیره خودکار بهترین مدل بر اساس کمترین val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='slice_selector_best_{epoch:02d}_{val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    # اگر مدل 5 ایپوک پشت سر هم پیشرفت نکرد، متوقف شود
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    # ساخت Trainerِ لایتنینگ
    # accelerator='auto' خودش گرافیک شما (GPU) را پیدا می‌کند
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto', 
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    # بوم! شروع آموزش :)
    print("Starting Training Pipeline...")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    run_training()