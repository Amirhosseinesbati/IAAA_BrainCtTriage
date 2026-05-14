import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# وارد کردن کلاس‌های فایل قبلی
from src.training.mls_keypoint_detector import KeypointDataset, KeypointModel

# ==========================================
# 1. ماژول لایتنینگ برای Keypoint
# ==========================================
class KeypointLitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
        # بهترین Loss برای مسائل تشخیص نقطه
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }

    def _calculate_pixel_error(self, preds, targets):
        """
        یک تابع کمکی برای محاسبه اینکه مدل به طور میانگین چند پیکسل خطا دارد.
        چون خروجی بین 0 و 1 است، آن را ضرب در 512 میکنیم تا به پیکسل واقعی برگردد.
        """
        # محاسبه فاصله مطلق بین پیش‌بینی و واقعیت
        abs_diff = torch.abs(preds - targets)
        # تبدیل به پیکسل
        pixel_error = abs_diff * 512.0
        # میانگین خطای تمام ۶ نقطه
        return pixel_error.mean()

    def training_step(self, batch, batch_idx):
        images, coords = batch
        outputs = self(images)
        loss = self.criterion(outputs, coords)
        
        # محاسبه خطای ملموس (پیکسل)
        pix_error = self._calculate_pixel_error(outputs, coords)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_pix_err', pix_error, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, coords = batch
        outputs = self(images)
        loss = self.criterion(outputs, coords)
        
        pix_error = self._calculate_pixel_error(outputs, coords)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pix_err', pix_error, on_step=False, on_epoch=True, prog_bar=True)
        return loss

# ==========================================
# 2. حلقه آموزش
# ==========================================
def run_training():
    print("Preparing Keypoint Dataset...")
    
    INTERIM_IMG = "data/interim/images"
    LABELS_CSV = "data/interim/masks/master_labels_phase2.csv"
    META_CSV = "data/interim/images/metadata_phase1.csv"
    
    # 🚨 نکته حرفه‌ای: در مسائل Keypoint، ما معمولا Augmentation فضایی (مثل چرخش و فلیپ) 
    # اعمال نمیکنیم مگر اینکه کدی بنویسیم که مختصات ریاضی نقاط را هم با تصویر بچرخاند.
    # برای حفظ دقت بالا در این مسابقه، ما تصاویر را بدون تغییر (فقط با Windowing) به مدل می‌دهیم.
    
    full_dataset = KeypointDataset(INTERIM_IMG, LABELS_CSV, META_CSV)
    
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size) # 20% برای ارزیابی
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Batch size را 16 یا 8 قرار می‌دهیم چون تصاویر 512x512 هستند و VRAM بیشتری می‌خواهند
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} slices, Validating on {len(val_dataset)} slices.")

    base_model = KeypointModel()
    lit_model = KeypointLitModel(model=base_model)
    
    os.makedirs("models/checkpoints", exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='keypoint_best_{epoch:02d}_{val_pix_err:.1f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=8, # کمی بیشتر از قبلی، چون رگرسیون زمان بیشتری برای همگرایی نیاز دارد
        mode='min',
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=40,
        accelerator='auto', 
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5
    )

    print("Starting Keypoint Training Pipeline...")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    run_training()