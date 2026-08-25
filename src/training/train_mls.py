import os
import tempfile
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pathlib import Path

from src.training.mls_models import SliceSelectorModel, KeypointModel
from src.config import (
    IMG_SIZE, IMG_SIZE_MLS_SELECTOR, MLFLOW_EXP_MLS_KEYPOINT,
    MLFLOW_EXP_MLS_SELECTOR, MLS_DEFAULTS, RANDOM_SEED,
)
from src.mlops.tracking import build_source_snapshot

# ==========================================
# 1. دیتاسِت‌های فوق سریع مبتنی بر PNG
# ==========================================
class FastMlsDataset(Dataset):
    def __init__(self, csv_path, img_dir, task="slice_selector"):
        """
        task: اگر "slice_selector" باشد، همه عکس‌ها (0 و 1) را برمی‌گرداند.
            اگر "keypoint" باشد، فقط عکس‌های هدف (1) را برمی‌گرداند.
        """
        df = pd.read_csv(csv_path)
        if task == "keypoint":
            df = df[df['is_target'] == 1].reset_index(drop=True)
            
        self.data = df
        self.img_dir = img_dir
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # خواندن سریع PNG و تبدیل از BGR (استاندارد OpenCV) به RGB
        img_path = os.path.join(self.img_dir, row['image_name'])
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # اگر عکس برای SliceSelector لازم است، معمولا آن را 256x256 میکنیم تا سریعتر آموزش ببیند
        if self.task == "slice_selector":
            img_rgb = cv2.resize(img_rgb, (IMG_SIZE_MLS_SELECTOR, IMG_SIZE_MLS_SELECTOR))
            
        # تبدیل به Tensor و نرمال‌سازی بین 0 و 1
        # شکل تصویر از (H, W, C) باید بشود (C, H, W)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        if self.task == "slice_selector":
            label = torch.tensor([row['is_target']], dtype=torch.float32)
            return img_tensor, label
            
        elif self.task == "keypoint":
            # مختصات در CSV ذخیره شده، فقط باید نرمال (تقسیم بر 512) شوند
            coords = [row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3']]
            coords_tensor = torch.tensor(coords, dtype=torch.float32) / float(IMG_SIZE)
            return img_tensor, coords_tensor

# ==========================================
# 2. ماژول‌های Lightning (بسیار تمیز شده)
# ==========================================
class SliceSelectorLit(pl.LightningModule):
    def __init__(self, model, lr=3e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = ((y_hat > 0).float() == y).float().mean()
        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = ((y_hat > 0).float() == y).float().mean()
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

class KeypointLit(pl.LightningModule):
    def __init__(self, model, lr=2e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        pix_err = (torch.abs(y_hat - y) * float(IMG_SIZE)).mean()
        self.log_dict({'train_loss': loss, 'train_pix_err': pix_err}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        pix_err = (torch.abs(y_hat - y) * float(IMG_SIZE)).mean()
        self.log_dict({'val_loss': loss, 'val_pix_err': pix_err}, prog_bar=True)




def get_mlflow_logger(experiment_name):
    """تابع کمکی برای ساخت لاگر یکپارچه با تنظیمات ابری (DagsHub)"""
    
    # گرفتن آدرس ترکینگ از متغیرهای محیطی که ZenML یا Vast ساخته است
    # اگر پیدا نشد، به صورت پیش‌فرض None می‌فرستد تا از اکتیو ران استفاده کند
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri, # هدایت لاگ‌ها به DagsHub
        log_model=True 
    )


def _log_legacy_artifacts(logger, checkpoint: Path) -> None:
    """Attach legacy checkpoints and the configured source snapshot to its run."""
    if checkpoint.exists():
        logger.experiment.log_artifact(logger.run_id, str(checkpoint), artifact_path="models")
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot = Path(temp_dir) / "project_source.zip"
        build_source_snapshot(snapshot)
        logger.experiment.log_artifact(logger.run_id, str(snapshot), artifact_path="source_snapshot")

# ==========================================
# 3. توابع اجرای آموزش
# ==========================================
def train_slice_selector(csv_path, img_dir, save_dir):
    print("--- Training MLS Slice Selector ---")
    dataset = FastMlsDataset(csv_path, img_dir, task="slice_selector")
    train_sz = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_sz, len(dataset)-train_sz], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    batch_size = int(MLS_DEFAULTS["selector_batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    # اضافه کردن لاگر
    mlf_logger = get_mlflow_logger(MLFLOW_EXP_MLS_SELECTOR)
    
    model = SliceSelectorLit(SliceSelectorModel())
    
    callbacks = [
        ModelCheckpoint(dirpath=save_dir, filename='slice_selector_best', monitor='val_loss', mode='min', save_top_k=1),
        EarlyStopping(monitor='val_loss', patience=5, mode='min')
    ]
    trainer = pl.Trainer(max_epochs=int(MLS_DEFAULTS["selector_epochs"]), accelerator='auto', callbacks=callbacks, logger=mlf_logger)
    trainer.fit(model, train_loader, val_loader)

    # لاگ صریح فایل checkpoint به MLflow (علاوه بر log_model=True لایتنینگ)
    best_ckpt = Path(save_dir) / "slice_selector_best.ckpt"
    _log_legacy_artifacts(mlf_logger, best_ckpt)

def train_keypoint_detector(csv_path, img_dir, save_dir):
    print("--- Training MLS Keypoint Detector ---")
    dataset = FastMlsDataset(csv_path, img_dir, task="keypoint")
    train_sz = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_sz, len(dataset)-train_sz], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    batch_size = int(MLS_DEFAULTS["keypoint_batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = KeypointLit(KeypointModel())


    # اضافه کردن لاگر
    mlf_logger = get_mlflow_logger(MLFLOW_EXP_MLS_KEYPOINT)
    
    callbacks = [
        ModelCheckpoint(dirpath=save_dir, filename='keypoint_best', monitor='val_pix_err', mode='min', save_top_k=1),
        EarlyStopping(monitor='val_loss', patience=8, mode='min')
    ]
    trainer = pl.Trainer(max_epochs=int(MLS_DEFAULTS["keypoint_epochs"]), accelerator='auto', callbacks=callbacks, logger=mlf_logger)
    trainer.fit(model, train_loader, val_loader)

    # لاگ صریح فایل checkpoint به MLflow
    best_ckpt = Path(save_dir) / "keypoint_best.ckpt"
    _log_legacy_artifacts(mlf_logger, best_ckpt)



if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    mls_csv = str(BASE_DIR / "Data" / "processed" / "mls_dataset" / "mls_labels.csv")
    mls_img = str(BASE_DIR / "Data" / "processed" / "mls_dataset" / "images")
    ckpt_dir = str(BASE_DIR / "models" / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # آموزش مدل A (پیدا کردن بهترین اسلایس)
    train_slice_selector(mls_csv, mls_img, ckpt_dir)
    
    # آموزش مدل B (پیدا کردن کی‌پوینت‌ها)
    train_keypoint_detector(mls_csv, mls_img, ckpt_dir)
