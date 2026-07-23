"""
monai/train.py — MONAI 3D training loop for ICH segmentation.

Sets up a MONAI network, loss, optimizer, and runs a standard
PyTorch training/validation loop with MLflow logging.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import MLFLOW_EXPERIMENT_PREFIX, ICH_LABELS, log_src_snapshot
from src.strategies.config_models import MONAIConfig
from src.strategies.monai.dataset import create_monai_dataloaders

logger = logging.getLogger(__name__)

NUM_CLASSES = len(ICH_LABELS)  # 6 including background


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a MONAI network by name."""
    if model_name == "UNETR":
        from monai.networks.nets import UNETR
        return UNETR(
            in_channels=1, out_channels=num_classes,
            img_size=(128, 128, 128),  # default; patched at runtime
            feature_size=16, hidden_size=768,
            mlp_dim=3072, num_heads=12,
            pos_embed="perceptron", norm_name="instance",
            res_block=True, dropout_rate=0.0,
        )
    elif model_name == "SwinUNETR":
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=1, out_channels=num_classes,
            feature_size=48, use_checkpoint=False,
        )
    elif model_name == "SegResNet":
        from monai.networks.nets import SegResNet
        return SegResNet(
            spatial_dims=3,
            in_channels=1, out_channels=num_classes,
            init_filters=16, blocks_down=(1, 2, 2, 4),
            dropout_prob=0.1,
        )
    elif model_name == "DynUNet":
        from monai.networks.nets import DynUNet
        return DynUNet(
            spatial_dims=3,
            in_channels=1, out_channels=num_classes,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=[32, 64, 128, 256, 320],
            dropout=0.1,
            deep_supervision=True,
            deep_sup_num=3,
            res_block=True,
        )
    else:
        raise ValueError(f"Unknown MONAI model: {model_name}")


def train_monai(config: MONAIConfig) -> None:
    """
    Train a MONAI 3D segmentation model with MLflow logging.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    experiment_name = f"{MLFLOW_EXPERIMENT_PREFIX}_ICH_monai"
    mlflow.set_experiment(experiment_name)
    run_name = f"{config.model}_bs{config.batch_size}_roi{config.roi_size}"

    with mlflow.start_run(run_name=run_name) as _:
        mlflow.log_params(config.model_dump())
        mlflow.set_tag("strategy", "monai")
        mlflow.set_tag("model", config.model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Data
        train_loader = create_monai_dataloaders(config, split="train")
        val_loader = create_monai_dataloaders(config, split="val")

        # Model
        model = _build_model(config.model, NUM_CLASSES).to(device)

        # Loss & Optimizer
        from monai.losses import DiceCELoss
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)

        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

        # Checkpoint dir
        ckpt_dir = BASE_DIR / "models" / "checkpoints" / "monai"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = config.early_stopping_patience

        # ── Training loop ────────────────────────────────────────
        for epoch in range(1, config.epochs + 1):
            # --- Train ---
            model.train()
            train_losses = []
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
            for batch in train_bar:
                images = batch["image"].to(device)   # (B, 1, D, H, W)
                labels = batch["label"].to(device)   # (B, 1, D, H, W)

                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = float(np.mean(train_losses))

            # --- Validation ---
            model.eval()
            val_losses = []
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [Val]")
            with torch.no_grad():
                for batch in val_bar:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    val_losses.append(loss.item())
                    val_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_val_loss = float(np.mean(val_losses))
            scheduler.step()

            # ── Log to MLflow ────────────────────────────────────
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            }, step=epoch)

            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
                  f"lr={current_lr:.2e}")

            # ── Checkpoint & Early Stopping ──────────────────────
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_path = ckpt_dir / f"{config.model}_best.pth"
                torch.save(model.state_dict(), str(best_path))
                try:
                    mlflow.log_artifact(str(best_path), artifact_path="models")
                    print(f"   ☁️  Best model saved & uploaded (val_loss={best_val_loss:.4f})")
                except Exception as e:
                    print(f"   ⚠️  Upload failed: {e}")
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"⏹️  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        # ── Final artifacts ──────────────────────────────────────
        log_src_snapshot()

    print(f"=== [MONAI] Training complete | best_val_loss={best_val_loss:.4f} ===")
