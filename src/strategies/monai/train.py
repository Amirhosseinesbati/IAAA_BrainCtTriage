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

from src.config import ICH_LABELS, ICH_TYPES, config_section
from src.mlops import context_from_environment, experiment_run, log_run_summary
from src.strategies.config_models import MONAIConfig
from src.strategies.losses import build_composite_loss
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
            in_channels=1, out_channels=num_classes,
            patch_size=(2, 2, 2),
            window_size=(7, 14, 14),
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

    run_name = f"{config.model}_bs{config.batch_size}_roi{config.roi_size}"

    context = context_from_environment(
        "ich_monai", run_name, config.model_dump(), strategy="monai"
    )
    with experiment_run(context):
        mlflow.set_tag("model", config.model)
        mlflow.set_tag("model_dimension", config.model_dimension)
        mlflow.log_param("loss_combination", config.loss_config.combination_string)
        # Log augmentation details
        aug = config.augmentation_config
        mlflow.log_param("aug_enabled", aug.enabled)
        for axis_name in ["flip_axis_0", "flip_axis_1", "flip_axis_2"]:
            t = getattr(aug, axis_name)
            mlflow.log_param(f"aug_{axis_name}", f"enabled={t.enabled}, prob={t.prob}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Data
        train_loader = create_monai_dataloaders(config, split="train")
        val_loader = create_monai_dataloaders(config, split="val")

        # Model
        model = _build_model(config.model, NUM_CLASSES).to(device)

        # Loss (weighted composite) & Optimizer
        loss_fn = build_composite_loss(config.loss_config, NUM_CLASSES)

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
            # Per-class Dice accumulators
            dice_sums: dict[str, float] = {c: 0.0 for c in ICH_TYPES}
            dice_counts: dict[str, int] = {c: 0 for c in ICH_TYPES}

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [Val]")
            with torch.no_grad():
                for batch in val_bar:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)  # (B, 1, D, H, W) int
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    val_losses.append(loss.item())

                    # Per-class Dice
                    preds = logits.argmax(dim=1)           # (B, D, H, W)
                    targets = labels.squeeze(1)            # (B, D, H, W)
                    for class_name in ICH_TYPES:
                        class_id = ICH_LABELS[class_name]
                        pred_mask = (preds == class_id).float()
                        true_mask = (targets == class_id).float()
                        intersection = (pred_mask * true_mask).sum().item()
                        denominator = pred_mask.sum().item() + true_mask.sum().item()
                        if denominator > 1e-8:
                            dice_sums[class_name] += 2.0 * intersection / denominator
                            dice_counts[class_name] += 1

                    val_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_val_loss = float(np.mean(val_losses))
            scheduler.step()

            # ── Log to MLflow ────────────────────────────────────
            current_lr = scheduler.get_last_lr()[0]
            metrics_dict: dict[str, float] = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            }
            # Log per-class Dice (averaged across batches)
            for class_name in ICH_TYPES:
                if dice_counts[class_name] > 0:
                    metrics_dict[f"val_dice_{class_name}"] = (
                        dice_sums[class_name] / dice_counts[class_name]
                    )
            mlflow.log_metrics(metrics_dict, step=epoch)

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
                    mlflow.log_artifact(str(best_path), artifact_path=config_section("mlflow", "artifact_paths", "models"))
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
        log_run_summary({
            "task": "ich", "strategy": "monai", "best_val_loss": best_val_loss,
            "epochs_completed": epoch, "checkpoint": str(best_path) if 'best_path' in locals() else None,
        })

    print(f"=== [MONAI] Training complete | best_val_loss={best_val_loss:.4f} ===")
