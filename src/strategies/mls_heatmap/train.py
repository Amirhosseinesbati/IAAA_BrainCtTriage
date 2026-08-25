"""
train.py — Pure PyTorch training loop for MLS heatmap regression.

Follows the same pattern as MONAI's train_monai() — no PyTorch Lightning.
Logs comprehensive metrics to MLflow: heatmap MSE, keypoint MAE (px),
MLS MAE (mm), and triage-relevant binning accuracy.

The validation metric that drives checkpointing and early stopping is
``val_mls_mae_mm`` — the mean absolute error of the final MLS measurement
in millimeters, not the raw heatmap loss.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import (
    MLFLOW_EXP_MLS_HEATMAP,
    MLS_DIR,
    MLS_CHECKPOINTS_DIR,
    log_src_snapshot,
)
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.utils import (
    decode_heatmap_dark_batch,
    compute_mls_batch,
    compute_mls_metrics,
)
from src.strategies.config_models import MLSHeatmapConfig

logger = logging.getLogger(__name__)


def _compute_validation_metrics(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    heatmap_size: int,
    img_size: int,
    spacing_x: float,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
) -> dict:
    """
    Compute comprehensive validation metrics.

    Predicted keypoints are decoded from heatmaps via DARK and compared
    against the **true keypoint coordinates** returned by the dataset —
    not a DARK re-decoding of the ground-truth heatmap. This avoids
    compounding the heatmap decode error into the reference MLS value,
    so ``mls_mae_mm`` reflects real model error around the 3mm/5mm
    triage thresholds.

    Returns dict with keys:
        val_loss, kp_mae_px, mls_mae_mm, mls_rmse_mm, mls_bin_acc,
        mls_mae_critical, mls_mae_low, n_samples
    """
    model.eval()
    val_losses = []
    all_mls_true = []
    all_mls_pred = []
    kp_errors_px = []

    with torch.no_grad():
        for images, heatmap_targets, masks, keypoints_true in val_loader:
            images = images.to(device)
            heatmap_targets = heatmap_targets.to(device)
            masks = masks.to(device)
            keypoints_true = keypoints_true.numpy()  # (B, K, 2) image pixels

            # Forward
            heatmap_pred = model(images)

            # Masked MSE loss
            loss = 0.0
            for k in range(masks.shape[1]):
                loss += criterion(
                    heatmap_pred[:, k:k+1] * masks[:, k:k+1, None, None],
                    heatmap_targets[:, k:k+1] * masks[:, k:k+1, None, None],
                )
            val_losses.append(loss.item())

            # Decode predicted keypoints via DARK
            coords_pred, _ = decode_heatmap_dark_batch(
                heatmap_pred.cpu(), heatmap_size, img_size
            )  # (B, K, 2) image pixels

            # Per-sample: keypoint MAE (px) and MLS comparison
            for b in range(len(coords_pred)):
                mask_b = masks[b].cpu().numpy()   # (K,)
                kp_true_b = keypoints_true[b]     # (K, 2)
                kp_pred_b = coords_pred[b]        # (K, 2)

                # Keypoint MAE over present + detected keypoints
                for k in range(masks.shape[1]):
                    if mask_b[k] > 0.5 and kp_pred_b[k, 0] >= 0:
                        err = float(np.hypot(
                            kp_pred_b[k, 0] - kp_true_b[k, 0],
                            kp_pred_b[k, 1] - kp_true_b[k, 1],
                        ))
                        kp_errors_px.append(err)

                # MLS comparison — only when all 3 keypoints present & detected
                if (mask_b > 0.5).all() and (kp_pred_b[:, 0] >= 0).all():
                    mls_pred = compute_mls_from_keypoints_np(kp_pred_b, spacing_x)
                    mls_true = compute_mls_from_keypoints_np(kp_true_b, spacing_x)
                    all_mls_pred.append(mls_pred)
                    all_mls_true.append(mls_true)

    avg_val_loss = float(np.mean(val_losses))
    avg_kp_mae_px = float(np.mean(kp_errors_px)) if kp_errors_px else 0.0

    metrics = {
        "val_loss": avg_val_loss,
        "kp_mae_px": avg_kp_mae_px,
        "mls_mae_mm": 0.0,
        "mls_rmse_mm": 0.0,
        "mls_bin_acc": 0.0,
        "mls_mae_critical": 0.0,
        "mls_mae_low": 0.0,
        "n_samples": len(all_mls_true),
    }

    if len(all_mls_true) == 0:
        return metrics

    mls_metrics = compute_mls_metrics(
        np.array(all_mls_true), np.array(all_mls_pred)
    )
    metrics.update(mls_metrics)
    return metrics


def compute_mls_from_keypoints_np(
    keypoints_pixels: np.ndarray,
    spacing_x: float,
) -> float:
    """Compute MLS from 3 keypoints. Numpy version (no torch dependency)."""
    x1, y1 = keypoints_pixels[0]
    x2, y2 = keypoints_pixels[1]
    x3, y3 = keypoints_pixels[2]
    dx = x2 - x1
    dy = y2 - y1
    denom = np.sqrt(dx ** 2 + dy ** 2)
    if denom < 1e-8:
        return 0.0
    numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    return (numerator / denom) * spacing_x


def train_mls_heatmap(
    config: MLSHeatmapConfig,
    csv_path: Optional[str] = None,
    img_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """
    Train the MLS heatmap regression model.

    Pure PyTorch training loop with MLflow logging, model checkpointing,
    early stopping, and ReduceLROnPlateau scheduling.

    Args:
        config: Validated MLSHeatmapConfig instance.
        csv_path: Path to MLS labels CSV. Defaults to config's MLS_DIR.
        img_dir: Path to MLS images directory.
        checkpoint_dir: Directory to save model checkpoints.
    """
    # ── Default paths ─────────────────────────────────────────────
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    csv_path = csv_path or str(BASE_DIR / "Data" / "processed" / "mls_dataset" / "mls_labels.csv")
    img_dir = img_dir or str(BASE_DIR / "Data" / "processed" / "mls_dataset" / "images")
    checkpoint_dir = checkpoint_dir or str(BASE_DIR / "models" / "checkpoints" / "mls_heatmap")

    heatmap_size = config.image_size // 4  # 1/4 resolution

    # ── MLflow setup ──────────────────────────────────────────────
    experiment_name = MLFLOW_EXP_MLS_HEATMAP
    mlflow.set_experiment(experiment_name)
    run_name = f"{config.backbone}_bs{config.batch_size}_lr{config.learning_rate:.0e}"

    with mlflow.start_run(run_name=run_name) as _:
        # Log configuration
        mlflow.log_params(config.model_dump())
        mlflow.set_tag("strategy", "mls_heatmap")
        mlflow.set_tag("backbone", config.backbone)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # ── Data ──────────────────────────────────────────────────
        train_loader, val_loader = create_mls_dataloaders(
            csv_path=csv_path,
            img_dir=img_dir,
            img_size=config.image_size,
            heatmap_size=heatmap_size,
            heatmap_sigma=config.heatmap_sigma,
            batch_size=config.batch_size,
            val_split=config.val_split,
            augment=True,
            rotation_deg=config.rotation_deg,
            translation=config.translation,
            intensity_jitter_scale=config.intensity_jitter,
            augment_prob=config.augment_prob,
            num_workers=config.num_workers,
            seed=config.seed,
        )

        # Use a fixed spacing_x for validation (average brain CT value)
        # Real spacing is read from DICOM during inference.
        VAL_SPACING_X = 0.5  # mm/px (typical for brain CT)

        # ── Model ─────────────────────────────────────────────────
        model = HRNetHeatmapModel(
            backbone_name=config.backbone,
            in_channels=config.input_channels,
            num_keypoints=3,
            pretrained=True,
            head_dropout=config.head_dropout,
        ).to(device)

        # ── Loss & Optimizer ──────────────────────────────────────
        criterion = nn.MSELoss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Warmup + cosine annealing schedule.
        # ReduceLROnPlateau halved the LR too aggressively on the plateaued
        # val_loss (small dataset overfits fast), so we use a smooth schedule
        # independent of noisy val metrics.
        warmup_epochs = min(5, max(1, config.epochs // 10))

        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # linear warmup → 1.0
            progress = (epoch - warmup_epochs) / max(1, config.epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # 1.0 → ~0

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

        # ── Checkpoint dir ────────────────────────────────────────
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── Training loop ─────────────────────────────────────────
        best_val_mae = float("inf")
        epochs_no_improve = 0
        patience = config.early_stopping_patience

        # Use AMP if requested and available
        use_amp = config.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        for epoch in range(1, config.epochs + 1):
            # ── Train ─────────────────────────────────────────────
            model.train()
            train_losses = []
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")

            for images, heatmap_targets, masks, _ in train_bar:
                images = images.to(device, non_blocking=True)
                heatmap_targets = heatmap_targets.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        heatmap_pred = model(images)
                        # Masked MSE loss
                        loss = 0.0
                        for k in range(masks.shape[1]):
                            loss += criterion(
                                heatmap_pred[:, k:k+1] * masks[:, k:k+1, None, None],
                                heatmap_targets[:, k:k+1] * masks[:, k:k+1, None, None],
                            )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    heatmap_pred = model(images)
                    loss = 0.0
                    for k in range(masks.shape[1]):
                        loss += criterion(
                            heatmap_pred[:, k:k+1] * masks[:, k:k+1, None, None],
                            heatmap_targets[:, k:k+1] * masks[:, k:k+1, None, None],
                        )
                    loss.backward()
                    optimizer.step()

                train_losses.append(loss.item())
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = float(np.mean(train_losses))

            # ── Validation ────────────────────────────────────────
            val_metrics = _compute_validation_metrics(
                model, val_loader, heatmap_size, config.image_size,
                VAL_SPACING_X, device, criterion, epoch,
            )

            # Step scheduler (warmup + cosine — independent of noisy val metrics)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Log to MLflow ─────────────────────────────────────
            log_dict = {
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_kp_mae_px": val_metrics.get("kp_mae_px", 0.0),
                "val_mls_mae_mm": val_metrics.get("mls_mae_mm", 0.0),
                "val_mls_rmse_mm": val_metrics.get("mls_rmse_mm", 0.0),
                "val_mls_bin_acc": val_metrics.get("mls_bin_acc", 0.0),
                "val_mls_mae_critical": val_metrics.get("mls_mae_critical", 0.0),
                "lr": current_lr,
                "epoch": epoch,
            }

            # Per-bin accuracy (triage-relevant: <1 / 1-3 / 3-5 / >=5 mm)
            per_bin = val_metrics.get("bin_acc_per_bin", {})
            for bin_idx, acc in per_bin.items():
                if acc is not None:
                    log_dict[f"val_bin_acc_{bin_idx}"] = acc

            # Remove None/NaN before logging
            log_dict = {k: v for k, v in log_dict.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
            mlflow.log_metrics(log_dict, step=epoch)

            # ── Console output ────────────────────────────────────
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"kp_MAE={val_metrics.get('kp_mae_px', 0.0):.2f}px | "
                f"MLS_MAE={val_metrics.get('mls_mae_mm', 0.0):.3f}mm | "
                f"bin_acc={val_metrics.get('mls_bin_acc', 0.0):.2%} | "
                f"lr={current_lr:.2e}"
            )

            # ── Checkpoint & Early Stopping ──────────────────────
            current_val_mae = val_metrics.get("mls_mae_mm", float("inf"))

            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                epochs_no_improve = 0
                best_path = ckpt_dir / "mls_heatmap_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": val_metrics,
                        "best_val_mls_mae_mm": best_val_mae,
                    },
                    str(best_path),
                )
                # Save as MLflow artifact
                try:
                    mlflow.log_artifact(str(best_path), artifact_path="models")
                    print(f"   ☁️  Best model saved & uploaded "
                          f"(val_MLS_MAE={best_val_mae:.3f}mm)")
                except Exception as e:
                    print(f"   ⚠️  MLflow upload failed: {e}")
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(
                        f"⏹️  Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs, "
                        f"best val_MLS_MAE={best_val_mae:.3f}mm)"
                    )
                    break

        # ── Final artifacts ───────────────────────────────────────
        # Save final model
        final_path = ckpt_dir / "mls_heatmap_final.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config.model_dump(),
                "best_val_mls_mae_mm": best_val_mae,
            },
            str(final_path),
        )

        # Log source code snapshot
        try:
            log_src_snapshot()
        except Exception as e:
            logger.warning(f"Could not log source snapshot: {e}")

    print(
        f"=== [MLS Heatmap] Training complete | "
        f"best_val_MLS_MAE={best_val_mae:.3f}mm ==="
    )
