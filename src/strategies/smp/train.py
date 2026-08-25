"""
smp/train.py — PyTorch Lightning training entry point for SMP ICH strategy.

Creates the SMP model, LightningModule, DataModule, and runs training
with MLflow logging. Designed to be called from SMPStrategy.train().
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from src.config import ICH_LABELS, ICH_TYPES, config_section
from src.mlops import context_from_environment, experiment_run, log_run_summary
from src.strategies.config_models import SMPConfig
from src.strategies.losses import build_composite_loss
from src.strategies.loss_config import LossConfig
from src.strategies.smp.dataset import ICHEmbeddingDataset


# ═════════════════════════════════════════════════════════════════════════
# Lightning Data Module
# ═════════════════════════════════════════════════════════════════════════

class ICHEmbeddingDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping ICHEmbeddingDataset for train/val."""

    def __init__(self, config: SMPConfig):
        super().__init__()
        self.config = config
        self._train_dataset: Optional[ICHEmbeddingDataset] = None
        self._val_dataset: Optional[ICHEmbeddingDataset] = None

    def setup(self, stage: Optional[str] = None):
        dataset_kwargs = dict(
            image_size=self.config.image_size,
            model_dimension=self.config.model_dimension,
            slices_per_stack=self.config.slices_per_stack,
            fold=self.config.fold,
            use_competition_folds=self.config.use_competition_folds,
        )
        if self._train_dataset is None:
            self._train_dataset = ICHEmbeddingDataset(
                augmentation=self.config.augmentation,
                augmentation_config=self.config.augmentation_config,
                split="train",
                **dataset_kwargs,
            )
        if self._val_dataset is None:
            self._val_dataset = ICHEmbeddingDataset(
                augmentation=False,
                split="val",
                **dataset_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


# ═════════════════════════════════════════════════════════════════════════
# Lightning Module
# ═════════════════════════════════════════════════════════════════════════

class ICHEmbeddingModule(pl.LightningModule):
    """LightningModule wrapping an SMP segmentation model."""

    def __init__(self, config: SMPConfig):
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config

        num_classes = ICHEmbeddingDataset.NUM_CLASSES
        encoder_weights = config.encoder_weights if config.encoder_weights else None

        # Determine in_channels based on model dimension
        in_channels = 1
        if config.model_dimension == "2.5D" and config.slices_per_stack is not None:
            in_channels = config.slices_per_stack

        self.model = smp.create_model(
            arch=config.architecture,
            encoder_name=config.encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        # Loss: weighted composite from config
        self.loss_composite = build_composite_loss(config.loss_config, num_classes)

        # Metrics
        self.train_iou = smp.utils.metrics.IoU(threshold=0.5)
        self.val_iou = smp.utils.metrics.IoU(threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, preds, target):
        return self.loss_composite(preds, target)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        logits = self(images)
        loss = self._compute_loss(logits, masks)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        preds = logits.argmax(dim=1)

        self.val_iou(preds, masks)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        # Per-class Dice
        # ICH_TYPES = ["IVH", "IPH", "SDH", "EDH", "SAH"] with labels 1-5
        for class_name in ICH_TYPES:
            class_id = ICH_LABELS[class_name]  # From config: 1..5
            pred_mask = (preds == class_id).float()
            true_mask = (masks == class_id).float()
            intersection = (pred_mask * true_mask).sum()
            denominator = pred_mask.sum() + true_mask.sum()
            dice = (2.0 * intersection) / denominator.clamp(min=1e-8)
            self.log(
                f"val_dice_{class_name}",
                dice,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        return loss

    def configure_optimizers(self):
        if self.config.optimizer == "AdamW":
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.config.learning_rate, weight_decay=1e-4,
            )
        elif self.config.optimizer == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        else:  # SGD
            opt = torch.optim.SGD(
                self.parameters(), lr=self.config.learning_rate,
                momentum=0.9, weight_decay=1e-4,
            )

        if self.config.scheduler == "CosineAnnealing":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.config.epochs,
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=10,
            )
        elif self.config.scheduler == "OneCycleLR":
            steps_per_epoch = max(
                len(self.trainer.datamodule.train_dataloader()), 1,
            )
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.config.learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=self.config.epochs,
            )
        else:
            return opt

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss" if self.config.scheduler == "ReduceLROnPlateau" else None,
                "interval": "epoch",
            },
        }


# ═════════════════════════════════════════════════════════════════════════
# Training Entry Point
# ═════════════════════════════════════════════════════════════════════════

def train_smp(config: SMPConfig) -> None:
    """
    Train an SMP model for ICH segmentation with MLflow tracking.

    All hyper-parameters, metrics, and the best checkpoint are logged
    to the configured MLflow tracking server.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    run_name = f"{config.architecture}_{config.encoder}_bs{config.batch_size}"
    context = context_from_environment("ich_smp", run_name, config.model_dump(), strategy="smp")

    with experiment_run(context) as active_run:
        mlf_logger = MLFlowLogger(
            experiment_name=context.experiment_name,
            run_id=active_run.info.run_id,
            log_model=True,
        )
        mlflow.set_tag("architecture", config.architecture)
        mlflow.set_tag("encoder", config.encoder)
        mlflow.set_tag("model_dimension", config.model_dimension)
        mlflow.log_param("in_channels", 1 if config.model_dimension == "2D" else config.slices_per_stack)
        mlflow.log_param("loss_combination", config.loss_config.combination_string)
        # Log augmentation details
        aug = config.augmentation_config
        mlflow.log_param("aug_enabled", aug.enabled)
        for aug_name in ["top_bottom_flip", "left_right_flip", "rotate90"]:
            t = getattr(aug, aug_name)
            mlflow.log_param(f"aug_{aug_name}", f"enabled={t.enabled}, prob={t.prob}")
        for aug_name in ["gauss_noise"]:
            t = getattr(aug, aug_name)
            mlflow.log_param(f"aug_{aug_name}", f"enabled={t.enabled}, prob={t.prob}, var_limit={t.var_limit}")

        # Data
        datamodule = ICHEmbeddingDataModule(config)

        # Model
        model = ICHEmbeddingModule(config)

        # Callbacks
        ckpt_dir = BASE_DIR / "models" / "checkpoints" / "smp"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{config.architecture}_{config.encoder}_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            mode="min",
            verbose=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Trainer
        precision = "16-mixed" if config.use_amp else "32-true"
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            accelerator="auto",
            devices=1,
            precision=precision,
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
            logger=mlf_logger,
            log_every_n_steps=20,
        )

        # Train
        trainer.fit(model, datamodule=datamodule)

        # ── Upload best model to MLflow ───────────────────────────
        if checkpoint_cb.best_model_path and os.path.exists(checkpoint_cb.best_model_path):
            try:
                mlflow.log_artifact(checkpoint_cb.best_model_path, artifact_path=config_section("mlflow", "artifact_paths", "models"))
                print(f"☁️  Best SMP model uploaded: {checkpoint_cb.best_model_path}")
            except Exception as e:
                print(f"⚠️  Model upload failed: {e}")

        log_run_summary({
            "task": "ich", "strategy": "smp",
            "best_val_loss": float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else None,
            "best_checkpoint": checkpoint_cb.best_model_path,
        })

    best_score = checkpoint_cb.best_model_score
    print(f"=== [SMP] Training complete | best_val_loss="
          f"{best_score:.4f}" if best_score is not None else "=== [SMP] Training complete (no checkpoint saved) ===")
