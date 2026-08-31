"""Staged, MLflow-tracked training for the corrected ICH-v2 pipeline."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.strategies.ich_v2.data import create_loaders
from src.strategies.ich_v2.evaluation import (
    VOLUME_KEYS,
    add_oracle_context_triage,
    ground_truth_ich_context,
    summarize_ich_predictions,
)
from src.strategies.ich_v2.geometry import remove_small_components, volumes_from_labelmap
from src.strategies.ich_v2.losses import MaskedDiceFocalLoss
from src.strategies.ich_v2.model import build_seg_resnet, load_model_weights
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


@dataclass(frozen=True)
class ICHV2TrainConfig:
    run_name: str
    output_dir: str
    dataset_dir: str = "Data/processed/ich_v2/BrainICHPartial"
    fold: int = 0
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 1e-5
    roi_size: int = 128
    samples_per_volume: int = 2
    workers: int = 4
    seed: int = 42
    init_checkpoint: str | None = None
    min_component_ml: float = 0.1
    overlap: float = 0.25
    eval_every: int = 1
    patience: int = 6
    max_train_studies: int | None = None
    max_val_studies: int | None = None
    dice_weight: float = 0.6
    focal_weight: float = 0.4
    focal_gamma: float = 2.0
    background_weight: float = 0.2


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def _batch_affine(batch_image: Any) -> np.ndarray:
    affine = getattr(batch_image, "affine", None)
    if affine is None:
        raise ValueError("Validation MetaTensor has no affine")
    array = affine.detach().cpu().numpy() if torch.is_tensor(affine) else np.asarray(affine)
    return np.asarray(array[0] if array.ndim == 3 else array, dtype=np.float64)


def _plain_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if hasattr(value, "as_tensor"):
        value = value.as_tensor()
    return value.to(device)


def _validate(
    model: torch.nn.Module,
    loader,
    loss_fn: MaskedDiceFocalLoss,
    truth_by_study: pd.DataFrame,
    *,
    device: torch.device,
    roi_size: int,
    overlap: float,
    min_component_ml: float,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    model.eval()
    truth = truth_by_study.set_index("study_id")
    losses: list[float] = []
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            source_image = batch["image"]
            affine = _batch_affine(source_image)
            image = _plain_tensor(source_image, device)
            target = _plain_tensor(batch["label"], device)
            supervision = _plain_tensor(batch["supervision"], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    image,
                    roi_size=(roi_size,) * 3,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
                loss = loss_fn(logits, target, supervision)
            losses.append(float(loss.detach().cpu()))
            labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            labels = remove_small_components(
                labels, affine, minimum_ml=min_component_ml
            )
            volumes = volumes_from_labelmap(labels, affine)
            study_id = str(batch["study_id"][0])
            patient_id = str(batch["patient_id"][0])
            gt = truth.loc[study_id]
            row: dict[str, object] = {
                "study_id": study_id,
                "patient_id": patient_id,
                "fold": int(batch["fold"][0]),
                "gt_triage_class": int(gt["gt_triage_class"]),
                "gt_fracture_prob": float(gt["gt_fracture_prob"]),
                "gt_MLS_mm": float(gt["gt_MLS_mm"]),
            }
            row.update({f"gt_{key}": float(gt[f"gt_{key}"]) for key in VOLUME_KEYS})
            row.update({f"pred_{key}": float(volumes[key]) for key in VOLUME_KEYS})
            rows.append(row)
    predictions = add_oracle_context_triage(pd.DataFrame(rows))
    summary = summarize_ich_predictions(predictions)
    return float(np.mean(losses)), predictions, summary


def _save_best(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: ICHV2TrainConfig,
    summary: dict[str, Any],
    manifest_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "schema_version": 2,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "selection_metric": "oracle_context_macro_f1",
        "validation_summary": summary,
        "git_commit": git_commit(),
        "dataset_manifest_sha256": manifest_sha256,
    }, temporary)
    os.replace(temporary, path)


def run_training(config: ICHV2TrainConfig) -> dict[str, Any]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pth"
    if best_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {best_path}")
    resolved_path = output / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    _seed_everything(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("ICH-v2 training requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    manifest_path = Path(config.dataset_dir) / "manifest.csv"
    manifest_sha = file_sha256(manifest_path)
    train_loader, val_loader, train_frame, val_frame = create_loaders(
        config.dataset_dir,
        fold=config.fold,
        roi_size=(config.roi_size,) * 3,
        samples_per_volume=config.samples_per_volume,
        workers=config.workers,
        max_train_studies=config.max_train_studies,
        max_val_studies=config.max_val_studies,
        seed=config.seed,
    )
    truth, metadata_source = ground_truth_ich_context()

    model = build_seg_resnet().to(device)
    if config.init_checkpoint:
        load_model_weights(model, config.init_checkpoint)
    loss_fn = MaskedDiceFocalLoss(
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight,
        focal_gamma=config.focal_gamma,
        background_weight=config.background_weight,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    run_kind = "smoke" if config.max_train_studies or config.max_val_studies else "full_fold"
    notify_campaign(
        "start",
        "آموزش ICH-v2 آغاز شد. تحلیل کوتاه: این اجرا اثر supervision صحیح، منفی‌های پاک و focal loss ماسک‌شده را با معماری ثابت SegResNet می‌سنجد؛ بنابراین بهبود قابل انتساب به داده/loss است نه افزایش اندازه مدل. اقدام بعدی: گیت فنی و سپس مقایسه Macro-F1 و FPR با baseline ۰٫۷۱۷۷.",
        run=config.run_name,
        kind=run_kind,
        fold=config.fold,
        train_studies=len(train_frame),
        val_studies=len(val_frame),
    )

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-v2")
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    try:
        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.set_tags({
                "task": "ich",
                "stage": "masked_finetune" if config.init_checkpoint else "masked_scratch",
                "run_kind": run_kind,
                "git_commit": git_commit(),
                "validation_context": "ground_truth_mls_fracture",
            })
            mlflow.log_params({
                **{key: value for key, value in asdict(config).items() if value is not None},
                "train_studies": len(train_frame),
                "val_studies": len(val_frame),
                "dataset_manifest_sha256": manifest_sha,
                "metadata_source": str(metadata_source),
            })

            for epoch in range(1, config.epochs + 1):
                model.train()
                epoch_loss: list[float] = []
                epoch_dice: list[float] = []
                epoch_focal: list[float] = []
                for step, batch in enumerate(train_loader, start=1):
                    image = _plain_tensor(batch["image"], device)
                    target = _plain_tensor(batch["label"], device)
                    supervision = _plain_tensor(batch["supervision"], device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(image)
                        components = loss_fn.components(logits, target, supervision)
                        loss = components["loss"]
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss.append(float(loss.detach().cpu()))
                    epoch_dice.append(float(components["dice"].detach().cpu()))
                    epoch_focal.append(float(components["focal"].detach().cpu()))
                    if step % 20 == 0 or step == len(train_loader):
                        print(
                            f"epoch={epoch}/{config.epochs} step={step}/{len(train_loader)} "
                            f"loss={np.mean(epoch_loss[-20:]):.5f}"
                        )
                scheduler.step()
                metrics: dict[str, float] = {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(epoch_loss)),
                    "train_dice_loss": float(np.mean(epoch_dice)),
                    "train_focal_loss": float(np.mean(epoch_focal)),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }

                if epoch % config.eval_every == 0 or epoch == config.epochs:
                    val_loss, predictions, summary = _validate(
                        model,
                        val_loader,
                        loss_fn,
                        truth,
                        device=device,
                        roi_size=config.roi_size,
                        overlap=config.overlap,
                        min_component_ml=config.min_component_ml,
                    )
                    score = float(summary["oracle_context_macro_f1"])
                    metrics.update({
                        "val_loss": val_loss,
                        "oracle_context_macro_f1": score,
                        "total_mae_ml": float(summary["total"]["mae_ml"]),
                        "total_bias_ml": float(summary["total"]["bias_ml"]),
                        "total_presence_f1": float(summary["total"]["presence_f1_at_0_1ml"]),
                        "normal_false_positive_rate": float(summary["total"]["normal_false_positive_rate"]),
                    })
                    if score > best_score + 1e-6:
                        best_score = score
                        best_epoch = epoch
                        stale_epochs = 0
                        _save_best(
                            best_path,
                            model,
                            optimizer,
                            epoch=epoch,
                            config=config,
                            summary=summary,
                            manifest_sha256=manifest_sha,
                        )
                        predictions.to_csv(output / "best_study_predictions.csv", index=False)
                        (output / "best_summary.json").write_text(
                            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
                        )
                        notify_campaign(
                            "checkpoint",
                            "checkpoint بهتر ثبت شد. تحلیل کوتاه: انتخاب فقط بر اساس Macro-F1 study-level انجام شده، نه Dice یا train loss؛ FPR نیز برای جلوگیری از بهبود ظاهری گزارش می‌شود. اقدام بعدی: ادامه تا patience و سنجش پایداری بهبود.",
                            run=config.run_name,
                            epoch=epoch,
                            macro_f1=f"{score:.4f}",
                            normal_fpr=f"{summary['total']['normal_false_positive_rate']:.4f}",
                            total_mae_ml=f"{summary['total']['mae_ml']:.3f}",
                        )
                    else:
                        stale_epochs += 1
                history.append(metrics)
                mlflow.log_metrics(
                    {key: value for key, value in metrics.items() if key != "epoch"},
                    step=epoch,
                )
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                print(json.dumps(metrics, sort_keys=True))
                if config.patience > 0 and stale_epochs >= config.patience:
                    print(f"early_stop epoch={epoch} stale_epochs={stale_epochs}")
                    break

            duration = time.perf_counter() - started
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            final = {
                "run_name": config.run_name,
                "run_id": run.info.run_id,
                "run_kind": run_kind,
                "best_epoch": best_epoch,
                "best_oracle_context_macro_f1": best_score,
                "epochs_completed": int(history[-1]["epoch"]),
                "duration_s": duration,
                "peak_vram_gb": peak_vram_gb,
                "checkpoint": str(best_path),
                "checkpoint_sha256": file_sha256(best_path),
                "dataset_manifest_sha256": manifest_sha,
                "git_commit": git_commit(),
            }
            (output / "run_summary.json").write_text(
                json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
            )
            mlflow.log_metrics({
                "best_oracle_context_macro_f1": best_score,
                "duration_s": duration,
                "peak_vram_gb": peak_vram_gb,
            })
            mlflow.log_artifacts(str(output), artifact_path="ich_v2_run")

        notify_campaign(
            "success",
            "آموزش ICH-v2 با موفقیت تمام شد. تحلیل کوتاه: عدد بهترین checkpoint با baseline معتبر ۰٫۷۱۷۷ مقایسه می‌شود؛ smoke فقط گیت فنی است و برای ادعای بهبود کافی نیست. اقدام بعدی: اگر گیت فنی پاس شده باشد اجرای fold کامل، و اگر fold کامل بهتر باشد تکرار روی fold 1.",
            run=config.run_name,
            kind=run_kind,
            best_epoch=best_epoch,
            best_macro_f1=f"{best_score:.4f}",
            peak_vram_gb=f"{peak_vram_gb:.2f}",
            duration_min=f"{duration / 60:.1f}",
        )
        return final
    except Exception as exc:
        notify_campaign(
            "failure",
            "آموزش ICH-v2 متوقف شد. تحلیل کوتاه: این failure فنی است و هیچ نتیجه کیفیتی از آن استنباط نمی‌شود؛ checkpoint ناقص promote نخواهد شد. اقدام بعدی: بررسی traceback و فقط یک تکرار کنترل‌شده پس از اصلاح علت.",
            run=config.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
