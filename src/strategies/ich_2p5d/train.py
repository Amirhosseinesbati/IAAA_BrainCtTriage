"""MLflow-tracked 2.5D ICH gate with an untouched outer validation fold."""

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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)

from .cache import OUTPUT_LABELS
from .data import create_loaders, positive_class_weights
from .evaluation import (
    PresenceRule,
    aggregate_studies,
    evaluate_presence_rule,
    rule_as_dict,
    select_presence_rule,
)
from .model import DEFAULT_MODEL_NAME, build_model, load_model_weights


@dataclass(frozen=True)
class ICH25DTrainConfig:
    run_name: str
    output_dir: str
    manifest_path: str = "Data/processed/ich_2p5d/slice_manifest.csv"
    model_name: str = DEFAULT_MODEL_NAME
    outer_fold: int = 0
    calibration_fold: int = 1
    epochs: int = 6
    batch_size: int = 32
    workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    focal_gamma: float = 1.0
    any_loss_weight: float = 2.0
    maximum_pos_weight: float = 20.0
    minimum_calibration_sensitivity: float = 0.95
    pretrained: bool = True
    seed: int = 42
    patience: int = 3
    max_train_steps: int | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _multilabel_focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    any_loss_weight: float,
) -> torch.Tensor:
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be non-negative")
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), pos_weight=pos_weight, reduction="none"
    )
    probabilities = torch.sigmoid(logits.float())
    correct_probability = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    focal = (1.0 - correct_probability).pow(focal_gamma)
    channel_weights = torch.ones(logits.shape[1], device=logits.device)
    channel_weights[0] = any_loss_weight
    return (bce * focal * channel_weights.unsqueeze(0)).mean()


def _predict_slices(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            targets = batch["target"].numpy()
            slice_indices = batch["slice_index"].numpy()
            for index, study_id in enumerate(batch["study_id"]):
                row: dict[str, object] = {
                    "study_id": str(study_id),
                    "slice_index": int(slice_indices[index]),
                }
                for output_index, label in enumerate(OUTPUT_LABELS):
                    row[f"truth_{label}"] = int(targets[index, output_index])
                    row[f"prob_{label}"] = float(probabilities[index, output_index])
                rows.append(row)
    return pd.DataFrame(rows)


def _slice_auc(predictions: pd.DataFrame, label: str = "any_ich") -> float:
    truth = predictions[f"truth_{label}"].to_numpy(dtype=np.int64)
    if len(np.unique(truth)) < 2:
        return float("nan")
    return float(roc_auc_score(truth, predictions[f"prob_{label}"]))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    config: ICH25DTrainConfig,
    epoch: int,
    rule: PresenceRule,
    calibration_metrics: dict[str, Any],
    manifest_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "schema_version": 1,
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "output_labels": OUTPUT_LABELS,
        "input_channels": 9,
        "presence_rule": rule_as_dict(rule),
        "calibration_metrics": calibration_metrics,
        "selection_metric": "calibration_presence_f1_with_sensitivity_constraint",
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit(),
    }, temporary)
    os.replace(temporary, path)


def run_training(config: ICH25DTrainConfig) -> dict[str, Any]:
    if config.outer_fold == config.calibration_fold:
        raise ValueError("outer_fold and calibration_fold must differ")
    if not torch.cuda.is_available():
        raise RuntimeError("2.5D ICH training requires CUDA")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pth"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {checkpoint_path}")
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    _seed_everything(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    (
        train_loader,
        calibration_loader,
        outer_loader,
        train_frame,
        calibration_frame,
        outer_frame,
    ) = create_loaders(
        config.manifest_path,
        outer_fold=config.outer_fold,
        calibration_fold=config.calibration_fold,
        batch_size=config.batch_size,
        workers=config.workers,
        seed=config.seed,
    )
    manifest_sha = file_sha256(config.manifest_path)
    run_kind = "smoke" if config.max_train_steps else "full_fold"
    notify_campaign(
        "start",
        "آموزش gate دوبعدونیم ICH آغاز شد. تحلیل کوتاه: سه slice مجاور و سه پنجره CT به مدل داده می‌شود؛ threshold فقط روی fold کالیبراسیون انتخاب و fold بیرونی تا پایان دست‌نخورده می‌ماند. اقدام بعدی: انتخاب checkpoint با F1 حضور تحت قید حساسیت و سپس ترکیب gate با حجم سه‌بعدی.",
        run=config.run_name,
        kind=run_kind,
        fold=f"outer={config.outer_fold}, calibration={config.calibration_fold}",
        train_studies=train_frame["study_id"].nunique(),
        val_studies=outer_frame["study_id"].nunique(),
    )

    model = build_model(
        config.model_name, pretrained=config.pretrained, dropout=config.dropout
    ).to(device)
    pos_weight = positive_class_weights(
        train_frame, maximum=config.maximum_pos_weight
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d")
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    try:
        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.set_tags({
                "task": "ich_presence",
                "stage": "2p5d_adjacent_multiwindow",
                "run_kind": run_kind,
                "git_commit": git_commit(),
                "outer_fold_policy": "untouched_until_best_checkpoint",
            })
            mlflow.log_params({
                **asdict(config),
                "train_slices": len(train_frame),
                "calibration_slices": len(calibration_frame),
                "outer_slices": len(outer_frame),
                "train_studies": train_frame["study_id"].nunique(),
                "calibration_studies": calibration_frame["study_id"].nunique(),
                "outer_studies": outer_frame["study_id"].nunique(),
                "manifest_sha256": manifest_sha,
                "positive_class_weights": json.dumps(pos_weight.cpu().tolist()),
            })

            for epoch in range(1, config.epochs + 1):
                model.train()
                losses: list[float] = []
                for step, batch in enumerate(train_loader, start=1):
                    images = batch["image"].to(device, non_blocking=True)
                    targets = batch["target"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images)
                        loss = _multilabel_focal_bce(
                            logits,
                            targets,
                            pos_weight=pos_weight,
                            focal_gamma=config.focal_gamma,
                            any_loss_weight=config.any_loss_weight,
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach().cpu()))
                    if step % 50 == 0:
                        print(
                            f"epoch={epoch}/{config.epochs} step={step}/{len(train_loader)} "
                            f"loss={np.mean(losses[-50:]):.5f}",
                            flush=True,
                        )
                    if config.max_train_steps and step >= config.max_train_steps:
                        break
                scheduler.step()

                calibration_slices = _predict_slices(
                    model, calibration_loader, device=device
                )
                calibration_studies = aggregate_studies(calibration_slices)
                rule = select_presence_rule(
                    calibration_studies,
                    minimum_sensitivity=config.minimum_calibration_sensitivity,
                )
                calibration_metrics = evaluate_presence_rule(calibration_studies, rule)
                calibration_slice_auc = _slice_auc(calibration_slices)
                score = float(calibration_metrics["f1"])
                auc = float(calibration_metrics["roc_auc"])
                metrics: dict[str, object] = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "calibration_slice_auc": calibration_slice_auc,
                    "calibration_study_auc": auc,
                    "calibration_presence_f1": score,
                    "calibration_sensitivity": float(calibration_metrics["sensitivity"]),
                    "calibration_specificity": float(calibration_metrics["specificity"]),
                    "calibration_threshold": float(rule.threshold),
                    "calibration_pooling": rule.pooling,
                }
                history.append(metrics)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics({
                    key: float(value) for key, value in metrics.items()
                    if key not in {"epoch", "calibration_pooling"}
                }, step=epoch)
                print(json.dumps(metrics, sort_keys=True), flush=True)

                improved = score > best_score + 1e-6 or (
                    abs(score - best_score) <= 1e-6 and auc > best_auc + 1e-6
                )
                if improved:
                    best_score, best_auc, best_epoch = score, auc, epoch
                    stale_epochs = 0
                    _save_checkpoint(
                        checkpoint_path,
                        model,
                        config=config,
                        epoch=epoch,
                        rule=rule,
                        calibration_metrics=calibration_metrics,
                        manifest_sha256=manifest_sha,
                    )
                    calibration_slices.to_csv(
                        output / "best_calibration_slice_predictions.csv", index=False
                    )
                    calibration_studies.to_csv(
                        output / "best_calibration_study_predictions.csv", index=False
                    )
                    notify_campaign(
                        "checkpoint",
                        "checkpoint بهتر gate ثبت شد. تحلیل کوتاه: انتخاب فقط با fold کالیبراسیون و تحت قید حساسیت انجام شده؛ fold0 هنوز دیده نشده است. اقدام بعدی: ادامه تا patience و فقط یک ارزیابی outer برای بهترین checkpoint.",
                        run=config.run_name,
                        epoch=epoch,
                        macro_f1=f"presence={score:.4f}",
                        normal_fpr=f"{1.0 - float(calibration_metrics['specificity']):.4f}",
                    )
                else:
                    stale_epochs += 1
                if config.patience > 0 and stale_epochs >= config.patience:
                    break

            payload = load_model_weights(model, checkpoint_path)
            model.to(device).eval()
            rule = PresenceRule(**payload["presence_rule"])
            outer_slices = _predict_slices(model, outer_loader, device=device)
            outer_studies = aggregate_studies(outer_slices)
            outer_metrics = evaluate_presence_rule(outer_studies, rule)
            outer_metrics["slice_auc"] = _slice_auc(outer_slices)
            outer_slices.to_csv(output / "outer_slice_predictions.csv", index=False)
            outer_studies.to_csv(output / "outer_study_predictions.csv", index=False)
            (output / "presence_rule.json").write_text(
                json.dumps(rule_as_dict(rule), indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "outer_metrics.json").write_text(
                json.dumps(outer_metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            duration = time.perf_counter() - started
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            summary = {
                "run_name": config.run_name,
                "run_id": run.info.run_id,
                "run_kind": run_kind,
                "best_epoch": best_epoch,
                "calibration_presence_f1": best_score,
                "calibration_study_auc": best_auc,
                "presence_rule": rule_as_dict(rule),
                "outer_metrics": outer_metrics,
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "manifest_sha256": manifest_sha,
                "git_commit": git_commit(),
            }
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            mlflow.log_metrics({
                "outer_presence_f1": float(outer_metrics["f1"]),
                "outer_sensitivity": float(outer_metrics["sensitivity"]),
                "outer_specificity": float(outer_metrics["specificity"]),
                "outer_study_auc": float(outer_metrics["roc_auc"]),
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
            })
            mlflow.log_artifacts(str(output), artifact_path="ich_2p5d_run")

        notify_campaign(
            "success",
            "آموزش gate دوبعدونیم تمام شد. تحلیل کوتاه: معیار outer فقط یک‌بار و با rule ثابت fold کالیبراسیون محاسبه شده است؛ پذیرش نهایی بعد از ترکیب با خروجی حجم سه‌بعدی انجام می‌شود. اقدام بعدی: اعمال gate روی baseline و exp03 و مقایسه مستقیم Macro-F1 triage.",
            run=config.run_name,
            kind=run_kind,
            best_epoch=best_epoch,
            macro_f1=f"presence={float(outer_metrics['f1']):.4f}",
            normal_fpr=f"{1.0 - float(outer_metrics['specificity']):.4f}",
            peak_vram_gb=f"{peak_vram:.2f}",
            duration_min=f"{duration / 60:.1f}",
        )
        return summary
    except Exception as exc:
        notify_campaign(
            "failure",
            "آموزش gate دوبعدونیم متوقف شد. تحلیل کوتاه: این رخداد تا بررسی traceback نتیجهٔ کیفیتی محسوب نمی‌شود و checkpoint ناقص promote نخواهد شد. اقدام بعدی: اصلاح علت فنی و تکرار فقط گیت کوچک.",
            run=config.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
