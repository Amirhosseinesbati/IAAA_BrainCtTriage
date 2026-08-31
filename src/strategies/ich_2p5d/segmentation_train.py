"""Leakage-safe training of the direct 2.5D ICH segmentation/volume model."""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)

from .cache import OUTPUT_LABELS
from .segmentation_data import (
    create_segmentation_loaders,
    segmentation_classification_weights,
    segmentation_foreground_weights,
    subtype_aware_sampling_weights,
)
from .segmentation_evaluation import summarize_segmentation_predictions
from .segmentation_loss import ICH25DSegmentationLoss
from .segmentation_model import (
    DEFAULT_SEGMENTATION_ARCHITECTURE,
    DEFAULT_SEGMENTATION_ENCODER,
    build_segmentation_model,
    load_segmentation_weights,
)


CHECKPOINT_SELECTION_STRATEGIES = ("legacy", "fpr_penalized")


def checkpoint_selection_score(
    summary: dict[str, Any], strategy: str
) -> float:
    selection = float(summary["selection_score"])
    if strategy == "legacy":
        return selection
    if strategy == "fpr_penalized":
        return selection - 0.10 * float(
            summary["normal_false_positive_rate_at_0_1ml"]
        )
    raise ValueError(
        "checkpoint_selection_strategy must be one of: "
        f"{', '.join(CHECKPOINT_SELECTION_STRATEGIES)}"
    )


@dataclass(frozen=True)
class ICH25DSegmentationTrainConfig:
    run_name: str
    output_dir: str
    manifest_path: str = "Data/processed/ich_2p5d/slice_manifest.csv"
    architecture: str = DEFAULT_SEGMENTATION_ARCHITECTURE
    encoder_name: str = DEFAULT_SEGMENTATION_ENCODER
    outer_fold: int = 0
    calibration_fold: int = 1
    epochs: int = 8
    batch_size: int = 8
    workers: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    classification_loss_weight: float = 0.25
    classification_focal_gamma: float = 1.0
    background_weight: float = 0.15
    empty_foreground_weight: float = 0.0
    empty_foreground_top_fraction: float = 1.0
    checkpoint_selection_strategy: str = "legacy"
    maximum_pos_weight: float = 20.0
    segmentation_class_weight_power: float = 0.0
    maximum_segmentation_class_weight: float = 8.0
    segmentation_class_weight_basis: str = "slice"
    sampler_study_balance_power: float = 0.0
    pretrained: bool = True
    seed: int = 42
    patience: int = 3
    max_train_steps: int | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unpack_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise TypeError("Segmentation model must return mask and auxiliary logits")
    return outputs[0], outputs[1]


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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask_logits, class_logits = _unpack_outputs(model(images))
            predicted_masks = mask_logits.float().argmax(dim=1).cpu()
            class_probabilities = torch.sigmoid(class_logits.float()).cpu().numpy()
            true_masks = batch["mask"]
            known = batch["known"].numpy()
            voxel_volumes = batch["voxel_volume_ml"].numpy()
            slice_indices = batch["slice_index"].numpy()
            for index, study_id in enumerate(batch["study_id"]):
                predicted = predicted_masks[index]
                observed = true_masks[index]
                is_known = bool(known[index] > 0.5)
                row: dict[str, object] = {
                    "study_id": str(study_id),
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_index": int(slice_indices[index]),
                    "known": int(is_known),
                    "voxel_volume_ml": float(voxel_volumes[index]),
                }
                for output_index, label in enumerate(OUTPUT_LABELS):
                    row[f"prob_{label}"] = float(class_probabilities[index, output_index])
                for class_id, label in enumerate(OUTPUT_LABELS[1:], start=1):
                    predicted_class = predicted == class_id
                    row[f"pred_pixels_{label}"] = int(predicted_class.sum())
                    if is_known:
                        observed_class = observed == class_id
                        row[f"intersection_{label}"] = int(
                            (predicted_class & observed_class).sum()
                        )
                        row[f"predicted_known_pixels_{label}"] = int(
                            predicted_class.sum()
                        )
                        row[f"observed_known_pixels_{label}"] = int(
                            observed_class.sum()
                        )
                    else:
                        row[f"intersection_{label}"] = 0
                        row[f"predicted_known_pixels_{label}"] = 0
                        row[f"observed_known_pixels_{label}"] = 0
                rows.append(row)
    return pd.DataFrame(rows)


def _flatten_summary_metrics(prefix: str, summary: dict[str, Any]) -> dict[str, float]:
    metrics = {
        f"{prefix}_selection_score": float(summary["selection_score"]),
        f"{prefix}_mean_foreground_dice": float(summary["mean_foreground_dice"]),
        f"{prefix}_any_ich_study_auc": float(summary["any_ich_study_auc"] or 0.0),
        f"{prefix}_macro_subtype_study_auc": float(summary["macro_subtype_study_auc"]),
        f"{prefix}_presence_f1_at_0_1ml": float(summary["presence_f1_at_0_1ml"]),
        f"{prefix}_normal_fpr_at_0_1ml": float(
            summary["normal_false_positive_rate_at_0_1ml"]
        ),
        f"{prefix}_total_volume_mae_ml": float(summary["total_volume_mae_ml"]),
        f"{prefix}_total_volume_bias_ml": float(summary["total_volume_bias_ml"]),
    }
    for label, subtype in summary["subtypes"].items():
        for name in ("dice_known_pixels", "study_auc", "mae_ml", "bias_ml"):
            value = subtype[name]
            if value is not None and np.isfinite(value):
                metrics[f"{prefix}_{label.lower()}_{name}"] = float(value)
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    config: ICH25DSegmentationTrainConfig,
    epoch: int,
    calibration_summary: dict[str, Any],
    manifest_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "schema_version": 1,
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "output_labels": OUTPUT_LABELS,
        "segmentation_classes": 6,
        "input_channels": 9,
        "selection_metric": (
            "ich_only_0.55_dice_0.30_any_auc_0.15_subtype_auc"
            if config.checkpoint_selection_strategy == "legacy"
            else "ich_only_selection_minus_0.10_normal_fpr"
        ),
        "calibration_summary": calibration_summary,
        "manifest_sha256": manifest_sha256,
        "git_commit": git_commit(),
    }, temporary)
    os.replace(temporary, path)


def run_segmentation_training(
    config: ICH25DSegmentationTrainConfig,
) -> dict[str, Any]:
    if config.outer_fold == config.calibration_fold:
        raise ValueError("outer_fold and calibration_fold must differ")
    if config.checkpoint_selection_strategy not in CHECKPOINT_SELECTION_STRATEGIES:
        raise ValueError(
            "checkpoint_selection_strategy must be one of: "
            f"{', '.join(CHECKPOINT_SELECTION_STRATEGIES)}"
        )
    if not 0.0 <= config.sampler_study_balance_power <= 1.0:
        raise ValueError("sampler_study_balance_power must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("2.5D ICH segmentation training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("2.5D ICH segmentation requires CUDA BF16 support")
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
    ) = create_segmentation_loaders(
        config.manifest_path,
        outer_fold=config.outer_fold,
        calibration_fold=config.calibration_fold,
        batch_size=config.batch_size,
        workers=config.workers,
        seed=config.seed,
        sampler_study_balance_power=config.sampler_study_balance_power,
    )
    truth, metadata_source = ground_truth_ich_context()
    truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
    manifest_sha = file_sha256(config.manifest_path)
    sampling_weights = subtype_aware_sampling_weights(
        train_frame,
        study_balance_power=config.sampler_study_balance_power,
    ).numpy()
    sampling_positive = train_frame[list(OUTPUT_LABELS[1:])].any(axis=1).to_numpy()
    sampling_weight_sum = float(sampling_weights.sum())
    sampler_diagnostics = {
        "weight_min": float(sampling_weights.min()),
        "weight_max": float(sampling_weights.max()),
        "weight_mean": float(sampling_weights.mean()),
        "effective_sample_size": float(
            sampling_weight_sum**2 / np.square(sampling_weights).sum()
        ),
        "positive_probability_mass": float(
            sampling_weights[sampling_positive].sum() / sampling_weight_sum
        ),
    }
    run_kind = "smoke" if config.max_train_steps else "full_fold"
    notify_campaign(
        "start",
        f"آموزش مدل مستقیم segmentation دوبعدونیم ICH آغاز شد. فرضیه: وزن empty-foreground={config.empty_foreground_weight:.3f} روی سخت‌ترین سهم={config.empty_foreground_top_fraction:.4f} از پیکسل‌های ماسک سالم باید false-positive موضعی را کم کند؛ توان study-balance sampler={config.sampler_study_balance_power:.2f} نیز در صورت غیرصفر exposure مطالعات کم‌حجم را افزایش می‌دهد، بدون تغییر جرم کل نمونه‌های مثبت. راهبرد انتخاب checkpoint={config.checkpoint_selection_strategy} است؛ در حالت fpr_penalized امتیاز برابر selection-0.10×FPR و MAE همچنان گیت مستقل است. اقدام بعدی: ارزیابی یک‌بارهٔ outer و مقایسه با baseline دقیقاً هم‌fold.",
        run=config.run_name,
        kind=run_kind,
        architecture=f"{config.architecture}/{config.encoder_name}",
        fold=f"outer={config.outer_fold}, calibration={config.calibration_fold}",
        train_slices=len(train_frame),
        spatially_supervised_slices=int(train_frame["segmentation_known"].sum()),
        empty_foreground_weight=f"{config.empty_foreground_weight:.3f}",
        empty_top_fraction=f"{config.empty_foreground_top_fraction:.4f}",
        checkpoint_selection=config.checkpoint_selection_strategy,
        sampler_study_balance_power=f"{config.sampler_study_balance_power:.2f}",
        sampler_weight_max=f"{sampler_diagnostics['weight_max']:.2f}",
    )

    model = build_segmentation_model(
        architecture=config.architecture,
        encoder_name=config.encoder_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)
    pos_weight = segmentation_classification_weights(
        train_frame, maximum=config.maximum_pos_weight
    ).to(device)
    segmentation_class_weights = segmentation_foreground_weights(
        train_frame,
        power=config.segmentation_class_weight_power,
        maximum=config.maximum_segmentation_class_weight,
        basis=config.segmentation_class_weight_basis,
    ).to(device)
    loss_fn = ICH25DSegmentationLoss(
        classification_pos_weight=pos_weight,
        segmentation_class_weights=segmentation_class_weights,
        classification_weight=config.classification_loss_weight,
        classification_focal_gamma=config.classification_focal_gamma,
        background_weight=config.background_weight,
        empty_foreground_weight=config.empty_foreground_weight,
        empty_foreground_top_fraction=config.empty_foreground_top_fraction,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_dice = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    try:
        with mlflow.start_run(run_name=config.run_name) as run:
            mlflow.set_tags({
                "task": "ich_segmentation_volume",
                "stage": "2p5d_multitask_segmentation",
                "run_kind": run_kind,
                "git_commit": git_commit(),
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
                "outer_fold_policy": "untouched_until_best_checkpoint",
            })
            mlflow.log_params({
                **asdict(config),
                "train_slices": len(train_frame),
                "train_segmentation_known_slices": int(
                    train_frame["segmentation_known"].sum()
                ),
                "train_classification_known_slices": int(
                    train_frame["classification_known"].sum()
                ),
                "calibration_slices": len(calibration_frame),
                "outer_slices": len(outer_frame),
                "train_studies": train_frame["study_id"].nunique(),
                "calibration_studies": calibration_frame["study_id"].nunique(),
                "outer_studies": outer_frame["study_id"].nunique(),
                "manifest_sha256": manifest_sha,
                "metadata_source": str(metadata_source),
                "positive_class_weights": json.dumps(pos_weight.cpu().tolist()),
                "segmentation_class_weights": json.dumps(
                    segmentation_class_weights.cpu().tolist()
                ),
                **{
                    f"sampler_{key}": value
                    for key, value in sampler_diagnostics.items()
                },
            })

            for epoch in range(1, config.epochs + 1):
                model.train()
                component_history: dict[str, list[float]] = {
                    name: [] for name in (
                        "loss", "segmentation", "dice", "focal",
                        "empty_foreground", "classification"
                    )
                }
                for step, batch in enumerate(train_loader, start=1):
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    targets = batch["target"].to(device, non_blocking=True)
                    segmentation_known = batch["segmentation_known"].to(
                        device, non_blocking=True
                    )
                    classification_known = batch["classification_known"].to(
                        device, non_blocking=True
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        mask_logits, class_logits = _unpack_outputs(model(images))
                        components = loss_fn.components(
                            mask_logits,
                            class_logits,
                            masks,
                            targets,
                            segmentation_known,
                            classification_known,
                        )
                    components["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    for name, value in components.items():
                        component_history[name].append(float(value.detach().cpu()))
                    if step % 50 == 0:
                        print(
                            f"epoch={epoch}/{config.epochs} step={step}/{len(train_loader)} "
                            f"loss={np.mean(component_history['loss'][-50:]):.5f}",
                            flush=True,
                        )
                    if config.max_train_steps and step >= config.max_train_steps:
                        break
                scheduler.step()

                calibration_slices = _predict_slices(
                    model, calibration_loader, device=device
                )
                calibration_studies, calibration_summary = summarize_segmentation_predictions(
                    calibration_slices, truth
                )
                score = checkpoint_selection_score(
                    calibration_summary, config.checkpoint_selection_strategy
                )
                dice = float(calibration_summary["mean_foreground_dice"])
                epoch_metrics: dict[str, object] = {
                    "epoch": epoch,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "calibration_checkpoint_score": score,
                    **{
                        f"train_{name}": float(np.mean(values))
                        for name, values in component_history.items()
                    },
                    **_flatten_summary_metrics("calibration", calibration_summary),
                }
                history.append(epoch_metrics)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics(
                    {
                        key: float(value) for key, value in epoch_metrics.items()
                        if key != "epoch"
                    },
                    step=epoch,
                )
                print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

                improved = score > best_score + 1e-6 or (
                    abs(score - best_score) <= 1e-6 and dice > best_dice + 1e-6
                )
                if improved:
                    best_score, best_dice, best_epoch = score, dice, epoch
                    stale_epochs = 0
                    _save_checkpoint(
                        checkpoint_path,
                        model,
                        config=config,
                        epoch=epoch,
                        calibration_summary=calibration_summary,
                        manifest_sha256=manifest_sha,
                    )
                    calibration_slices.to_csv(
                        output / "best_calibration_slice_predictions.csv", index=False
                    )
                    calibration_studies.to_csv(
                        output / "best_calibration_study_predictions.csv", index=False
                    )
                    (output / "best_calibration_summary.json").write_text(
                        json.dumps(calibration_summary, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    notify_campaign(
                        "checkpoint",
                        "checkpoint بهتر segmentation ICH ثبت شد. تحلیل کوتاه: انتخاب فقط براساس Dice، AUC و حجم خود خونریزی است و fold بیرونی هنوز دیده نشده است. اقدام بعدی: ادامهٔ آموزش تا patience و سپس یک ارزیابی outer.",
                        run=config.run_name,
                        epoch=epoch,
                        dice=f"{dice:.4f}",
                        any_auc=f"{float(calibration_summary['any_ich_study_auc'] or 0):.4f}",
                        volume_mae_ml=f"{float(calibration_summary['total_volume_mae_ml']):.3f}",
                    )
                else:
                    stale_epochs += 1
                if config.patience > 0 and stale_epochs >= config.patience:
                    break

            payload = load_segmentation_weights(model, checkpoint_path)
            model.to(device).eval()
            outer_slices = _predict_slices(model, outer_loader, device=device)
            outer_studies, outer_summary = summarize_segmentation_predictions(
                outer_slices, truth
            )
            outer_slices.to_csv(output / "outer_slice_predictions.csv", index=False)
            outer_studies.to_csv(output / "outer_study_predictions.csv", index=False)
            (output / "outer_summary.json").write_text(
                json.dumps(outer_summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            duration = time.perf_counter() - started
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            summary = {
                "run_name": config.run_name,
                "run_id": run.info.run_id,
                "run_kind": run_kind,
                "best_epoch": best_epoch,
                "best_calibration_checkpoint_score": best_score,
                "best_calibration_selection_score": float(
                    payload["calibration_summary"]["selection_score"]
                ),
                "checkpoint_selection_strategy": config.checkpoint_selection_strategy,
                "sampler_diagnostics": sampler_diagnostics,
                "best_calibration_mean_foreground_dice": best_dice,
                "outer_summary": outer_summary,
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "manifest_sha256": manifest_sha,
                "git_commit": git_commit(),
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
            }
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            mlflow.log_metrics({
                **_flatten_summary_metrics("outer", outer_summary),
                "duration_s": duration,
                "peak_vram_gb": peak_vram,
            })
            mlflow.log_artifacts(str(output), artifact_path="ich_2p5d_segmentation_run")

        notify_campaign(
            "success",
            f"آموزش segmentation دوبعدونیم ICH تمام شد. روی outer fold، selection={float(outer_summary['selection_score']):.4f}، Dice={float(outer_summary['mean_foreground_dice']):.4f}، Any-AUC={float(outer_summary['any_ich_study_auc'] or 0):.4f}، FPR نرمال={float(outer_summary['normal_false_positive_rate_at_0_1ml']):.4f} و MAE حجم={float(outer_summary['total_volume_mae_ml']):.3f}mL ثبت شد. تحلیل کوتاه: checkpoint فقط با calibration انتخاب و outer یک‌بار دیده شده است؛ بااین‌حال بالا بودن یک معیار به‌تنهایی مجوز promotion نیست و باید با reference همان split، FPR و خطای حجم هم‌زمان مقایسه شود. اقدام بعدی: ساخت جدول اختلاف با baseline هم‌fold و ادامه فقط در صورت بهبود قابل تکرار.",
            run=config.run_name,
            kind=run_kind,
            best_epoch=best_epoch,
            selection=f"{float(outer_summary['selection_score']):.4f}",
            dice=f"{float(outer_summary['mean_foreground_dice']):.4f}",
            any_auc=f"{float(outer_summary['any_ich_study_auc'] or 0):.4f}",
            presence_f1=f"{float(outer_summary['presence_f1_at_0_1ml']):.4f}",
            normal_fpr=f"{float(outer_summary['normal_false_positive_rate_at_0_1ml']):.4f}",
            volume_mae_ml=f"{float(outer_summary['total_volume_mae_ml']):.3f}",
            peak_vram_gb=f"{peak_vram:.2f}",
            duration_min=f"{duration / 60:.1f}",
        )
        return summary
    except Exception as exc:
        notify_campaign(
            "failure",
            "آموزش segmentation دوبعدونیم ICH متوقف شد. تحلیل کوتاه: این رخداد تا بررسی traceback نتیجهٔ کیفیتی نیست و checkpoint ناقص پذیرفته نمی‌شود. اقدام بعدی: اصلاح کوچک‌ترین علت فنی و تکرار smoke test.",
            run=config.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
