"""CUDA-only multitask MLS training for low-VRAM local experiments."""

from __future__ import annotations

import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import PROJECT_ROOT, config_section
from src.mlops import (
    context_from_environment,
    experiment_run,
    log_artifact_resilient,
    log_metrics_resilient,
    log_run_summary,
    resilient_mlflow_call,
)
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train import (
    differentiable_keypoints_from_heatmaps,
    differentiable_mls_mm,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "mls-multitask"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt the last state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        str(numpy_state["bit_generator"]),
        numpy_state["state"].cpu().numpy(),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _resolve_resume_checkpoint(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MLS resume checkpoint does not exist: {path}")
    return path


def _append_global_log(run_name: str, status: str, detail: str) -> None:
    path = PROJECT_ROOT / "reports" / "mls_experiments" / "experiment_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# MLS experiment ledger\n\n| UTC | Run | Status | Detail |\n|---|---|---|---|\n",
            encoding="utf-8",
        )
    clean = detail.replace("|", "/").replace("\n", " ")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"| {datetime.now(timezone.utc).isoformat()} | {run_name} | {status} | {clean} |\n"
        )


def spatial_distribution_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted and Gaussian spatial distributions."""
    flat_logits = logits.flatten(2)
    flat_targets = targets.flatten(2)
    distributions = flat_targets / flat_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(distributions * F.log_softmax(flat_logits, dim=-1)).sum(dim=-1).mean()


def multitask_loss(
    heatmap_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    heatmap_targets: torch.Tensor,
    masks: torch.Tensor,
    keypoints_true: torch.Tensor,
    spacing_x: torch.Tensor,
    is_target: torch.Tensor,
    study_mls: torch.Tensor,
    config: MLSHeatmapConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = (is_target > 0.5) & (masks > 0.5).all(dim=1)
    zero = heatmap_logits.new_zeros(())
    spatial = zero
    coordinate = zero
    mls = zero
    threshold = zero
    if valid.any():
        spatial = spatial_distribution_loss(heatmap_logits[valid], heatmap_targets[valid])
        predicted_keypoints = differentiable_keypoints_from_heatmaps(
            heatmap_logits[valid], config.image_size, config.softargmax_temperature,
        )
        coordinate = F.smooth_l1_loss(
            predicted_keypoints / 32.0, keypoints_true[valid] / 32.0,
        )
        predicted_mls = differentiable_mls_mm(predicted_keypoints, spacing_x[valid])
        true_mls = differentiable_mls_mm(keypoints_true[valid], spacing_x[valid])
        mls = F.smooth_l1_loss(predicted_mls, true_mls, beta=1.0)
        thresholds = heatmap_logits.new_tensor([1.0, 3.0, 5.0])
        ordinal_logits = (
            predicted_mls[:, None] - thresholds[None, :]
        ) / config.threshold_temperature_mm
        ordinal_targets = (true_mls[:, None] >= thresholds[None, :]).to(heatmap_logits.dtype)
        threshold = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)
    selector_targets = is_target
    if config.selector_target_mode == "peak_aware_soft":
        selector_targets = torch.zeros_like(is_target)
        if valid.any():
            relative_severity = (true_mls / study_mls[valid].clamp_min(0.1)).clamp(0.0, 1.0)
            relative_severity = relative_severity.pow(config.selector_peak_power)
            selector_targets[valid] = config.selector_peak_base + (
                1.0 - config.selector_peak_base
            ) * relative_severity
    selector = F.binary_cross_entropy_with_logits(selector_logits, selector_targets)
    total = (
        config.spatial_loss_weight * spatial
        + config.coordinate_loss_weight * coordinate
        + config.mls_loss_weight * mls
        + config.threshold_loss_weight * threshold
        + config.selector_loss_weight * selector
    )
    return total, {
        "spatial": spatial.detach(),
        "coordinate": coordinate.detach(),
        "mls": mls.detach(),
        "threshold": threshold.detach(),
        "selector": selector.detach(),
    }


@torch.inference_mode()
def validate(
    model: HRNetHeatmapModel,
    loader,
    device: torch.device,
    config: MLSHeatmapConfig,
) -> dict[str, float]:
    model.eval()
    total_losses: list[float] = []
    selector_truth: list[float] = []
    selector_probs: list[float] = []
    peak_truth: list[float] = []
    validation_rows: list[dict[str, float | str]] = []
    mls_truth: list[float] = []
    mls_prediction: list[float] = []
    keypoint_errors: list[float] = []
    for batch in loader:
        images, targets, masks, keypoints, spacing, is_target, study_mls, study_ids = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)
        is_target = is_target.to(device, non_blocking=True)
        study_mls = study_mls.to(device, non_blocking=True)
        heatmaps, selector = model.forward_multitask(images)
        if not torch.isfinite(heatmaps).all() or not torch.isfinite(selector).all():
            raise FloatingPointError("Non-finite CUDA model output during validation")
        loss, _ = multitask_loss(
            heatmaps, selector, targets, masks, keypoints, spacing, is_target,
            study_mls, config,
        )
        total_losses.append(float(loss))
        selector_truth.extend(is_target.float().cpu().tolist())
        selector_probs.extend(torch.sigmoid(selector).float().cpu().tolist())

        hard_mls = torch.full_like(is_target, config.negative_value_mm)
        heatmap_probabilities = torch.softmax(heatmaps.flatten(2), dim=-1).reshape_as(heatmaps)
        heatmap_confidence = heatmap_probabilities.flatten(2).amax(dim=-1).amin(dim=1)

        valid = is_target > 0.5
        if valid.any():
            selected = heatmaps[valid]
            batch_size, keypoint_count, _, width = selected.shape
            maxima = selected.flatten(2).argmax(dim=-1)
            x = (maxima % width).to(torch.float32) * (config.image_size / width)
            y = torch.div(maxima, width, rounding_mode="floor").to(torch.float32) * (
                config.image_size / selected.shape[-2]
            )
            decoded = torch.stack((x, y), dim=-1)
            truth_kp = keypoints[valid]
            errors = torch.linalg.vector_norm(decoded - truth_kp, dim=-1)
            keypoint_errors.extend(errors.flatten().cpu().tolist())
            predicted_mls = differentiable_mls_mm(decoded, spacing[valid])
            true_mls = differentiable_mls_mm(truth_kp, spacing[valid])
            mls_prediction.extend(predicted_mls.cpu().tolist())
            mls_truth.extend(true_mls.cpu().tolist())
            hard_mls[valid] = predicted_mls
            peak_batch = torch.zeros_like(is_target)
            peak_batch[valid] = (
                true_mls >= 0.9 * study_mls[valid].clamp_min(0.1)
            ).to(is_target.dtype)
        else:
            peak_batch = torch.zeros_like(is_target)
        peak_truth.extend(peak_batch.cpu().tolist())
        probabilities_cpu = torch.sigmoid(selector).float().cpu().tolist()
        for sample_index, study_id in enumerate(study_ids):
            validation_rows.append({
                "study_id": str(study_id),
                "study_mls_mm": float(study_mls[sample_index]),
                "selector_probability": float(probabilities_cpu[sample_index]),
                "mls_mm": float(hard_mls[sample_index]),
                "heatmap_peak": float(heatmap_confidence[sample_index]),
            })

    truth = np.asarray(selector_truth, dtype=int)
    probability = np.asarray(selector_probs, dtype=float)
    binary = probability >= 0.5
    mls_y = np.asarray(mls_truth, dtype=float)
    mls_p = np.asarray(mls_prediction, dtype=float)
    selector_auc = float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else 0.0
    selector_f1 = float(f1_score(truth, binary, zero_division=0))
    peak_y = np.asarray(peak_truth, dtype=int)
    peak_auc = float(roc_auc_score(peak_y, probability)) if len(np.unique(peak_y)) == 2 else 0.0

    study_frame = {}
    for row in validation_rows:
        study_frame.setdefault(row["study_id"], []).append(row)
    study_truth: list[float] = []
    study_prediction: list[float] = []
    for rows in study_frame.values():
        ranked = sorted(rows, key=lambda row: float(row["selector_probability"]), reverse=True)
        study_truth.append(float(rows[0]["study_mls_mm"]))
        if (
            float(ranked[0]["selector_probability"]) < config.selector_threshold
            or sum(
                float(row["selector_probability"]) >= config.selector_threshold for row in rows
            ) < config.min_active_slices
        ):
            study_prediction.append(config.negative_value_mm)
            continue
        selected = ranked[: config.top_k_slices]
        if config.heatmap_guard_ratio > 0:
            maximum_peak = max(float(row["heatmap_peak"]) for row in selected)
            guarded = [
                row for row in selected
                if float(row["heatmap_peak"]) >= maximum_peak * config.heatmap_guard_ratio
            ]
            if guarded:
                selected = guarded
        selected_values = np.asarray([float(row["mls_mm"]) for row in selected], dtype=float)
        if config.aggregation_probability_weighted:
            selected_weights = np.asarray([
                float(row["selector_probability"]) for row in selected
            ], dtype=float)
            order = np.argsort(selected_values)
            cumulative = np.cumsum(np.maximum(selected_weights[order], 1e-8))
            cutoff = config.aggregation_quantile * cumulative[-1]
            quantile_index = min(
                int(np.searchsorted(cumulative, cutoff)), len(selected_values) - 1,
            )
            study_prediction.append(float(selected_values[order][quantile_index]))
        else:
            study_prediction.append(float(np.quantile(
                selected_values, config.aggregation_quantile,
            )))
    study_y = np.asarray(study_truth, dtype=float)
    study_p = np.asarray(study_prediction, dtype=float)
    study_f1_3 = float(f1_score(study_y >= 3, study_p >= 3, zero_division=0))
    study_f1_5 = float(f1_score(study_y >= 5, study_p >= 5, zero_division=0))
    study_mae = float(np.mean(np.abs(study_p - study_y)))
    metrics = {
        "val_loss": float(np.mean(total_losses)),
        "selector_auc": selector_auc,
        "selector_f1": selector_f1,
        "selector_accuracy": float(np.mean(truth == binary)),
        "selector_peak_auc": peak_auc,
        "selector_positive_mean": float(probability[truth == 1].mean()),
        "selector_negative_mean": float(probability[truth == 0].mean()),
        "keypoint_mae_px": float(np.mean(keypoint_errors)),
        "mls_mae_mm": float(np.mean(np.abs(mls_p - mls_y))),
        "mls_rmse_mm": float(np.sqrt(np.mean((mls_p - mls_y) ** 2))),
        "mls_f1_3mm": float(f1_score(mls_y >= 3, mls_p >= 3, zero_division=0)),
        "mls_f1_5mm": float(f1_score(mls_y >= 5, mls_p >= 5, zero_division=0)),
        "study_mls_mae_mm": study_mae,
        "study_mls_f1_3mm": study_f1_3,
        "study_mls_f1_5mm": study_f1_5,
        "study_boundary_f1": 0.5 * (study_f1_3 + study_f1_5),
    }
    metrics["selection_objective"] = (
        metrics["study_mls_mae_mm"]
        + 2.0 * (1.0 - metrics["study_boundary_f1"])
        + 0.5 * (1.0 - metrics["selector_auc"])
    )
    return metrics


def _render_report(
    run_name: str,
    config: MLSHeatmapConfig,
    status: str,
    history: list[dict[str, float]],
    best: dict[str, float] | None,
    run_id: str | None,
    error: str | None = None,
) -> str:
    lines = [
        f"# MLS experiment: {run_name}", "",
        f"- Status: `{status}`",
        f"- Updated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- MLflow run id: `{run_id or 'pending'}`",
        "- Compute policy: model forward/backward/validation are CUDA-only; no CPU fallback.",
        f"- Config: `{json.dumps(config.model_dump(), ensure_ascii=False)}`",
    ]
    if error:
        lines.extend([f"- Error: `{error}`"])
    if best:
        lines.extend(["", "## Best validation", "", "```json", json.dumps(best, indent=2), "```"])
    lines.extend([
        "", "## Epoch history", "",
        "| epoch | train loss | slice MAE | study MAE | study boundary F1 | kp MAE | selector AUC | peak AUC | VRAM GB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in history:
        lines.append(
            f"| {int(row['epoch'])} | {row['train_loss']:.4f} | {row['mls_mae_mm']:.4f} | "
            f"{row['study_mls_mae_mm']:.4f} | {row['study_boundary_f1']:.4f} | "
            f"{row['keypoint_mae_px']:.2f} | {row['selector_auc']:.4f} | "
            f"{row['selector_peak_auc']:.4f} | {row['peak_vram_gb']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def train_mls_multitask(config: MLSHeatmapConfig) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only MLS training requested, but CUDA is unavailable")
    if not config.use_selector or config.dataset_variant != "multitask_v2":
        raise ValueError("Multitask trainer requires use_selector=true and dataset_variant=multitask_v2")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    run_name = os.getenv(
        "IAAA_RUN_NAME",
        f"mls-multitask-{config.backbone}-fold{config.fold}",
    )
    safe_run = _safe_name(run_name)
    run_dir = PROJECT_ROOT / "reports" / "mls_experiments" / safe_run
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints" / "mls_multitask" / safe_run
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _resolve_resume_checkpoint(config.resume_checkpoint)
    report_path = run_dir / "report.md"
    history_path = run_dir / "epoch_metrics.jsonl"
    if resume_path is None:
        _atomic_text(history_path, "")
    _atomic_text(report_path, _render_report(run_name, config, "planned", [], None, None))
    _append_global_log(run_name, "planned", "multitask selector + spatial heatmap loss")

    dataset_root = PROJECT_ROOT / "Data" / "processed" / "mls_multitask_v2"
    csv_path = dataset_root / "mls_labels_multitask.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Build multitask dataset first: {csv_path}")

    context = context_from_environment(
        "mls_heatmap", run_name, config.model_dump(), strategy="mls_heatmap_multitask_v2",
    )
    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_objective = float("inf")
    best_mls_mae = float("inf")
    best_selector_auc = float("-inf")
    best_study_mae = float("inf")
    best_study_boundary_rank = (float("-inf"), float("-inf"))
    epochs_without_improvement = 0
    mlflow_run_id: str | None = None
    try:
        with experiment_run(context) as active_run:
            mlflow_run_id = active_run.info.run_id
            training_tags = {
                "compute_policy": "cuda_only_no_cpu_fallback",
                "dataset_variant": config.dataset_variant,
                "gpu": torch.cuda.get_device_name(0),
            }
            resilient_mlflow_call(
                "set_tags",
                lambda: (mlflow.set_tags(training_tags), True)[1],
                payload={"tags": training_tags},
            )
            _atomic_text(
                report_path,
                _render_report(run_name, config, "running", history, best_metrics, mlflow_run_id),
            )
            train_loader, val_loader = create_mls_dataloaders(
                csv_path=str(csv_path), img_dir=str(dataset_root / "images"),
                img_size=config.image_size, heatmap_size=config.image_size // 4,
                heatmap_sigma=config.heatmap_sigma, batch_size=config.batch_size,
                val_split=config.val_split, augment=True,
                rotation_deg=config.rotation_deg, translation=config.translation,
                intensity_jitter_scale=config.intensity_jitter,
                augment_prob=config.augment_prob, num_workers=config.num_workers,
                seed=config.seed, fold=config.fold,
                use_competition_folds=config.use_competition_folds,
                include_negatives=True, return_selector=True, balanced_sampling=True,
                sampling_mode=config.sampling_mode,
            )
            training_params = {
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "cuda_device": torch.cuda.get_device_name(0),
                "sampling_mode": config.sampling_mode,
            }
            resilient_mlflow_call(
                "log_params",
                lambda: (mlflow.log_params(training_params), True)[1],
                payload={"params": training_params},
            )
            model = HRNetHeatmapModel(
                backbone_name=config.backbone, in_channels=config.input_channels,
                num_keypoints=3, pretrained=resume_path is None,
                head_dropout=config.head_dropout,
                use_selector=True,
            ).to(device)
            if next(model.parameters()).device.type != "cuda":
                raise RuntimeError("CUDA guard failed: model parameters are not on GPU")
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            warmup = min(3, max(1, config.epochs // 10))

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, config.epochs - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
            accumulation = config.gradient_accumulation_steps
            start_epoch = 1

            if resume_path is not None:
                recovery = torch.load(
                    resume_path, map_location=device, weights_only=True,
                )
                if int(recovery.get("schema_version", 0)) < 5:
                    raise ValueError(
                        "MLS resume requires a schema_version>=5 full-state checkpoint"
                    )
                stored_config = recovery.get("config", {})
                for key in ("backbone", "fold", "input_channels", "sampling_mode"):
                    if stored_config.get(key) != config.model_dump().get(key):
                        raise ValueError(
                            f"Resume config mismatch for {key}: "
                            f"checkpoint={stored_config.get(key)!r}, current={config.model_dump().get(key)!r}"
                        )
                model.load_state_dict(recovery["model_state_dict"], strict=True)
                optimizer.load_state_dict(recovery["optimizer_state_dict"])
                scheduler.load_state_dict(recovery["scheduler_state_dict"])
                scaler.load_state_dict(recovery["scaler_state_dict"])
                history = [dict(row) for row in recovery.get("history", [])]
                trainer_state = recovery["trainer_state"]
                best_metrics = trainer_state.get("best_metrics")
                best_objective = float(trainer_state["best_objective"])
                best_mls_mae = float(trainer_state["best_mls_mae"])
                best_selector_auc = float(trainer_state["best_selector_auc"])
                best_study_mae = float(trainer_state["best_study_mae"])
                best_study_boundary_rank = tuple(
                    float(item) for item in trainer_state["best_study_boundary_rank"]
                )
                epochs_without_improvement = int(
                    trainer_state["epochs_without_improvement"]
                )
                start_epoch = int(recovery["epoch"]) + 1
                if start_epoch > config.epochs:
                    raise ValueError(
                        f"Resume checkpoint already completed epoch {start_epoch - 1}, "
                        f"but config.epochs={config.epochs}"
                    )
                _restore_rng_state(recovery["rng_state"])
                history_text = "".join(
                    json.dumps(row) + "\n" for row in history
                )
                _atomic_text(history_path, history_text)
                resume_tags = {
                    "resume.enabled": "true",
                    "resume.from_epoch": str(start_epoch - 1),
                    "resume.checkpoint": resume_path.name,
                }
                resilient_mlflow_call(
                    "set_tags",
                    lambda: (mlflow.set_tags(resume_tags), True)[1],
                    payload={"tags": resume_tags},
                )

            for epoch in range(start_epoch, config.epochs + 1):
                model.train()
                torch.cuda.reset_peak_memory_stats(device)
                running: list[float] = []
                parts = {name: [] for name in ("spatial", "coordinate", "mls", "threshold", "selector")}
                optimizer.zero_grad(set_to_none=True)
                progress = tqdm(train_loader, desc=f"MLS v2 epoch {epoch}/{config.epochs}")
                for batch_index, batch in enumerate(progress, start=1):
                    images, targets, masks, keypoints, spacing, is_target, study_mls, _study_ids = batch
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    keypoints = keypoints.to(device, non_blocking=True)
                    spacing = spacing.to(device, non_blocking=True)
                    is_target = is_target.to(device, non_blocking=True)
                    study_mls = study_mls.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=config.use_amp):
                        heatmaps, selector = model.forward_multitask(images)
                        if not torch.isfinite(heatmaps).all() or not torch.isfinite(selector).all():
                            raise FloatingPointError("Non-finite CUDA model output during training")
                        loss, loss_parts = multitask_loss(
                            heatmaps, selector, targets, masks, keypoints,
                            spacing, is_target, study_mls, config,
                        )
                        scaled_loss = loss / accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite CUDA loss during training")
                    scaler.scale(scaled_loss).backward()
                    if batch_index % accumulation == 0 or batch_index == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    running.append(float(loss.detach()))
                    for name, value in loss_parts.items():
                        parts[name].append(float(value))
                    progress.set_postfix(loss=f"{running[-1]:.3f}")

                metrics = validate(model, val_loader, device, config)
                scheduler.step()
                row = {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(running)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 2**30),
                    **metrics,
                }
                for name, values in parts.items():
                    row[f"train_{name}_loss"] = float(np.mean(values))
                history.append(row)
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                mae_improved = metrics["mls_mae_mm"] < best_mls_mae
                if mae_improved:
                    best_mls_mae = metrics["mls_mae_mm"]
                    _atomic_torch_save({
                        "schema_version": 2,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": metrics,
                        "selection_objective": metrics["selection_objective"],
                        "checkpoint_selection": "best_slice_mls_mae",
                        "mlflow_run_id": mlflow_run_id,
                    }, checkpoint_dir / "mls_multitask_best_mae.pth")

                auc_improved = metrics["selector_auc"] > best_selector_auc
                if auc_improved:
                    best_selector_auc = metrics["selector_auc"]
                    _atomic_torch_save({
                        "schema_version": 2,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": metrics,
                        "selection_objective": metrics["selection_objective"],
                        "checkpoint_selection": "best_selector_auc",
                        "mlflow_run_id": mlflow_run_id,
                    }, checkpoint_dir / "mls_multitask_best_selector_auc.pth")

                study_boundary_rank = (
                    metrics["study_boundary_f1"], -metrics["study_mls_mae_mm"]
                )
                if study_boundary_rank > best_study_boundary_rank:
                    best_study_boundary_rank = study_boundary_rank
                    _atomic_torch_save({
                        "schema_version": 3,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": metrics,
                        "selection_objective": metrics["selection_objective"],
                        "checkpoint_selection": "best_study_boundary_f1_then_mae",
                        "mlflow_run_id": mlflow_run_id,
                    }, checkpoint_dir / "mls_multitask_best_study_boundary.pth")

                study_mae_improved = metrics["study_mls_mae_mm"] < best_study_mae
                if study_mae_improved:
                    best_study_mae = metrics["study_mls_mae_mm"]
                    _atomic_torch_save({
                        "schema_version": 3,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": metrics,
                        "selection_objective": metrics["selection_objective"],
                        "checkpoint_selection": "best_study_mae",
                        "mlflow_run_id": mlflow_run_id,
                    }, checkpoint_dir / "mls_multitask_best_study.pth")

                improved = metrics["selection_objective"] < best_objective
                if improved:
                    best_objective = metrics["selection_objective"]
                    best_metrics = dict(row)
                    epochs_without_improvement = 0
                    _atomic_torch_save({
                        "schema_version": 2,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": metrics,
                        "selection_objective": best_objective,
                        "mlflow_run_id": mlflow_run_id,
                    }, checkpoint_dir / "mls_multitask_best.pth")
                else:
                    epochs_without_improvement += 1

                snapshot_enabled = (
                    config.snapshot_start_epoch > 0
                    and config.snapshot_every_n_epochs > 0
                    and epoch >= config.snapshot_start_epoch
                    and (epoch - config.snapshot_start_epoch) % config.snapshot_every_n_epochs == 0
                )
                if snapshot_enabled:
                    # The slice-level validation proxy can rank full-study inference
                    # incorrectly. Keep sparse local states so a CUDA-only E2E audit
                    # can select the deployable epoch after training. Deliberately do
                    # not upload these ~119 MB files automatically.
                    _atomic_torch_save({
                        "schema_version": 4,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.model_dump(),
                        "val_metrics": metrics,
                        "selection_objective": metrics["selection_objective"],
                        "checkpoint_selection": "periodic_full_study_audit_candidate",
                        "mlflow_run_id": mlflow_run_id,
                    }, checkpoint_dir / f"mls_multitask_epoch_{epoch:03d}.pth")
                recovery_payload = {
                    "schema_version": 5,
                    "checkpoint_selection": "latest_exact_training_recovery",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config.model_dump(),
                    "val_metrics": metrics,
                    "history": history,
                    "trainer_state": {
                        "best_metrics": best_metrics,
                        "best_objective": best_objective,
                        "best_mls_mae": best_mls_mae,
                        "best_selector_auc": best_selector_auc,
                        "best_study_mae": best_study_mae,
                        "best_study_boundary_rank": list(best_study_boundary_rank),
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                    "rng_state": _capture_rng_state(),
                    "mlflow_run_id": mlflow_run_id,
                }
                _atomic_torch_save(
                    recovery_payload,
                    checkpoint_dir / "mls_multitask_resume_latest.pth",
                )
                _atomic_text(
                    report_path,
                    _render_report(run_name, config, "running", history, best_metrics, mlflow_run_id),
                )
                log_metrics_resilient(
                    {key: value for key, value in row.items() if key != "epoch"},
                    step=epoch,
                )
                print(
                    f"epoch={epoch} slice_MAE={metrics['mls_mae_mm']:.3f} "
                    f"study_MAE={metrics['study_mls_mae_mm']:.3f} "
                    f"study_BF1={metrics['study_boundary_f1']:.3f} "
                    f"KP={metrics['keypoint_mae_px']:.1f}px selector_AUC={metrics['selector_auc']:.3f} "
                    f"selector_F1={metrics['selector_f1']:.3f} objective={metrics['selection_objective']:.3f}",
                    flush=True,
                )
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

            final_path = checkpoint_dir / "mls_multitask_final.pth"
            _atomic_torch_save({
                "schema_version": 2,
                "epoch": int(history[-1]["epoch"]),
                "model_state_dict": model.state_dict(),
                "config": config.model_dump(),
                "val_metrics": history[-1],
                "mlflow_run_id": mlflow_run_id,
            }, final_path)
            _atomic_text(
                report_path,
                _render_report(
                    run_name, config, "completed", history, best_metrics, mlflow_run_id,
                ),
            )
            # Remote artifact uploads can take several minutes. Release every
            # training-owned CUDA allocation first so the GPU is immediately
            # available for the checkpoint audit instead of sitting idle with
            # nearly all VRAM reserved while MLflow transfers files.
            del optimizer, scheduler, scaler, model
            torch.cuda.empty_cache()
            print(
                "MLS training complete; CUDA allocations released before MLflow artifact upload.",
                flush=True,
            )
            best_path = checkpoint_dir / "mls_multitask_best.pth"
            model_artifact_path = config_section("mlflow", "artifact_paths", "models")
            report_artifact_path = config_section("mlflow", "artifact_paths", "reports")
            log_artifact_resilient(best_path, artifact_path=model_artifact_path)
            best_mae_path = checkpoint_dir / "mls_multitask_best_mae.pth"
            if best_mae_path.is_file():
                log_artifact_resilient(best_mae_path, artifact_path=model_artifact_path)
            best_auc_path = checkpoint_dir / "mls_multitask_best_selector_auc.pth"
            if best_auc_path.is_file():
                log_artifact_resilient(best_auc_path, artifact_path=model_artifact_path)
            best_study_path = checkpoint_dir / "mls_multitask_best_study.pth"
            if best_study_path.is_file():
                log_artifact_resilient(best_study_path, artifact_path=model_artifact_path)
            # Keep the last epoch locally for audit/recovery. It is normally
            # inferior after early stopping and duplicates another ~119 MB on
            # the remote store, so only selection checkpoints are uploaded.
            log_artifact_resilient(report_path, artifact_path=report_artifact_path)
            log_artifact_resilient(history_path, artifact_path=report_artifact_path)
            log_run_summary({
                "task": "mls", "strategy": "mls_heatmap_multitask_v2",
                "best_metrics": best_metrics, "epochs_completed": len(history),
                "best_checkpoint": str(best_path), "compute_policy": "cuda_only",
            })
        _atomic_text(
            report_path,
            _render_report(run_name, config, "completed", history, best_metrics, mlflow_run_id),
        )
        _append_global_log(
            run_name, "completed",
            f"best MLS MAE={best_metrics['mls_mae_mm']:.4f}, selector AUC={best_metrics['selector_auc']:.4f}",
        )
        return checkpoint_dir / "mls_multitask_best.pth"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _atomic_text(
            report_path,
            _render_report(run_name, config, "failed", history, best_metrics, mlflow_run_id, error),
        )
        _append_global_log(run_name, "failed", error)
        raise
