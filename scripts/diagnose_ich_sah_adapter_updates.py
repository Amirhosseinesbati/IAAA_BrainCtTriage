"""Measure whether the exact exp65 adapter recipe can move/fits train SAH.

The probe performs one train-only optimizer epoch on a fresh zero-initialized
adapter, never reads calibration or outer data, persists no row-level data, and
reports only aggregate parameter and conversion statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import AdamW

from src.strategies.ich_2p5d.segmentation_data import (
    create_segmentation_loaders,
    segmentation_classification_weights,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import ICH25DSegmentationLoss
from src.strategies.ich_2p5d.segmentation_model import (
    SahBackgroundExpansionAdapter,
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import (
    _unpack_outputs,
    configure_trainable_parameters,
    set_segmentation_training_mode,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _quantiles(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {
            "sample_count": 0,
            "q10": None,
            "median": None,
            "q90": None,
            "q99": None,
        }
    array = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "sample_count": int(array.size),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
    }


def update_probe_interpretation(
    *,
    sah_conversion_fraction: float,
    background_conversion_fraction: float,
    sah_saturation_fraction: float,
    sah_residual_q99: float,
    maximum_logit_residual: float,
) -> str:
    if background_conversion_fraction > 0.0001:
        return "train_fit_has_unsafe_background_pressure"
    if sah_conversion_fraction >= 0.05:
        return "adapter_fits_train_selectively_calibration_failure_is_generalization_or_support"
    if sah_saturation_fraction >= 0.10:
        return "adapter_saturates_but_cannot_fit_train_cap_or_support_limited"
    if sah_residual_q99 < 0.50 * maximum_logit_residual:
        return "adapter_under_updates_on_train"
    return "adapter_moves_but_train_margin_or_support_remains_limiting"


def _parameter_delta(
    parameters: tuple[torch.nn.Parameter, ...], initial: tuple[torch.Tensor, ...]
) -> dict[str, float]:
    delta_sq = 0.0
    initial_sq = 0.0
    final_sq = 0.0
    for parameter, start in zip(parameters, initial, strict=True):
        final = parameter.detach().float().cpu()
        delta_sq += float(torch.sum((final - start) ** 2))
        initial_sq += float(torch.sum(start**2))
        final_sq += float(torch.sum(final**2))
    delta = math.sqrt(delta_sq)
    initial_norm = math.sqrt(initial_sq)
    return {
        "delta_l2": delta,
        "initial_l2": initial_norm,
        "final_l2": math.sqrt(final_sq),
        "relative_delta": delta / max(initial_norm, 1e-12),
    }


def _probe_conversions(
    model: SahBackgroundExpansionAdapter,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    positive_batches: int,
    negative_batches: int,
    maximum_scanned_batches: int,
    background_sample_stride: int,
) -> dict[str, Any]:
    model.eval()
    counts = {
        "scanned_batches": 0,
        "positive_batches": 0,
        "negative_batches": 0,
        "eligible_true_sah_pixels": 0,
        "converted_true_sah_pixels": 0,
        "saturated_true_sah_pixels": 0,
        "eligible_true_background_pixels": 0,
        "converted_true_background_pixels": 0,
        "eligible_true_other_hemorrhage_pixels": 0,
        "converted_true_other_hemorrhage_pixels": 0,
    }
    sah_residual_samples: list[np.ndarray] = []
    background_residual_samples: list[np.ndarray] = []
    cap = float(model.maximum_logit_residual)
    with torch.no_grad():
        for batch in loader:
            counts["scanned_batches"] += 1
            cpu_masks = batch["mask"]
            cpu_known = batch["segmentation_known"] > 0.5
            has_sah = bool(torch.any(cpu_known & (cpu_masks == 5).flatten(1).any(1)))
            if has_sah:
                if counts["positive_batches"] >= positive_batches:
                    if counts["scanned_batches"] >= maximum_scanned_batches:
                        break
                    continue
                counts["positive_batches"] += 1
            else:
                if counts["negative_batches"] >= negative_batches:
                    if counts["scanned_batches"] >= maximum_scanned_batches:
                        break
                    continue
                counts["negative_batches"] += 1

            images = batch["image"].to(device, non_blocking=True)
            masks = cpu_masks.to(device, non_blocking=True)
            known = cpu_known.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded, base_logits, _ = model._frozen_base_forward(images)
                raw = model.sah_residual_head(torch.cat([decoded, base_logits], dim=1))
                residual = cap * torch.tanh(raw[:, 0].float())
            base_prediction = base_logits.argmax(dim=1)
            eligible = (base_prediction == 0) & known[:, None, None]
            conversion = eligible & (
                base_logits[:, 5].float() + residual > base_logits[:, 0].float()
            )
            true_sah = (masks == 5) & known[:, None, None]
            true_background = (masks == 0) & known[:, None, None]
            true_other = ((masks > 0) & (masks != 5)) & known[:, None, None]
            sah_eligible = eligible & true_sah
            background_eligible = eligible & true_background
            other_eligible = eligible & true_other
            counts["eligible_true_sah_pixels"] += int(sah_eligible.sum())
            counts["converted_true_sah_pixels"] += int((conversion & true_sah).sum())
            counts["saturated_true_sah_pixels"] += int(
                (sah_eligible & (residual.abs() >= 0.95 * cap)).sum()
            )
            counts["eligible_true_background_pixels"] += int(background_eligible.sum())
            counts["converted_true_background_pixels"] += int(
                (conversion & true_background).sum()
            )
            counts["eligible_true_other_hemorrhage_pixels"] += int(other_eligible.sum())
            counts["converted_true_other_hemorrhage_pixels"] += int(
                (conversion & true_other).sum()
            )
            if torch.any(sah_eligible):
                sah_residual_samples.append(
                    residual[sah_eligible].detach().float().cpu().numpy()
                )
            if torch.any(background_eligible):
                sampled = residual[background_eligible].flatten()[::background_sample_stride]
                background_residual_samples.append(sampled.detach().float().cpu().numpy())
            if (
                counts["positive_batches"] >= positive_batches
                and counts["negative_batches"] >= negative_batches
            ):
                break
            if counts["scanned_batches"] >= maximum_scanned_batches:
                break

    if counts["positive_batches"] < positive_batches:
        raise ValueError("Insufficient SAH-positive probe batches")
    if counts["negative_batches"] < negative_batches:
        raise ValueError("Insufficient SAH-negative probe batches")
    sah_denominator = max(1, counts["eligible_true_sah_pixels"])
    background_denominator = max(1, counts["eligible_true_background_pixels"])
    other_denominator = max(1, counts["eligible_true_other_hemorrhage_pixels"])
    counts.update(
        {
            "true_sah_conversion_fraction": counts["converted_true_sah_pixels"]
            / sah_denominator,
            "true_sah_saturation_fraction": counts["saturated_true_sah_pixels"]
            / sah_denominator,
            "true_background_conversion_fraction": counts[
                "converted_true_background_pixels"
            ]
            / background_denominator,
            "true_other_hemorrhage_conversion_fraction": counts[
                "converted_true_other_hemorrhage_pixels"
            ]
            / other_denominator,
            "true_sah_residual": _quantiles(sah_residual_samples),
            "true_background_residual_sample": _quantiles(background_residual_samples),
        }
    )
    return counts


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SAH adapter update probe requires BF16 CUDA")
    if args.optimizer_steps < 0:
        raise ValueError("optimizer_steps must be zero (full epoch) or positive")
    started = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a training config")
    source = payload["config"]
    expected = {
        "outer_fold": args.outer_fold,
        "calibration_fold": args.calibration_fold,
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"Checkpoint {key} mismatch")
    _seed_everything(args.seed)
    train_loader, _, _, train_frame, _, _ = create_segmentation_loaders(
        args.manifest,
        outer_fold=args.outer_fold,
        calibration_fold=args.calibration_fold,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        sampler_study_balance_power=0.0,
        context_radius=1,
    )
    probe_loader, _, _, _, _, _ = create_segmentation_loaders(
        args.manifest,
        outer_fold=args.outer_fold,
        calibration_fold=args.calibration_fold,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed + 1000,
        sampler_study_balance_power=0.0,
        context_radius=1,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        pretrained=False,
        sah_residual_adapter=True,
        sah_residual_hidden_channels=args.hidden_channels,
        sah_maximum_logit_residual=args.maximum_logit_residual,
    ).to(device)
    if not isinstance(model, SahBackgroundExpansionAdapter):
        raise TypeError("Expected SAH adapter")
    load_segmentation_weights(model.base_model, args.checkpoint)
    parameters = tuple(configure_trainable_parameters(model, freeze_base_model=True))
    if sum(parameter.numel() for parameter in parameters) != 3217:
        raise ValueError("Unexpected trainable parameter count")
    initial_parameters = tuple(
        parameter.detach().float().cpu().clone() for parameter in parameters
    )
    pos_weight = segmentation_classification_weights(train_frame, maximum=20.0).to(device)
    class_weights = segmentation_foreground_weights(
        train_frame, power=1.0, maximum=8.0, basis="pixel"
    ).to(device)
    loss_fn = ICH25DSegmentationLoss(
        classification_pos_weight=pos_weight,
        segmentation_class_weights=class_weights,
        classification_weight=0.0,
        classification_focal_gamma=1.0,
        background_weight=0.15,
        empty_foreground_weight=0.05,
        empty_foreground_top_fraction=0.001,
        sah_tversky_loss_weight=args.sah_tversky_weight,
    ).to(device)
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    set_segmentation_training_mode(model, freeze_base_model=True)
    losses: list[float] = []
    segmentation_losses: list[float] = []
    sah_losses: list[float] = []
    positive_step_count = 0
    for step, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        segmentation_known = batch["segmentation_known"].to(device, non_blocking=True)
        classification_known = batch["classification_known"].to(device, non_blocking=True)
        positive_step_count += int(
            torch.any(
                (segmentation_known > 0.5)
                & (masks == 5).flatten(start_dim=1).any(dim=1)
            )
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
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        losses.append(float(components["loss"].detach()))
        segmentation_losses.append(float(components["segmentation"].detach()))
        sah_losses.append(float(components["sah_tversky"].detach()))
        if args.optimizer_steps and step >= args.optimizer_steps:
            break
    completed_steps = len(losses)
    if args.optimizer_steps and completed_steps != args.optimizer_steps:
        raise ValueError("Requested optimizer steps exceed one train epoch")
    probe = _probe_conversions(
        model,
        probe_loader,
        device=device,
        positive_batches=args.probe_positive_batches,
        negative_batches=args.probe_negative_batches,
        maximum_scanned_batches=args.maximum_probe_scanned_batches,
        background_sample_stride=args.background_sample_stride,
    )
    parameter_delta = _parameter_delta(parameters, initial_parameters)
    sah_q99 = float(probe["true_sah_residual"]["q99"])
    decision = update_probe_interpretation(
        sah_conversion_fraction=float(probe["true_sah_conversion_fraction"]),
        background_conversion_fraction=float(probe["true_background_conversion_fraction"]),
        sah_saturation_fraction=float(probe["true_sah_saturation_fraction"]),
        sah_residual_q99=sah_q99,
        maximum_logit_residual=args.maximum_logit_residual,
    )
    tail = max(1, completed_steps // 4)
    result = {
        "analysis_kind": "train_only_exact_exp65_optimizer_update_probe",
        "decision": decision,
        "train_only_no_calibration_or_outer": True,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "git_commit": git_commit(),
        "optimizer": "AdamW",
        "optimizer_steps": completed_steps,
        "positive_optimizer_steps": positive_step_count,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "sah_tversky_weight": args.sah_tversky_weight,
        "maximum_logit_residual": args.maximum_logit_residual,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "loss_first_quarter_mean": float(np.mean(losses[:tail])),
        "loss_last_quarter_mean": float(np.mean(losses[-tail:])),
        "segmentation_last_quarter_mean": float(np.mean(segmentation_losses[-tail:])),
        "sah_tversky_last_quarter_mean": float(np.mean(sah_losses[-tail:])),
        "parameter_delta": parameter_delta,
        "probe": probe,
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "sah_adapter_update_probe",
                "evaluation_scope": "train_only_no_calibration_or_outer",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "optimizer_steps": completed_steps,
                "learning_rate": args.learning_rate,
                "sah_tversky_weight": args.sah_tversky_weight,
                "maximum_logit_residual": args.maximum_logit_residual,
            }
        )
        mlflow.log_metrics(
            {
                "parameter_delta_l2": parameter_delta["delta_l2"],
                "true_sah_conversion_fraction": probe["true_sah_conversion_fraction"],
                "true_background_conversion_fraction": probe[
                    "true_background_conversion_fraction"
                ],
                "true_other_conversion_fraction": probe[
                    "true_other_hemorrhage_conversion_fraction"
                ],
                "true_sah_residual_q99": sah_q99,
                "duration_s": result["duration_s"],
                "peak_vram_gb": result["peak_vram_gb"],
            }
        )
        mlflow.log_artifact(str(args.output), artifact_path="ich_diagnostics")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.notify:
        notify_campaign(
            "info",
            "🔬 آپدیت‌پروب SAH مسابقه IAAA کامل شد. تحلیل کوتاه: recipe دقیق exp65 "
            "یک epoch فقط روی train بازسازی شد و توان عبور از margin در برابر فشار "
            "خطای پس‌زمینه سنجیده شد؛ این نتیجه مسیر cap/optimizer یا تغییر معماری را تعیین می‌کند.",
            experiment="exp65_postmortem_sah_update_probe",
            decision=decision,
            steps=completed_steps,
            sah_converted=f"{100 * float(probe['true_sah_conversion_fraction']):.3f}%",
            background_converted=f"{100 * float(probe['true_background_conversion_fraction']):.5f}%",
            sah_residual_q99=f"{sah_q99:.3f}",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--architecture", default="unetplusplus")
    parser.add_argument("--encoder-name", default="tu-efficientnetv2_rw_s")
    parser.add_argument("--outer-fold", type=int, default=2)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sah-tversky-weight", type=float, default=0.03)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--maximum-logit-residual", type=float, default=8.0)
    parser.add_argument("--probe-positive-batches", type=int, default=12)
    parser.add_argument("--probe-negative-batches", type=int, default=12)
    parser.add_argument("--maximum-probe-scanned-batches", type=int, default=200)
    parser.add_argument("--background-sample-stride", type=int, default=512)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_probe(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
