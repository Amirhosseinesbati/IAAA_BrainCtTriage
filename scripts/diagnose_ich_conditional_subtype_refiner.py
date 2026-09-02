"""Train-only proof for a two-stage foreground-conditional subtype refiner.

The audited incumbent permanently owns the foreground/background and auxiliary
classification decisions.  A decoder/head copy learns a five-way subtype
decision only inside incumbent foreground.  This script uses train folds only,
persists aggregate metrics only, and never evaluates calibration or outer data.
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
from torch.utils.data import DataLoader

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    create_segmentation_loaders,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    conditional_subtype_loss_components,
)
from src.strategies.ich_2p5d.segmentation_model import (
    ConditionalSubtypeRefinementModel,
    build_segmentation_model,
    conditional_subtype_trainable_parameters,
    load_segmentation_weights,
    set_conditional_subtype_training_mode,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


MIN_SAH_FROM_IPH_RECOVERY_FRACTION = 0.20
MAX_CORRECT_IPH_HARM_FRACTION = 0.01
MAX_CORRECT_OTHER_HARM_FRACTION = 0.01
MAX_TRUE_BACKGROUND_SUBTYPE_CHANGE_FRACTION = 0.02
MIN_CONDITIONAL_ACCURACY_GAIN = 0.005
MIN_MACRO_RECALL_GAIN = 0.01


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _notify(enabled: bool, event: str, message: str, **fields: object) -> None:
    if enabled:
        notify_campaign(event, message, **fields)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / max(1, int(denominator))


def conditional_subtype_probe_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered train-only capacity/selectivity gates."""
    gates = {
        "initial_hard_mask_identity_exact": int(
            metrics["initial"]["changed_hard_mask_pixels"]
        )
        == 0,
        "foreground_support_lock_exact": int(
            metrics["final"]["foreground_support_mismatch_pixels"]
        )
        == 0,
        "sah_from_iph_recovery_at_least_0_20": float(
            metrics["final"]["sah_from_iph_recovery_fraction"]
        )
        >= MIN_SAH_FROM_IPH_RECOVERY_FRACTION,
        "correct_iph_harm_at_most_0_01": float(
            metrics["final"]["correct_iph_harm_fraction"]
        )
        <= MAX_CORRECT_IPH_HARM_FRACTION,
        "correct_other_harm_at_most_0_01": float(
            metrics["final"]["correct_other_harm_fraction"]
        )
        <= MAX_CORRECT_OTHER_HARM_FRACTION,
        "true_background_subtype_change_at_most_0_02": float(
            metrics["final"]["true_background_subtype_change_fraction"]
        )
        <= MAX_TRUE_BACKGROUND_SUBTYPE_CHANGE_FRACTION,
        "conditional_accuracy_gain_at_least_0_005": float(
            metrics["final"]["conditional_accuracy_delta"]
        )
        >= MIN_CONDITIONAL_ACCURACY_GAIN,
        "macro_recall_gain_at_least_0_01": float(
            metrics["final"]["conditional_macro_recall_delta"]
        )
        >= MIN_MACRO_RECALL_GAIN,
    }
    all_passed = all(gates.values())
    return {
        "gates": {**gates, "all_passed": all_passed},
        "decision": (
            "authorize_one_locked_patient_safe_calibration_screen"
            if all_passed
            else "reject_or_redesign_before_any_calibration"
        ),
    }


def _parameter_delta(
    parameters: tuple[torch.nn.Parameter, ...],
    initial: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    delta_sq = 0.0
    initial_sq = 0.0
    final_sq = 0.0
    for parameter, start in zip(parameters, initial, strict=True):
        final = parameter.detach().float().cpu()
        delta_sq += float(torch.sum((final - start) ** 2))
        initial_sq += float(torch.sum(start**2))
        final_sq += float(torch.sum(final**2))
    initial_norm = math.sqrt(initial_sq)
    delta = math.sqrt(delta_sq)
    return {
        "delta_l2": delta,
        "initial_l2": initial_norm,
        "final_l2": math.sqrt(final_sq),
        "relative_delta": delta / max(initial_norm, 1e-12),
    }


def _confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape != (5, 5):
        raise ValueError("Conditional subtype confusion must be 5x5")
    denominators = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        denominators,
        out=np.zeros(5, dtype=np.float64),
        where=denominators > 0,
    )
    available = recalls[denominators > 0]
    return {
        "pixels": int(matrix.sum()),
        "accuracy": _safe_fraction(int(np.trace(matrix)), int(matrix.sum())),
        "macro_recall": float(available.mean()) if len(available) else 0.0,
        "per_subtype_recall": {
            label: float(recalls[index])
            for index, label in enumerate(OUTPUT_LABELS[1:])
        },
        "matrix_true_rows_predicted_columns": matrix.astype(int).tolist(),
    }


def _aggregate_probe(
    model: ConditionalSubtypeRefinementModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    incumbent_confusion = np.zeros((5, 5), dtype=np.int64)
    candidate_confusion = np.zeros((5, 5), dtype=np.int64)
    counts = {
        "all_pixels": 0,
        "known_pixels": 0,
        "changed_hard_mask_pixels": 0,
        "foreground_support_mismatch_pixels": 0,
        "true_foreground_pixels": 0,
        "supported_true_foreground_pixels": 0,
        "true_sah_pixels": 0,
        "supported_true_sah_pixels": 0,
        "true_sah_predicted_iph_pixels": 0,
        "true_sah_predicted_iph_recovered_pixels": 0,
        "correct_true_iph_pixels": 0,
        "correct_true_iph_harmed_pixels": 0,
        "correct_true_iph_changed_to_sah_pixels": 0,
        "correct_true_other_pixels": 0,
        "correct_true_other_harmed_pixels": 0,
        "true_background_incumbent_foreground_pixels": 0,
        "true_background_subtype_changed_pixels": 0,
    }
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            known_rows = batch["segmentation_known"].to(
                device, non_blocking=True
            ) > 0.5
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.forward_components(images)
            incumbent = outputs["incumbent_mask_logits"].argmax(dim=1)
            candidate = outputs["mask_logits"].argmax(dim=1)
            known = known_rows[:, None, None].expand_as(masks)
            incumbent_foreground = incumbent > 0
            candidate_foreground = candidate > 0
            true_foreground = known & (masks > 0)
            supported_true_foreground = true_foreground & incumbent_foreground

            counts["all_pixels"] += int(masks.numel())
            counts["known_pixels"] += int(known.sum())
            counts["changed_hard_mask_pixels"] += int((candidate != incumbent).sum())
            counts["foreground_support_mismatch_pixels"] += int(
                (candidate_foreground != incumbent_foreground).sum()
            )
            counts["true_foreground_pixels"] += int(true_foreground.sum())
            counts["supported_true_foreground_pixels"] += int(
                supported_true_foreground.sum()
            )
            true_sah = known & (masks == 5)
            counts["true_sah_pixels"] += int(true_sah.sum())
            counts["supported_true_sah_pixels"] += int(
                (true_sah & incumbent_foreground).sum()
            )

            if torch.any(supported_true_foreground):
                truth_index = masks[supported_true_foreground].long() - 1
                incumbent_index = incumbent[supported_true_foreground].long() - 1
                candidate_index = candidate[supported_true_foreground].long() - 1
                incumbent_bins = torch.bincount(
                    truth_index * 5 + incumbent_index, minlength=25
                ).reshape(5, 5)
                candidate_bins = torch.bincount(
                    truth_index * 5 + candidate_index, minlength=25
                ).reshape(5, 5)
                incumbent_confusion += incumbent_bins.cpu().numpy()
                candidate_confusion += candidate_bins.cpu().numpy()

            sah_from_iph = true_sah & (incumbent == 2)
            counts["true_sah_predicted_iph_pixels"] += int(sah_from_iph.sum())
            counts["true_sah_predicted_iph_recovered_pixels"] += int(
                (sah_from_iph & (candidate == 5)).sum()
            )
            correct_iph = known & (masks == 2) & (incumbent == 2)
            counts["correct_true_iph_pixels"] += int(correct_iph.sum())
            counts["correct_true_iph_harmed_pixels"] += int(
                (correct_iph & (candidate != 2)).sum()
            )
            counts["correct_true_iph_changed_to_sah_pixels"] += int(
                (correct_iph & (candidate == 5)).sum()
            )
            correct_other = (
                known
                & ((masks == 1) | (masks == 3) | (masks == 4))
                & (incumbent == masks)
            )
            counts["correct_true_other_pixels"] += int(correct_other.sum())
            counts["correct_true_other_harmed_pixels"] += int(
                (correct_other & (candidate != incumbent)).sum()
            )
            background_foreground = known & (masks == 0) & incumbent_foreground
            counts["true_background_incumbent_foreground_pixels"] += int(
                background_foreground.sum()
            )
            counts["true_background_subtype_changed_pixels"] += int(
                (background_foreground & (candidate != incumbent)).sum()
            )

    incumbent_metrics = _confusion_metrics(incumbent_confusion)
    candidate_metrics = _confusion_metrics(candidate_confusion)
    return {
        **counts,
        "incumbent_supported_true_foreground_fraction": _safe_fraction(
            counts["supported_true_foreground_pixels"],
            counts["true_foreground_pixels"],
        ),
        "incumbent_supported_true_sah_fraction": _safe_fraction(
            counts["supported_true_sah_pixels"], counts["true_sah_pixels"]
        ),
        "sah_from_iph_recovery_fraction": _safe_fraction(
            counts["true_sah_predicted_iph_recovered_pixels"],
            counts["true_sah_predicted_iph_pixels"],
        ),
        "correct_iph_harm_fraction": _safe_fraction(
            counts["correct_true_iph_harmed_pixels"],
            counts["correct_true_iph_pixels"],
        ),
        "correct_other_harm_fraction": _safe_fraction(
            counts["correct_true_other_harmed_pixels"],
            counts["correct_true_other_pixels"],
        ),
        "true_background_subtype_change_fraction": _safe_fraction(
            counts["true_background_subtype_changed_pixels"],
            counts["true_background_incumbent_foreground_pixels"],
        ),
        "incumbent_conditional": incumbent_metrics,
        "candidate_conditional": candidate_metrics,
        "conditional_accuracy_delta": float(
            candidate_metrics["accuracy"] - incumbent_metrics["accuracy"]
        ),
        "conditional_macro_recall_delta": float(
            candidate_metrics["macro_recall"] - incumbent_metrics["macro_recall"]
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Conditional subtype probe requires BF16 CUDA")
    if args.epochs != 1:
        raise ValueError("The preregistered train-only probe requires exactly one epoch")
    if args.optimizer_steps < 0:
        raise ValueError("optimizer_steps must be zero or positive")
    if args.stability_weight < 0:
        raise ValueError("stability_weight must be non-negative")
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
    probe_dataset = ICHAdjacentSegmentationDataset(train_frame, augment=False)
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    incumbent = build_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        pretrained=False,
    )
    load_segmentation_weights(incumbent, args.checkpoint)
    model = ConditionalSubtypeRefinementModel(
        incumbent,
        conditional_margin=args.conditional_margin,
    ).to(device)
    parameters = tuple(conditional_subtype_trainable_parameters(model))
    parameter_count = sum(parameter.numel() for parameter in parameters)
    initial_parameters = tuple(
        parameter.detach().float().cpu().clone() for parameter in parameters
    )
    class_weights = segmentation_foreground_weights(
        train_frame,
        power=1.0,
        maximum=8.0,
        basis="pixel",
    ).to(device)

    _notify(
        args.notify,
        "start",
        "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
        "🔬 پروب train-only معماری دو‌مرحله‌ای subtype آغاز شد. مرحلهٔ اول "
        "foreground/background و تمام classification scoreها را قفل می‌کند؛ decoder "
        "و mask-head مرحلهٔ دوم فقط میان پنج زیرنوع تصمیم می‌گیرند. تحلیل کوتاه: "
        "هدف اصلی بازیابی SAHهای اشتباه‌شده به IPH بدون تخریب IPH/سایر زیرنوع‌هاست. "
        "این اجرا calibration و outer را ارزیابی نمی‌کند و checkpoint نگه نمی‌دارد.",
        experiment="exp68_pre_conditional_subtype_decoder_train_probe_v1",
        trainable_parameters=parameter_count,
        train_slices=len(train_frame),
        learning_rate=f"{args.learning_rate:.2e}",
        stability_weight=f"{args.stability_weight:.3f}",
    )

    initial_probe = _aggregate_probe(model, probe_loader, device=device)
    if initial_probe["changed_hard_mask_pixels"] != 0:
        raise ValueError(
            "Conditional subtype refiner is not identity at initialization: "
            f"changed={initial_probe['changed_hard_mask_pixels']} "
            f"of={initial_probe['all_pixels']}"
        )
    optimizer = AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    set_conditional_subtype_training_mode(model)
    loss_history: list[float] = []
    supervised_history: list[float] = []
    stability_history: list[float] = []
    supervised_pixels = 0
    stability_pixels = 0
    for step, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        segmentation_known = batch["segmentation_known"].to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.forward_components(images)
            components = conditional_subtype_loss_components(
                outputs["subtype_logits"],
                outputs["incumbent_mask_logits"],
                masks,
                segmentation_known,
                foreground_class_weights=class_weights,
                stability_weight=args.stability_weight,
            )
        components["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        loss_history.append(float(components["loss"].detach()))
        supervised_history.append(float(components["supervised"].detach()))
        stability_history.append(float(components["stability"].detach()))
        incumbent_prediction = outputs["incumbent_mask_logits"].detach().argmax(dim=1)
        known = (segmentation_known > 0.5)[:, None, None]
        supervised = known & (masks > 0) & (incumbent_prediction > 0)
        supervised_pixels += int(supervised.sum())
        stability_pixels += int(((incumbent_prediction > 0) & ~supervised).sum())
        if step % 50 == 0:
            print(
                f"step={step}/{len(train_loader)} "
                f"loss={np.mean(loss_history[-50:]):.5f}",
                flush=True,
            )
        if args.optimizer_steps and step >= args.optimizer_steps:
            break
    if args.optimizer_steps and len(loss_history) != args.optimizer_steps:
        raise ValueError("Requested optimizer steps exceed one train epoch")

    final_probe = _aggregate_probe(model, probe_loader, device=device)
    metrics = {"initial": initial_probe, "final": final_probe}
    decision = conditional_subtype_probe_decision(metrics)
    tail = max(1, len(loss_history) // 4)
    result = {
        "analysis_kind": "train_only_two_stage_conditional_subtype_decoder_probe",
        "decision": decision["decision"],
        "preregistered_gates": decision["gates"],
        "preregistered_thresholds": {
            "minimum_sah_from_iph_recovery_fraction": MIN_SAH_FROM_IPH_RECOVERY_FRACTION,
            "maximum_correct_iph_harm_fraction": MAX_CORRECT_IPH_HARM_FRACTION,
            "maximum_correct_other_harm_fraction": MAX_CORRECT_OTHER_HARM_FRACTION,
            "maximum_true_background_subtype_change_fraction": MAX_TRUE_BACKGROUND_SUBTYPE_CHANGE_FRACTION,
            "minimum_conditional_accuracy_gain": MIN_CONDITIONAL_ACCURACY_GAIN,
            "minimum_macro_recall_gain": MIN_MACRO_RECALL_GAIN,
        },
        "train_only_no_calibration_or_outer": True,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "git_commit": git_commit(),
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
        "trainable_scope": "copied_decoder_and_segmentation_head_encoder_frozen",
        "foreground_support_policy": "exact_incumbent_hard_support_lock",
        "classification_policy": "exact_frozen_incumbent_logits",
        "trainable_parameter_count": parameter_count,
        "epochs": args.epochs,
        "optimizer_steps": len(loss_history),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stability_weight": args.stability_weight,
        "conditional_margin": args.conditional_margin,
        "foreground_class_weights": class_weights.detach().cpu().tolist(),
        "supervised_pixel_observations": supervised_pixels,
        "stability_pixel_observations": stability_pixels,
        "loss_first_quarter_mean": float(np.mean(loss_history[:tail])),
        "loss_last_quarter_mean": float(np.mean(loss_history[-tail:])),
        "supervised_loss_last_quarter_mean": float(
            np.mean(supervised_history[-tail:])
        ),
        "stability_loss_last_quarter_mean": float(
            np.mean(stability_history[-tail:])
        ),
        "parameter_delta": _parameter_delta(parameters, initial_parameters),
        "probe": metrics,
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "two_stage_conditional_subtype_train_probe",
                "evaluation_scope": "train_only_no_calibration_or_outer",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "trainable_parameter_count": parameter_count,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "stability_weight": args.stability_weight,
                "conditional_margin": args.conditional_margin,
                "optimizer_steps": len(loss_history),
                "foreground_class_weights": json.dumps(
                    result["foreground_class_weights"]
                ),
            }
        )
        mlflow.log_metrics(
            {
                "sah_from_iph_recovery_fraction": final_probe[
                    "sah_from_iph_recovery_fraction"
                ],
                "correct_iph_harm_fraction": final_probe[
                    "correct_iph_harm_fraction"
                ],
                "correct_other_harm_fraction": final_probe[
                    "correct_other_harm_fraction"
                ],
                "true_background_subtype_change_fraction": final_probe[
                    "true_background_subtype_change_fraction"
                ],
                "conditional_accuracy_delta": final_probe[
                    "conditional_accuracy_delta"
                ],
                "conditional_macro_recall_delta": final_probe[
                    "conditional_macro_recall_delta"
                ],
                "incumbent_supported_true_sah_fraction": final_probe[
                    "incumbent_supported_true_sah_fraction"
                ],
                "parameter_relative_delta": result["parameter_delta"][
                    "relative_delta"
                ],
                "duration_s": result["duration_s"],
                "peak_vram_gb": result["peak_vram_gb"],
            }
        )
        mlflow.log_artifact(str(args.output), artifact_path="ich_diagnostics")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    _notify(
        args.notify,
        "info" if decision["gates"]["all_passed"] else "warning",
        "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
        "🧪 پروب دو‌مرحله‌ای subtype کامل شد. تحلیل کوتاه: foreground و "
        "classification دقیقاً تحت کنترل incumbent باقی مانده‌اند؛ تصمیم ادامه فقط "
        "بر اساس بازیابی SAH↔IPH، حفظ زیرنوع‌های درست و بهبود accuracy/macro-recall "
        "گرفته شده است. عبور، صرفاً مجوز یک calibration قفل‌شده است و نه promotion.",
        experiment="exp68_pre_conditional_subtype_decoder_train_probe_v1",
        decision=decision["decision"],
        gates_passed="yes" if decision["gates"]["all_passed"] else "no",
        sah_from_iph_recovered=(
            f"{100 * final_probe['sah_from_iph_recovery_fraction']:.2f}%"
        ),
        correct_iph_harmed=(
            f"{100 * final_probe['correct_iph_harm_fraction']:.3f}%"
        ),
        other_correct_harmed=(
            f"{100 * final_probe['correct_other_harm_fraction']:.3f}%"
        ),
        conditional_accuracy_gain=(
            f"{100 * final_probe['conditional_accuracy_delta']:+.3f}pp"
        ),
        macro_recall_gain=(
            f"{100 * final_probe['conditional_macro_recall_delta']:+.3f}pp"
        ),
        peak_vram_gb=f"{result['peak_vram_gb']:.2f}",
        duration_min=f"{result['duration_s'] / 60:.1f}",
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
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--optimizer-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--conditional-margin", type=float, default=1.0)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    try:
        result = run_probe(args)
    except Exception as exc:
        _notify(
            args.notify,
            "failure",
            "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
            "⚠️ پروب train-only معماری دو‌مرحله‌ای با خطای فنی متوقف شد. تحلیل "
            "کوتاه: این رخداد نتیجهٔ کیفیتی نیست؛ calibration/outer استفاده نشده و "
            "هیچ checkpoint پذیرفته نمی‌شود. اقدام بعدی اصلاح کوچک‌ترین علت فنی و "
            "تکرار دقیق recipe پیش‌ثبت‌شده است.",
            experiment="exp68_pre_conditional_subtype_decoder_train_probe_v1",
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
