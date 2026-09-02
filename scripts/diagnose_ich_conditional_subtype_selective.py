"""Train-only proof for a selectively gated subtype residual corrector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.diagnose_ich_conditional_subtype_refiner import (
    _aggregate_probe,
    _notify,
    _parameter_delta,
    _safe_fraction,
    _seed_everything,
)
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    create_segmentation_loaders,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    conditional_subtype_selective_loss_components,
)
from src.strategies.ich_2p5d.segmentation_model import (
    ConditionalSubtypeSelectiveResidualAdapter,
    build_segmentation_model,
    conditional_subtype_selective_trainable_parameters,
    load_segmentation_weights,
    set_conditional_subtype_selective_training_mode,
)
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
)


MAX_TRAINABLE_PARAMETERS = 5_000
MIN_SAH_FROM_IPH_ERROR_PIXELS = 100
MIN_SAH_FROM_IPH_RECOVERY_FRACTION = 0.10
MAX_CORRECT_IPH_HARM_FRACTION = 0.003
MAX_CORRECT_OTHER_HARM_FRACTION = 0.003
MAX_TRUE_BACKGROUND_SUBTYPE_CHANGE_FRACTION = 0.003
MIN_CONDITIONAL_ACCURACY_GAIN = 0.0
MIN_MACRO_RECALL_GAIN = 0.0
MIN_GATE_ERROR_PRECISION = 0.10
MIN_GATE_ERROR_RECALL = 0.10
MAX_GATE_COVERAGE = 0.05


def _aggregate_gate_probe(
    model: ConditionalSubtypeSelectiveResidualAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    counts = {
        "incumbent_foreground_pixels": 0,
        "incumbent_error_pixels": 0,
        "gate_active_pixels": 0,
        "gate_true_positive_pixels": 0,
        "gate_false_positive_pixels": 0,
        "gate_false_negative_pixels": 0,
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
            incumbent_foreground = incumbent > 0
            known = known_rows[:, None, None].expand_as(masks)
            errors = (
                known
                & incumbent_foreground
                & (masks > 0)
                & (incumbent != masks)
            )
            active = outputs["selection_gate_active"].squeeze(1)
            active = active & incumbent_foreground
            counts["incumbent_foreground_pixels"] += int(
                incumbent_foreground.sum()
            )
            counts["incumbent_error_pixels"] += int(errors.sum())
            counts["gate_active_pixels"] += int(active.sum())
            counts["gate_true_positive_pixels"] += int((active & errors).sum())
            counts["gate_false_positive_pixels"] += int(
                (active & ~errors).sum()
            )
            counts["gate_false_negative_pixels"] += int(
                (~active & errors).sum()
            )
    return {
        **counts,
        "gate_error_precision": _safe_fraction(
            counts["gate_true_positive_pixels"], counts["gate_active_pixels"]
        ),
        "gate_error_recall": _safe_fraction(
            counts["gate_true_positive_pixels"], counts["incumbent_error_pixels"]
        ),
        "gate_coverage": _safe_fraction(
            counts["gate_active_pixels"], counts["incumbent_foreground_pixels"]
        ),
        "incumbent_error_prevalence": _safe_fraction(
            counts["incumbent_error_pixels"],
            counts["incumbent_foreground_pixels"],
        ),
    }


def conditional_subtype_selective_probe_decision(
    metrics: dict[str, Any],
    gate_metrics: dict[str, Any],
    *,
    trainable_parameter_count: int,
) -> dict[str, Any]:
    final = metrics["final"]
    gates = {
        "trainable_parameters_at_most_5000": (
            int(trainable_parameter_count) <= MAX_TRAINABLE_PARAMETERS
        ),
        "initial_hard_mask_identity_exact": (
            int(metrics["initial"]["changed_hard_mask_pixels"]) == 0
        ),
        "foreground_support_lock_exact": (
            int(final["foreground_support_mismatch_pixels"]) == 0
        ),
        "sah_from_iph_error_pixels_at_least_100": (
            int(final["true_sah_predicted_iph_pixels"])
            >= MIN_SAH_FROM_IPH_ERROR_PIXELS
        ),
        "sah_from_iph_recovery_at_least_0_10": (
            float(final["sah_from_iph_recovery_fraction"])
            >= MIN_SAH_FROM_IPH_RECOVERY_FRACTION
        ),
        "correct_iph_harm_at_most_0_003": (
            float(final["correct_iph_harm_fraction"])
            <= MAX_CORRECT_IPH_HARM_FRACTION
        ),
        "correct_other_harm_at_most_0_003": (
            float(final["correct_other_harm_fraction"])
            <= MAX_CORRECT_OTHER_HARM_FRACTION
        ),
        "true_background_subtype_change_at_most_0_003": (
            float(final["true_background_subtype_change_fraction"])
            <= MAX_TRUE_BACKGROUND_SUBTYPE_CHANGE_FRACTION
        ),
        "conditional_accuracy_nonnegative": (
            float(final["conditional_accuracy_delta"])
            >= MIN_CONDITIONAL_ACCURACY_GAIN
        ),
        "macro_recall_nonnegative": (
            float(final["conditional_macro_recall_delta"])
            >= MIN_MACRO_RECALL_GAIN
        ),
        "gate_error_precision_at_least_0_10": (
            float(gate_metrics["gate_error_precision"])
            >= MIN_GATE_ERROR_PRECISION
        ),
        "gate_error_recall_at_least_0_10": (
            float(gate_metrics["gate_error_recall"])
            >= MIN_GATE_ERROR_RECALL
        ),
        "gate_coverage_at_most_0_05": (
            float(gate_metrics["gate_coverage"]) <= MAX_GATE_COVERAGE
        ),
    }
    all_passed = all(gates.values())
    return {
        "gates": {**gates, "all_passed": all_passed},
        "decision": (
            "authorize_one_locked_patient_safe_calibration_screen"
            if all_passed
            else "reject_before_any_calibration_or_outer"
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exp71 requires BF16 CUDA")
    if args.epochs != 1:
        raise ValueError("The preregistered Exp71 probe requires one epoch")
    if args.optimizer_steps < 0:
        raise ValueError("optimizer_steps must be zero or positive")
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
    probe_loader = DataLoader(
        ICHAdjacentSegmentationDataset(train_frame, augment=False),
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
    model = ConditionalSubtypeSelectiveResidualAdapter(
        incumbent,
        hidden_channels=args.hidden_channels,
        maximum_logit_residual=args.maximum_logit_residual,
        conditional_margin=args.conditional_margin,
        gate_threshold=args.gate_threshold,
        initial_gate_probability=args.initial_gate_probability,
    ).to(device)
    parameters = tuple(conditional_subtype_selective_trainable_parameters(model))
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count > MAX_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"Exp71 parameter budget exceeded: {parameter_count} > "
            f"{MAX_TRAINABLE_PARAMETERS}"
        )
    initial_parameters = tuple(
        parameter.detach().float().cpu().clone() for parameter in parameters
    )
    class_weights = segmentation_foreground_weights(
        train_frame,
        power=args.class_weight_power,
        maximum=args.maximum_class_weight,
        basis="pixel",
    ).to(device)

    _notify(
        args.notify,
        "start",
        "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
        "🧭 Exp71 آغاز شد: corrector کم‌ظرفیت Exp70 اکنون یک gate مستقل دارد که "
        "باید پیش از اعمال residual خطای incumbent را تشخیص دهد. تحلیل کوتاه: "
        "هدف مستقیماً افزایش نسبت اصلاح درست به آسیب است؛ gate، correction و "
        "preservation هم‌زمان آموزش می‌بینند ولی خروجی نهایی فقط با آستانه ثابت "
        "۰.۵ فعال می‌شود. calibration و outer همچنان ممنوع‌اند.",
        experiment="exp71_selective_gated_residual_train_probe_v1",
        trainable_parameters=parameter_count,
        train_slices=len(train_frame),
        gate_threshold=f"{args.gate_threshold:.2f}",
        gate_positive_weight=f"{args.gate_positive_weight:.1f}",
    )

    initial_probe = _aggregate_probe(model, probe_loader, device=device)
    initial_gate = _aggregate_gate_probe(model, probe_loader, device=device)
    if int(initial_probe["changed_hard_mask_pixels"]) != 0:
        raise ValueError("Exp71 adapter is not hard-mask identity at initialization")

    optimizer = AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    set_conditional_subtype_selective_training_mode(model)
    histories: dict[str, list[float]] = {
        "loss": [],
        "correction_population": [],
        "stability_population": [],
        "gate": [],
    }
    correction_pixels = 0
    stability_pixels = 0
    gate_positive_pixels = 0
    gate_negative_pixels = 0
    for step, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        segmentation_known = batch["segmentation_known"].to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.forward_components(images)
            components = conditional_subtype_selective_loss_components(
                outputs["subtype_logits"],
                outputs["selection_gate_logits"],
                outputs["incumbent_mask_logits"],
                masks,
                segmentation_known,
                correction_class_weights=class_weights,
                correction_weight=args.correction_weight,
                stability_weight=args.stability_weight,
                gate_weight=args.gate_weight,
                gate_positive_weight=args.gate_positive_weight,
            )
        components["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
        for key in histories:
            histories[key].append(float(components[key].detach()))
        correction_pixels += int(components["correction_pixel_count"])
        stability_pixels += int(components["stability_pixel_count"])
        gate_positive_pixels += int(components["gate_positive_pixel_count"])
        gate_negative_pixels += int(components["gate_negative_pixel_count"])
        if step % 50 == 0:
            print(
                f"step={step}/{len(train_loader)} "
                f"loss={np.mean(histories['loss'][-50:]):.7f} "
                f"gate={np.mean(histories['gate'][-50:]):.5f}",
                flush=True,
            )
        if args.optimizer_steps and step >= args.optimizer_steps:
            break
    if args.optimizer_steps and len(histories["loss"]) != args.optimizer_steps:
        raise ValueError("Requested optimizer steps exceed one train epoch")

    final_probe = _aggregate_probe(model, probe_loader, device=device)
    final_gate = _aggregate_gate_probe(model, probe_loader, device=device)
    metrics = {"initial": initial_probe, "final": final_probe}
    gate_metrics = {"initial": initial_gate, "final": final_gate}
    decision = conditional_subtype_selective_probe_decision(
        metrics,
        final_gate,
        trainable_parameter_count=parameter_count,
    )
    tail = max(1, len(histories["loss"]) // 4)
    result: dict[str, Any] = {
        "analysis_kind": "train_only_selective_gated_subtype_residual_probe",
        "decision": decision["decision"],
        "preregistered_gates": decision["gates"],
        "preregistered_thresholds": {
            "maximum_trainable_parameters": MAX_TRAINABLE_PARAMETERS,
            "minimum_sah_from_iph_error_pixels": MIN_SAH_FROM_IPH_ERROR_PIXELS,
            "minimum_sah_from_iph_recovery_fraction": MIN_SAH_FROM_IPH_RECOVERY_FRACTION,
            "maximum_correct_iph_harm_fraction": MAX_CORRECT_IPH_HARM_FRACTION,
            "maximum_correct_other_harm_fraction": MAX_CORRECT_OTHER_HARM_FRACTION,
            "maximum_true_background_subtype_change_fraction": MAX_TRUE_BACKGROUND_SUBTYPE_CHANGE_FRACTION,
            "minimum_conditional_accuracy_gain": MIN_CONDITIONAL_ACCURACY_GAIN,
            "minimum_macro_recall_gain": MIN_MACRO_RECALL_GAIN,
            "minimum_gate_error_precision": MIN_GATE_ERROR_PRECISION,
            "minimum_gate_error_recall": MIN_GATE_ERROR_RECALL,
            "maximum_gate_coverage": MAX_GATE_COVERAGE,
        },
        "train_only_no_calibration_or_outer": True,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "git_commit": git_commit(),
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
        "trainable_scope": "zero_initialized_residual_plus_supervised_selection_gate",
        "foreground_support_policy": "exact_incumbent_hard_support_lock",
        "classification_policy": "exact_frozen_incumbent_logits",
        "normalization_policy": "groupnorm_adapter_incumbent_eval",
        "loss_normalization": "shared_incumbent_foreground_population",
        "trainable_parameter_count": parameter_count,
        "hidden_channels": args.hidden_channels,
        "maximum_logit_residual": args.maximum_logit_residual,
        "conditional_margin": args.conditional_margin,
        "gate_threshold": args.gate_threshold,
        "initial_gate_probability": args.initial_gate_probability,
        "gate_positive_weight": args.gate_positive_weight,
        "gate_weight": args.gate_weight,
        "epochs": args.epochs,
        "optimizer_steps": len(histories["loss"]),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "correction_weight": args.correction_weight,
        "stability_weight": args.stability_weight,
        "class_weight_power": args.class_weight_power,
        "maximum_class_weight": args.maximum_class_weight,
        "foreground_class_weights": class_weights.detach().cpu().tolist(),
        "correction_pixel_observations": correction_pixels,
        "stability_pixel_observations": stability_pixels,
        "gate_positive_pixel_observations": gate_positive_pixels,
        "gate_negative_pixel_observations": gate_negative_pixels,
        "loss_first_quarter_mean": float(np.mean(histories["loss"][:tail])),
        "loss_last_quarter_mean": float(np.mean(histories["loss"][-tail:])),
        "gate_loss_last_quarter_mean": float(np.mean(histories["gate"][-tail:])),
        "parameter_delta": _parameter_delta(parameters, initial_parameters),
        "probe": metrics,
        "gate_probe": gate_metrics,
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
                "stage": "selective_gated_residual_train_probe",
                "evaluation_scope": "train_only_no_calibration_or_outer",
                "not_promoted": "true",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": result["checkpoint_sha256"],
                "manifest_sha256": result["manifest_sha256"],
                "trainable_parameter_count": parameter_count,
                "learning_rate": args.learning_rate,
                "correction_weight": args.correction_weight,
                "stability_weight": args.stability_weight,
                "gate_weight": args.gate_weight,
                "gate_positive_weight": args.gate_positive_weight,
                "gate_threshold": args.gate_threshold,
                "maximum_logit_residual": args.maximum_logit_residual,
                "optimizer_steps": len(histories["loss"]),
            }
        )
        mlflow.log_metrics(
            {
                "gate_error_precision": final_gate["gate_error_precision"],
                "gate_error_recall": final_gate["gate_error_recall"],
                "gate_coverage": final_gate["gate_coverage"],
                "sah_from_iph_recovery_fraction": final_probe["sah_from_iph_recovery_fraction"],
                "correct_iph_harm_fraction": final_probe["correct_iph_harm_fraction"],
                "correct_other_harm_fraction": final_probe["correct_other_harm_fraction"],
                "true_background_subtype_change_fraction": final_probe["true_background_subtype_change_fraction"],
                "conditional_accuracy_delta": final_probe["conditional_accuracy_delta"],
                "conditional_macro_recall_delta": final_probe["conditional_macro_recall_delta"],
                "parameter_relative_delta": result["parameter_delta"]["relative_delta"],
                "duration_s": result["duration_s"],
                "peak_vram_gb": result["peak_vram_gb"],
            }
        )
        mlflow.log_artifact(str(args.output), artifact_path="ich_diagnostics")
        result["mlflow_run_id"] = run.info.run_id
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    _notify(
        args.notify,
        "info" if decision["gates"]["all_passed"] else "warning",
        "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
        "📊 Exp71 کامل شد. تحلیل کوتاه: ارزش gate با precision/recall و سپس اثر "
        "نهایی روی correction-versus-harm سنجیده شد؛ صرف فعال‌شدن gate موفقیت نیست. "
        "عبور همه گیت‌ها فقط مجوز یک calibration بیمارمحور قفل‌شده است.",
        experiment="exp71_selective_gated_residual_train_probe_v1",
        decision=decision["decision"],
        gates_passed="بله" if decision["gates"]["all_passed"] else "خیر",
        gate_precision=f"{100 * final_gate['gate_error_precision']:.2f}%",
        gate_recall=f"{100 * final_gate['gate_error_recall']:.2f}%",
        gate_coverage=f"{100 * final_gate['gate_coverage']:.2f}%",
        sah_from_iph_recovered=f"{100 * final_probe['sah_from_iph_recovery_fraction']:.2f}%",
        conditional_accuracy_gain=f"{100 * final_probe['conditional_accuracy_delta']:+.3f}pp",
        macro_recall_gain=f"{100 * final_probe['conditional_macro_recall_delta']:+.3f}pp",
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
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-weight", type=float, default=4.0)
    parser.add_argument("--stability-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=0.25)
    parser.add_argument("--gate-positive-weight", type=float, default=200.0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--initial-gate-probability", type=float, default=0.01)
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    parser.add_argument("--maximum-class-weight", type=float, default=2.0)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--maximum-logit-residual", type=float, default=4.0)
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
            "⚠️ Exp71 با خطای فنی متوقف شد. تحلیل کوتاه: این رخداد نتیجهٔ کیفیتی "
            "نیست؛ calibration/outer مصرف نشده و هیچ مدل پذیرفته نشده است. علت فنی "
            "پیش از تکرار همان recipe قفل‌شده بررسی می‌شود.",
            experiment="exp71_selective_gated_residual_train_probe_v1",
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
