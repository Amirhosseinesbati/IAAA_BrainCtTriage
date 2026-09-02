"""Patient-disjoint five-fold development probe for subtype correction.

Each source checkpoint was trained without its declared outer fold.  A copied
decoder/head is trained only on that checkpoint's training folds, while the
incumbent permanently owns foreground support and classification.  Only
aggregate confusion/count metrics are persisted; this is development OOF, not
the competition test or final leaderboard validation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import AdamW

from scripts.diagnose_ich_conditional_subtype_refiner import (
    _aggregate_probe,
    _confusion_metrics,
    _notify,
    _parameter_delta,
    _safe_fraction,
    _seed_everything,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import (
    create_segmentation_loaders,
    segmentation_foreground_weights,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    conditional_subtype_correction_loss_components,
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
)


MIN_AGGREGATE_SAH_FROM_IPH_ERRORS = 100
MIN_AGGREGATE_SAH_FROM_IPH_RECOVERY = 0.10
MAX_AGGREGATE_CORRECT_IPH_HARM = 0.01
MAX_AGGREGATE_CORRECT_OTHER_HARM = 0.01
MAX_AGGREGATE_BACKGROUND_SUBTYPE_CHANGE = 0.02
MIN_AGGREGATE_ACCURACY_GAIN = 0.001
MIN_AGGREGATE_MACRO_RECALL_GAIN = 0.002
MIN_POSITIVE_ACCURACY_FOLDS = 3
MIN_POSITIVE_MACRO_RECALL_FOLDS = 3
MIN_WORST_FOLD_ACCURACY_DELTA = -0.005
MIN_WORST_FOLD_MACRO_RECALL_DELTA = -0.01

COUNT_KEYS = (
    "all_pixels",
    "known_pixels",
    "changed_hard_mask_pixels",
    "foreground_support_mismatch_pixels",
    "true_foreground_pixels",
    "supported_true_foreground_pixels",
    "true_sah_pixels",
    "supported_true_sah_pixels",
    "true_sah_predicted_iph_pixels",
    "true_sah_predicted_iph_recovered_pixels",
    "correct_true_iph_pixels",
    "correct_true_iph_harmed_pixels",
    "correct_true_iph_changed_to_sah_pixels",
    "correct_true_other_pixels",
    "correct_true_other_harmed_pixels",
    "true_background_incumbent_foreground_pixels",
    "true_background_subtype_changed_pixels",
)


@dataclass(frozen=True)
class FoldSpec:
    outer_fold: int
    calibration_fold: int
    checkpoint: Path


def parse_fold_spec(value: str) -> FoldSpec:
    """Parse ``OUTER:CALIBRATION:CHECKPOINT`` without constraining the path."""
    pieces = value.split(":", maxsplit=2)
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(
            "fold checkpoint must be OUTER:CALIBRATION:CHECKPOINT"
        )
    try:
        outer_fold, calibration_fold = (int(pieces[0]), int(pieces[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold indices must be integers") from exc
    if outer_fold not in range(5) or calibration_fold not in range(5):
        raise argparse.ArgumentTypeError("fold indices must be in [0, 4]")
    if outer_fold == calibration_fold:
        raise argparse.ArgumentTypeError("outer and calibration folds must differ")
    return FoldSpec(outer_fold, calibration_fold, Path(pieces[2]))


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _combined_probe(fold_probes: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_probes:
        raise ValueError("At least one fold probe is required")
    counts = {
        key: int(sum(int(probe[key]) for probe in fold_probes))
        for key in COUNT_KEYS
    }
    incumbent_matrix = np.sum(
        [
            np.asarray(
                probe["incumbent_conditional"]
                ["matrix_true_rows_predicted_columns"],
                dtype=np.int64,
            )
            for probe in fold_probes
        ],
        axis=0,
    )
    candidate_matrix = np.sum(
        [
            np.asarray(
                probe["candidate_conditional"]
                ["matrix_true_rows_predicted_columns"],
                dtype=np.int64,
            )
            for probe in fold_probes
        ],
        axis=0,
    )
    incumbent = _confusion_metrics(incumbent_matrix)
    candidate = _confusion_metrics(candidate_matrix)
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
        "incumbent_conditional": incumbent,
        "candidate_conditional": candidate,
        "conditional_accuracy_delta": float(
            candidate["accuracy"] - incumbent["accuracy"]
        ),
        "conditional_macro_recall_delta": float(
            candidate["macro_recall"] - incumbent["macro_recall"]
        ),
    }


def conditional_subtype_oof_decision(
    fold_results: list[dict[str, Any]], aggregate: dict[str, Any]
) -> dict[str, Any]:
    """Apply the preregistered cross-fold development gates."""
    if len(fold_results) != 5:
        raise ValueError("Conditional subtype OOF decision requires five folds")
    finals = [result["probe"]["final"] for result in fold_results]
    initials = [result["probe"]["initial"] for result in fold_results]
    accuracy_deltas = [float(probe["conditional_accuracy_delta"]) for probe in finals]
    macro_deltas = [
        float(probe["conditional_macro_recall_delta"]) for probe in finals
    ]
    gates = {
        "five_unique_outer_folds": {
            int(result["outer_fold"]) for result in fold_results
        }
        == set(range(5)),
        "initial_hard_mask_identity_exact_all_folds": all(
            int(probe["changed_hard_mask_pixels"]) == 0 for probe in initials
        ),
        "foreground_support_lock_exact_all_folds": all(
            int(probe["foreground_support_mismatch_pixels"]) == 0
            for probe in finals
        ),
        "aggregate_sah_from_iph_errors_at_least_100": int(
            aggregate["true_sah_predicted_iph_pixels"]
        )
        >= MIN_AGGREGATE_SAH_FROM_IPH_ERRORS,
        "aggregate_sah_from_iph_recovery_at_least_0_10": float(
            aggregate["sah_from_iph_recovery_fraction"]
        )
        >= MIN_AGGREGATE_SAH_FROM_IPH_RECOVERY,
        "aggregate_correct_iph_harm_at_most_0_01": float(
            aggregate["correct_iph_harm_fraction"]
        )
        <= MAX_AGGREGATE_CORRECT_IPH_HARM,
        "aggregate_correct_other_harm_at_most_0_01": float(
            aggregate["correct_other_harm_fraction"]
        )
        <= MAX_AGGREGATE_CORRECT_OTHER_HARM,
        "aggregate_background_subtype_change_at_most_0_02": float(
            aggregate["true_background_subtype_change_fraction"]
        )
        <= MAX_AGGREGATE_BACKGROUND_SUBTYPE_CHANGE,
        "aggregate_accuracy_gain_at_least_0_001": float(
            aggregate["conditional_accuracy_delta"]
        )
        >= MIN_AGGREGATE_ACCURACY_GAIN,
        "aggregate_macro_recall_gain_at_least_0_002": float(
            aggregate["conditional_macro_recall_delta"]
        )
        >= MIN_AGGREGATE_MACRO_RECALL_GAIN,
        "accuracy_nonnegative_on_at_least_three_folds": sum(
            delta >= 0 for delta in accuracy_deltas
        )
        >= MIN_POSITIVE_ACCURACY_FOLDS,
        "macro_recall_nonnegative_on_at_least_three_folds": sum(
            delta >= 0 for delta in macro_deltas
        )
        >= MIN_POSITIVE_MACRO_RECALL_FOLDS,
        "worst_fold_accuracy_delta_at_least_minus_0_005": min(accuracy_deltas)
        >= MIN_WORST_FOLD_ACCURACY_DELTA,
        "worst_fold_macro_recall_delta_at_least_minus_0_01": min(macro_deltas)
        >= MIN_WORST_FOLD_MACRO_RECALL_DELTA,
    }
    all_passed = all(gates.values())
    return {
        "gates": {**gates, "all_passed": all_passed},
        "accuracy_nonnegative_folds": int(sum(delta >= 0 for delta in accuracy_deltas)),
        "macro_recall_nonnegative_folds": int(sum(delta >= 0 for delta in macro_deltas)),
        "worst_fold_accuracy_delta": float(min(accuracy_deltas)),
        "worst_fold_macro_recall_delta": float(min(macro_deltas)),
        "decision": (
            "authorize_locked_full_metric_oof_screen"
            if all_passed
            else "reject_or_redesign_before_full_metric_oof"
        ),
    }


def _validate_checkpoint(
    payload: dict[str, Any], spec: FoldSpec, args: argparse.Namespace
) -> dict[str, Any]:
    source = payload.get("config")
    if not isinstance(source, dict):
        raise ValueError("Checkpoint must contain a training config")
    expected = {
        "outer_fold": spec.outer_fold,
        "calibration_fold": spec.calibration_fold,
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
    }
    mismatches = {
        key: {"checkpoint": source.get(key), "requested": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint provenance mismatch: {mismatches}")
    if payload.get("output_labels") != OUTPUT_LABELS:
        raise ValueError("Checkpoint output labels do not match")
    if int(payload.get("segmentation_classes", -1)) != 6:
        raise ValueError("Checkpoint segmentation class count does not match")
    return source


def _run_fold(
    spec: FoldSpec, args: argparse.Namespace, *, device: torch.device
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("2.5D segmentation checkpoint must be a dictionary")
    source = _validate_checkpoint(payload, spec, args)
    _seed_everything(args.seed)
    train_loader, _, outer_loader, train_frame, calibration_frame, outer_frame = (
        create_segmentation_loaders(
            args.manifest,
            outer_fold=spec.outer_fold,
            calibration_fold=spec.calibration_fold,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            sampler_study_balance_power=0.0,
            context_radius=1,
        )
    )
    patient_sets = [
        set(frame["patient_id"].astype(str))
        for frame in (train_frame, calibration_frame, outer_frame)
    ]
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Patient overlap detected after split construction")

    incumbent = build_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        pretrained=False,
    )
    load_segmentation_weights(incumbent, spec.checkpoint)
    model = ConditionalSubtypeRefinementModel(
        incumbent, conditional_margin=args.conditional_margin
    ).to(device)
    parameters = tuple(conditional_subtype_trainable_parameters(model))
    initial_parameters = tuple(
        parameter.detach().float().cpu().clone() for parameter in parameters
    )
    class_weights = segmentation_foreground_weights(
        train_frame,
        power=args.class_weight_power,
        maximum=args.maximum_class_weight,
        basis="pixel",
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_probe = _aggregate_probe(model, outer_loader, device=device)
    if int(initial_probe["changed_hard_mask_pixels"]) != 0:
        raise ValueError("Conditional refiner is not identity before training")

    optimizer = AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    set_conditional_subtype_training_mode(model)
    losses: list[float] = []
    correction_losses: list[float] = []
    stability_losses: list[float] = []
    correction_observations = 0
    stability_observations = 0
    for _epoch in range(args.epochs):
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            known = batch["segmentation_known"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.forward_components(images)
                components = conditional_subtype_correction_loss_components(
                    outputs["subtype_logits"],
                    outputs["incumbent_mask_logits"],
                    masks,
                    known,
                    correction_class_weights=class_weights,
                    stability_weight=args.stability_weight,
                )
            components["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            losses.append(float(components["loss"].detach()))
            correction_losses.append(float(components["correction"].detach()))
            stability_losses.append(float(components["stability"].detach()))
            correction_observations += int(
                components["correction_pixel_count"].detach()
            )
            stability_observations += int(
                components["stability_pixel_count"].detach()
            )

    final_probe = _aggregate_probe(model, outer_loader, device=device)
    fold_dir = args.output_dir / f"fold{spec.outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_checkpoint = fold_dir / "diagnostic_refiner.pth"
    torch.save(
        {
            "format_version": 1,
            "artifact_kind": "diagnostic_conditional_subtype_refiner",
            "not_promoted": True,
            "source_checkpoint": str(spec.checkpoint),
            "source_checkpoint_sha256": file_sha256(spec.checkpoint),
            "source_config": source,
            "manifest_sha256": file_sha256(args.manifest),
            "outer_fold": spec.outer_fold,
            "calibration_fold": spec.calibration_fold,
            "subtype_decoder_state_dict": _cpu_state_dict(model.subtype_decoder),
            "subtype_segmentation_head_state_dict": _cpu_state_dict(
                model.subtype_segmentation_head
            ),
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "stability_weight": args.stability_weight,
                "class_weight_power": args.class_weight_power,
                "maximum_class_weight": args.maximum_class_weight,
                "class_weights": class_weights.detach().cpu().tolist(),
                "seed": args.seed,
            },
        },
        diagnostic_checkpoint,
    )
    tail = max(1, len(losses) // 4)
    result = {
        "outer_fold": spec.outer_fold,
        "calibration_fold": spec.calibration_fold,
        "source_checkpoint": str(spec.checkpoint),
        "source_checkpoint_sha256": file_sha256(spec.checkpoint),
        "diagnostic_checkpoint": str(diagnostic_checkpoint),
        "diagnostic_checkpoint_sha256": file_sha256(diagnostic_checkpoint),
        "train_slices": len(train_frame),
        "train_patients": len(patient_sets[0]),
        "calibration_slices": len(calibration_frame),
        "calibration_patients": len(patient_sets[1]),
        "outer_slices": len(outer_frame),
        "outer_patients": len(patient_sets[2]),
        "patient_overlap_counts": {
            "train_calibration": 0,
            "train_outer": 0,
            "calibration_outer": 0,
        },
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "optimizer_steps": len(losses),
        "class_weights": class_weights.detach().cpu().tolist(),
        "correction_pixel_observations": correction_observations,
        "stability_pixel_observations": stability_observations,
        "loss_first_quarter_mean": float(np.mean(losses[:tail])),
        "loss_last_quarter_mean": float(np.mean(losses[-tail:])),
        "correction_loss_last_quarter_mean": float(
            np.mean(correction_losses[-tail:])
        ),
        "stability_loss_last_quarter_mean": float(
            np.mean(stability_losses[-tail:])
        ),
        "parameter_delta": _parameter_delta(parameters, initial_parameters),
        "probe": {"initial": initial_probe, "final": final_probe},
        "duration_s": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    (fold_dir / "probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    del model, incumbent, payload, optimizer
    torch.cuda.empty_cache()
    return result


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Conditional subtype OOF probe requires BF16 CUDA")
    if args.epochs != 1:
        raise ValueError("The preregistered OOF probe requires exactly one epoch")
    specs = list(args.fold_checkpoint)
    if len(specs) != 5 or {spec.outer_fold for spec in specs} != set(range(5)):
        raise ValueError("Exactly one checkpoint for every outer fold 0..4 is required")
    if any(not spec.checkpoint.is_file() for spec in specs):
        raise FileNotFoundError("One or more fold checkpoints do not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _notify(
        args.notify,
        "start",
        "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
        "🔬 آزمون پنج‌فولد پالایش subtype آغاز شد. هر پالایشگر فقط روی بیماران "
        "سه فولد آموزشی همان checkpoint آموزش می‌بیند و روی outer بیمارمحور آن "
        "ارزیابی می‌شود. تحلیل کوتاه: هدف، اصلاح خطاهای واقعی subtype با KL نرم "
        "برای حفظ تصمیم‌های درست است؛ این OOF توسعه‌ای است و test نهایی نیست.",
        experiment="exp69_conditional_subtype_correction_oof_v1",
        folds=5,
        learning_rate=f"{args.learning_rate:.2e}",
        stability_weight=f"{args.stability_weight:.2f}",
    )
    device = torch.device("cuda")
    fold_results = [
        _run_fold(spec, args, device=device)
        for spec in sorted(specs, key=lambda item: item.outer_fold)
    ]
    aggregate = _combined_probe(
        [result["probe"]["final"] for result in fold_results]
    )
    decision = conditional_subtype_oof_decision(fold_results, aggregate)
    result = {
        "analysis_kind": "five_fold_patient_disjoint_conditional_subtype_correction_probe",
        "evaluation_scope": "development_oof_not_final_test_or_leaderboard",
        "outer_folds_previously_seen_by_project": True,
        "recipe_locked_before_this_suite_outer_inference": True,
        "decision": decision["decision"],
        "preregistered_gates": decision["gates"],
        "decision_diagnostics": {
            key: value for key, value in decision.items() if key not in {"gates", "decision"}
        },
        "preregistered_thresholds": {
            "minimum_aggregate_sah_from_iph_errors": MIN_AGGREGATE_SAH_FROM_IPH_ERRORS,
            "minimum_aggregate_sah_from_iph_recovery": MIN_AGGREGATE_SAH_FROM_IPH_RECOVERY,
            "maximum_aggregate_correct_iph_harm": MAX_AGGREGATE_CORRECT_IPH_HARM,
            "maximum_aggregate_correct_other_harm": MAX_AGGREGATE_CORRECT_OTHER_HARM,
            "maximum_aggregate_background_subtype_change": MAX_AGGREGATE_BACKGROUND_SUBTYPE_CHANGE,
            "minimum_aggregate_accuracy_gain": MIN_AGGREGATE_ACCURACY_GAIN,
            "minimum_aggregate_macro_recall_gain": MIN_AGGREGATE_MACRO_RECALL_GAIN,
            "minimum_positive_accuracy_folds": MIN_POSITIVE_ACCURACY_FOLDS,
            "minimum_positive_macro_recall_folds": MIN_POSITIVE_MACRO_RECALL_FOLDS,
            "minimum_worst_fold_accuracy_delta": MIN_WORST_FOLD_ACCURACY_DELTA,
            "minimum_worst_fold_macro_recall_delta": MIN_WORST_FOLD_MACRO_RECALL_DELTA,
        },
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "git_commit": git_commit(),
        "architecture": args.architecture,
        "encoder_name": args.encoder_name,
        "training_objective": "error_only_ground_truth_ce_plus_soft_kl_preservation",
        "normalization_policy": "frozen_running_statistics",
        "foreground_support_policy": "exact_incumbent_hard_support_lock",
        "classification_policy": "exact_frozen_incumbent_logits",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stability_weight": args.stability_weight,
        "class_weight_power": args.class_weight_power,
        "maximum_class_weight": args.maximum_class_weight,
        "folds": fold_results,
        "aggregate": aggregate,
        "duration_s": time.perf_counter() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-diagnostics")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "task": "ich_segmentation_volume",
                "stage": "conditional_subtype_correction_development_oof",
                "evaluation_scope": result["evaluation_scope"],
                "git_commit": result["git_commit"],
            }
        )
        mlflow.log_params(
            {
                "manifest_sha256": result["manifest_sha256"],
                "folds": 5,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "stability_weight": args.stability_weight,
                "class_weight_power": args.class_weight_power,
                "maximum_class_weight": args.maximum_class_weight,
            }
        )
        mlflow.log_metrics(
            {
                "aggregate_sah_from_iph_recovery": aggregate[
                    "sah_from_iph_recovery_fraction"
                ],
                "aggregate_correct_iph_harm": aggregate[
                    "correct_iph_harm_fraction"
                ],
                "aggregate_correct_other_harm": aggregate[
                    "correct_other_harm_fraction"
                ],
                "aggregate_background_subtype_change": aggregate[
                    "true_background_subtype_change_fraction"
                ],
                "aggregate_accuracy_delta": aggregate[
                    "conditional_accuracy_delta"
                ],
                "aggregate_macro_recall_delta": aggregate[
                    "conditional_macro_recall_delta"
                ],
                "worst_fold_accuracy_delta": decision[
                    "worst_fold_accuracy_delta"
                ],
                "worst_fold_macro_recall_delta": decision[
                    "worst_fold_macro_recall_delta"
                ],
                "duration_s": result["duration_s"],
            }
        )
        mlflow.log_artifact(str(output), artifact_path="ich_diagnostics")
        for fold_result in fold_results:
            mlflow.log_artifact(
                fold_result["diagnostic_checkpoint"],
                artifact_path=f"ich_diagnostics/fold{fold_result['outer_fold']}",
            )
        result["mlflow_run_id"] = run.info.run_id
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    _notify(
        args.notify,
        "info" if decision["gates"]["all_passed"] else "warning",
        "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
        "🧪 آزمون پنج‌فولد پالایش subtype کامل شد. تحلیل کوتاه: نتیجه فقط در "
        "صورت بهبود تجمیعی، پایداری حداقل سه فولد و محدودبودن بدترین افت اجازهٔ "
        "محاسبهٔ Dice/حجم کامل را می‌دهد؛ مدل‌های این مرحله diagnostic هستند و "
        "هنوز برای استفاده یا لیدربرد پذیرفته نشده‌اند.",
        experiment="exp69_conditional_subtype_correction_oof_v1",
        decision=decision["decision"],
        gates_passed="yes" if decision["gates"]["all_passed"] else "no",
        sah_from_iph_recovered=(
            f"{100 * aggregate['sah_from_iph_recovery_fraction']:.2f}%"
        ),
        correct_iph_harmed=f"{100 * aggregate['correct_iph_harm_fraction']:.3f}%",
        accuracy_gain=f"{100 * aggregate['conditional_accuracy_delta']:+.3f}pp",
        macro_recall_gain=(
            f"{100 * aggregate['conditional_macro_recall_delta']:+.3f}pp"
        ),
        positive_accuracy_folds=decision["accuracy_nonnegative_folds"],
        positive_macro_folds=decision["macro_recall_nonnegative_folds"],
        duration_min=f"{result['duration_s'] / 60:.1f}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-checkpoint",
        action="append",
        type=parse_fold_spec,
        required=True,
        help="OUTER:CALIBRATION:CHECKPOINT; repeat exactly five times",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--architecture", default="unetplusplus")
    parser.add_argument("--encoder-name", default="efficientnet-b2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stability-weight", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--maximum-class-weight", type=float, default=4.0)
    parser.add_argument("--conditional-margin", type=float, default=1.0)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    try:
        result = run_suite(args)
    except Exception as exc:
        _notify(
            args.notify,
            "failure",
            "🧠 مسابقه IAAA 2026 | مدل خونریزی (ICH)\n\n"
            "⚠️ آزمون پنج‌فولد پالایش subtype با خطای فنی متوقف شد. تحلیل کوتاه: "
            "این رخداد نتیجهٔ کیفیتی نیست و هیچ مدل diagnostic پذیرفته نمی‌شود؛ "
            "اقدام بعدی فقط رفع علت فنی و تکرار همان recipe قفل‌شده است.",
            experiment="exp69_conditional_subtype_correction_oof_v1",
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
