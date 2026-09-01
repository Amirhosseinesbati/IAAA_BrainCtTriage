"""Calibration-only screen for presence-locked soft ICH volume decoding.

The spatial mask, classification scores, and all spatial Dice statistics stay
hard/unchanged. Only the submitted study volumes may use temperature-scaled
softmax mass, and only when hard and soft decoding agree on the 0.1 mL presence
side. The reserved outer fold is never evaluated by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import mlflow
import numpy as np
import pandas as pd
import torch

from scripts.evaluate_ich_2p5d_segmentation_checkpoint import checkpoint_config
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_data import create_segmentation_loaders
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_model import (
    build_segmentation_model,
    load_segmentation_weights,
)
from src.strategies.ich_2p5d.segmentation_train import (
    _flatten_summary_metrics,
    _unpack_outputs,
    checkpoint_selection_score,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


TEMPERATURES = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
PRESENCE_THRESHOLD_ML = 0.1
INVARIANT_METRICS = (
    "selection_score",
    "mean_foreground_dice",
    "any_ich_study_auc",
    "macro_subtype_study_auc",
    "normal_false_positive_rate_at_0_1ml",
    "presence_f1_at_0_1ml",
)


def _temperature_key(temperature: float) -> str:
    return f"t{int(round(temperature * 100)):03d}"


def _validate_temperatures(temperatures: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in temperatures)
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Temperatures must be finite and positive")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Temperatures must be strictly increasing and unique")
    return values


def _predict_hard_and_tempered_mass(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    temperatures: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    temperatures = _validate_temperatures(temperatures)
    model.eval()
    hard_rows: list[dict[str, object]] = []
    soft_rows: dict[float, list[dict[str, object]]] = {
        temperature: [] for temperature in temperatures
    }
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask_logits, class_logits = _unpack_outputs(model(images))
            mask_logits = mask_logits.float()
            hard_masks = mask_logits.argmax(dim=1).cpu()
            class_probabilities = torch.sigmoid(class_logits.float()).cpu().numpy()
            tempered_mass = {
                temperature: torch.softmax(mask_logits / temperature, dim=1)[:, 1:]
                .sum(dim=(-2, -1))
                .cpu()
                .numpy()
                for temperature in temperatures
            }
            true_masks = batch["mask"]
            known = batch["known"].numpy()
            voxel_volumes = batch["voxel_volume_ml"].numpy()
            slice_indices = batch["slice_index"].numpy()
            for index, study_id in enumerate(batch["study_id"]):
                predicted = hard_masks[index]
                observed = true_masks[index]
                is_known = bool(known[index] > 0.5)
                identity = {
                    "study_id": str(study_id),
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_index": int(slice_indices[index]),
                    "known": int(is_known),
                    "voxel_volume_ml": float(voxel_volumes[index]),
                }
                hard_row: dict[str, object] = dict(identity)
                for output_index, label in enumerate(OUTPUT_LABELS):
                    hard_row[f"prob_{label}"] = float(
                        class_probabilities[index, output_index]
                    )
                for class_id, label in enumerate(OUTPUT_LABELS[1:], start=1):
                    predicted_class = predicted == class_id
                    hard_row[f"pred_pixels_{label}"] = int(predicted_class.sum())
                    if is_known:
                        observed_class = observed == class_id
                        hard_row[f"intersection_{label}"] = int(
                            (predicted_class & observed_class).sum()
                        )
                        hard_row[f"predicted_known_pixels_{label}"] = int(
                            predicted_class.sum()
                        )
                        hard_row[f"observed_known_pixels_{label}"] = int(
                            observed_class.sum()
                        )
                    else:
                        hard_row[f"intersection_{label}"] = 0
                        hard_row[f"predicted_known_pixels_{label}"] = 0
                        hard_row[f"observed_known_pixels_{label}"] = 0
                hard_rows.append(hard_row)
                for temperature in temperatures:
                    soft_row: dict[str, object] = dict(identity)
                    for output_index, label in enumerate(OUTPUT_LABELS[1:]):
                        soft_row[f"soft_pixels_{label}"] = float(
                            tempered_mass[temperature][index, output_index]
                        )
                    soft_rows[temperature].append(soft_row)
    return (
        pd.DataFrame(hard_rows),
        {
            temperature: pd.DataFrame(rows)
            for temperature, rows in soft_rows.items()
        },
    )


def presence_locked_soft_predictions(
    hard: pd.DataFrame,
    soft: pd.DataFrame,
    *,
    presence_threshold_ml: float = PRESENCE_THRESHOLD_ML,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace volume pixels only when hard and soft presence decisions agree."""
    if not np.isfinite(presence_threshold_ml) or presence_threshold_ml <= 0:
        raise ValueError("presence_threshold_ml must be finite and positive")
    identity = ["study_id", "patient_id", "slice_index", "voxel_volume_ml"]
    missing_hard = (
        set(identity)
        | {f"pred_pixels_{label}" for label in OUTPUT_LABELS[1:]}
    ) - set(hard)
    missing_soft = (
        set(identity)
        | {f"soft_pixels_{label}" for label in OUTPUT_LABELS[1:]}
    ) - set(soft)
    if missing_hard:
        raise ValueError(f"Hard predictions are missing: {sorted(missing_hard)}")
    if missing_soft:
        raise ValueError(f"Soft predictions are missing: {sorted(missing_soft)}")
    if len(hard) != len(soft) or not hard[identity].reset_index(drop=True).equals(
        soft[identity].reset_index(drop=True)
    ):
        raise ValueError("Hard and soft slice predictions are misaligned")

    candidate = hard.copy().reset_index(drop=True)
    soft = soft.reset_index(drop=True)
    hard_slice_ml = sum(
        candidate[f"pred_pixels_{label}"].to_numpy(dtype=np.float64)
        for label in OUTPUT_LABELS[1:]
    ) * candidate["voxel_volume_ml"].to_numpy(dtype=np.float64)
    soft_slice_ml = sum(
        soft[f"soft_pixels_{label}"].to_numpy(dtype=np.float64)
        for label in OUTPUT_LABELS[1:]
    ) * soft["voxel_volume_ml"].to_numpy(dtype=np.float64)
    if not np.isfinite(hard_slice_ml).all() or not np.isfinite(soft_slice_ml).all():
        raise ValueError("Decoded slice volumes must be finite")
    if np.any(hard_slice_ml < 0) or np.any(soft_slice_ml < 0):
        raise ValueError("Decoded slice volumes must be non-negative")

    study_ids = candidate["study_id"].astype(str)
    hard_total = pd.Series(hard_slice_ml).groupby(study_ids, sort=False).sum()
    soft_total = pd.Series(soft_slice_ml).groupby(study_ids, sort=False).sum()
    hard_positive = hard_total >= presence_threshold_ml
    soft_positive = soft_total >= presence_threshold_ml
    use_soft = hard_positive & soft_positive
    row_use_soft = study_ids.map(use_soft).to_numpy(dtype=bool)
    for label in OUTPUT_LABELS[1:]:
        candidate.loc[row_use_soft, f"pred_pixels_{label}"] = soft.loc[
            row_use_soft, f"soft_pixels_{label}"
        ].to_numpy(dtype=np.float64)

    candidate_slice_ml = sum(
        candidate[f"pred_pixels_{label}"].to_numpy(dtype=np.float64)
        for label in OUTPUT_LABELS[1:]
    ) * candidate["voxel_volume_ml"].to_numpy(dtype=np.float64)
    candidate_total = pd.Series(candidate_slice_ml).groupby(study_ids, sort=False).sum()
    if not np.array_equal(
        (candidate_total >= presence_threshold_ml).to_numpy(),
        hard_positive.to_numpy(),
    ):
        raise AssertionError("Presence lock changed the hard study decision")
    return candidate, {
        "studies_total": int(len(hard_total)),
        "studies_using_soft": int(use_soft.sum()),
        "studies_using_hard": int((~use_soft).sum()),
        "hard_positive_studies": int(hard_positive.sum()),
        "soft_positive_studies_before_lock": int(soft_positive.sum()),
    }


def soft_volume_screen_decision(
    baseline: dict[str, Any],
    candidates: dict[float, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered calibration screen and neighborhood stability gate."""
    temperatures = _validate_temperatures(candidates)
    scored = {
        temperature: checkpoint_selection_score(
            candidates[temperature], "fpr_volume_penalized"
        )
        for temperature in temperatures
    }
    best_temperature = max(
        temperatures,
        key=lambda temperature: (scored[temperature], -temperature),
    )
    candidate = candidates[best_temperature]
    baseline_score = checkpoint_selection_score(
        baseline, "fpr_volume_penalized"
    )
    invariant_checks = {
        name: bool(
            np.isclose(
                float(candidate[name]),
                float(baseline[name]),
                rtol=0.0,
                atol=1e-12,
            )
        )
        for name in INVARIANT_METRICS
    }
    subtype_noninferiority = {
        label: bool(
            float(candidate["subtypes"][label]["mae_ml"])
            <= float(baseline["subtypes"][label]["mae_ml"]) + 0.25
        )
        for label in OUTPUT_LABELS[1:]
    }
    best_index = temperatures.index(best_temperature)
    neighbor_temperatures = [
        temperatures[index]
        for index in (best_index - 1, best_index + 1)
        if 0 <= index < len(temperatures)
    ]
    stable_neighbors = [
        temperature
        for temperature in neighbor_temperatures
        if (
            float(candidates[temperature]["total_volume_mae_ml"])
            <= float(baseline["total_volume_mae_ml"]) - 0.25
            and abs(float(candidates[temperature]["total_volume_bias_ml"]))
            <= abs(float(baseline["total_volume_bias_ml"]))
        )
    ]
    gates = {
        "checkpoint_score_at_least_0_58900": scored[best_temperature] >= 0.58900,
        "mae_improves_at_least_0_5ml": (
            float(candidate["total_volume_mae_ml"])
            <= float(baseline["total_volume_mae_ml"]) - 0.50
        ),
        "mae_at_most_b2_10_26777ml": (
            float(candidate["total_volume_mae_ml"]) <= 10.26777
        ),
        "absolute_bias_at_most_b2_6_06151ml": (
            abs(float(candidate["total_volume_bias_ml"])) <= 6.06151
        ),
        "at_least_one_adjacent_temperature_stable": bool(stable_neighbors),
        "all_presence_spatial_and_auc_metrics_invariant": all(
            invariant_checks.values()
        ),
        "all_subtype_mae_noninferior_0_25ml": all(
            subtype_noninferiority.values()
        ),
    }
    return {
        "decision": (
            "advance_to_crossfit_oof"
            if all(gates.values())
            else "reject_before_outer"
        ),
        "best_temperature": best_temperature,
        "baseline_checkpoint_score": baseline_score,
        "candidate_checkpoint_score": scored[best_temperature],
        "candidate_minus_baseline": {
            "total_volume_mae_ml": float(candidate["total_volume_mae_ml"])
            - float(baseline["total_volume_mae_ml"]),
            "total_volume_bias_ml": float(candidate["total_volume_bias_ml"])
            - float(baseline["total_volume_bias_ml"]),
            "checkpoint_score": scored[best_temperature] - baseline_score,
        },
        "gates": gates,
        "invariant_checks": invariant_checks,
        "subtype_mae_noninferiority": subtype_noninferiority,
        "stable_neighbor_temperatures": stable_neighbors,
        "scores_by_temperature": {
            str(temperature): score for temperature, score in scored.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Tempered soft-volume screening requires CUDA BF16")
    output_path = args.output_dir / "soft_volume_screen.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite soft-volume screen: {output_path}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("ICH checkpoint must be a dictionary")
    config = checkpoint_config(payload)
    workers = int(args.workers if args.workers is not None else config["workers"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    try:
        _, calibration_loader, _, _, calibration_frame, _ = create_segmentation_loaders(
            args.manifest_path,
            outer_fold=int(config["outer_fold"]),
            calibration_fold=int(config["calibration_fold"]),
            batch_size=int(config["batch_size"]),
            workers=workers,
            seed=int(config["seed"]),
            context_radius=int(config.get("slice_context_radius", 1)),
        )
        truth, metadata_source = ground_truth_ich_context()
        truth = truth.loc[:, ["study_id", *[f"gt_{key}" for key in VOLUME_KEYS]]]
        model = build_segmentation_model(
            architecture=str(config["architecture"]),
            encoder_name=str(config["encoder_name"]),
            pretrained=False,
            dropout=float(config["dropout"]),
            horizontal_symmetry_adapter=bool(
                config.get("horizontal_symmetry_adapter", False)
            ),
            five_slice_context_adapter=bool(
                config.get("five_slice_context_adapter", False)
            ),
        ).to(device)
        load_segmentation_weights(model, args.checkpoint)
        hard_slices, soft_by_temperature = _predict_hard_and_tempered_mass(
            model,
            calibration_loader,
            device=device,
            temperatures=TEMPERATURES,
        )
        _, baseline_summary = summarize_segmentation_predictions(hard_slices, truth)
        candidate_summaries: dict[float, dict[str, Any]] = {}
        lock_diagnostics: dict[float, dict[str, int]] = {}
        curve_rows: list[dict[str, object]] = []
        for temperature in TEMPERATURES:
            candidate_slices, lock = presence_locked_soft_predictions(
                hard_slices,
                soft_by_temperature[temperature],
            )
            _, summary = summarize_segmentation_predictions(candidate_slices, truth)
            candidate_summaries[temperature] = summary
            lock_diagnostics[temperature] = lock
            curve_rows.append({
                "temperature": temperature,
                "checkpoint_score": checkpoint_selection_score(
                    summary, "fpr_volume_penalized"
                ),
                "total_volume_mae_ml": summary["total_volume_mae_ml"],
                "total_volume_bias_ml": summary["total_volume_bias_ml"],
                "normal_fpr": summary["normal_false_positive_rate_at_0_1ml"],
                "presence_f1": summary["presence_f1_at_0_1ml"],
                "studies_using_soft": lock["studies_using_soft"],
            })

        decision = soft_volume_screen_decision(
            baseline_summary, candidate_summaries
        )
        duration = time.perf_counter() - started
        best_temperature = float(decision["best_temperature"])
        result = {
            "analysis_kind": "calibration_only_tempered_soft_volume_screen",
            "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
            "outer_evaluation_performed": False,
            "run_name": args.run_name,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "manifest_path": str(args.manifest_path),
            "manifest_sha256": file_sha256(args.manifest_path),
            "metadata_source": str(metadata_source),
            "outer_fold_reserved": int(config["outer_fold"]),
            "calibration_fold": int(config["calibration_fold"]),
            "calibration_studies": int(calibration_frame["study_id"].nunique()),
            "temperatures": list(TEMPERATURES),
            "presence_threshold_ml": PRESENCE_THRESHOLD_ML,
            "selection_policy": (
                "maximize fpr_volume_penalized score on the locked calibration fold; "
                "ties prefer the lower temperature; hard/soft two-sided presence lock"
            ),
            "duration_s": duration,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "baseline_summary": baseline_summary,
            "candidate_summaries": {
                str(temperature): summary
                for temperature, summary in candidate_summaries.items()
            },
            "lock_diagnostics": {
                str(temperature): diagnostics
                for temperature, diagnostics in lock_diagnostics.items()
            },
            "best_candidate_summary": candidate_summaries[best_temperature],
            **decision,
        }
        curve_path = args.output_dir / "soft_volume_temperature_curve.csv"
        pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

        configure_remote_mlflow()
        mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.set_tags({
                "task": "ich_segmentation_volume",
                "stage": "calibration_only_soft_volume_screen",
                "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
                "outer_fold_policy": "not_evaluated",
                "decision": result["decision"],
            })
            mlflow.log_params({
                "checkpoint_sha256": result["checkpoint_sha256"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "manifest_sha256": result["manifest_sha256"],
                "evaluator_git_commit": git_commit(),
                "outer_fold_reserved": result["outer_fold_reserved"],
                "calibration_fold": result["calibration_fold"],
                "temperature_grid": ",".join(map(str, TEMPERATURES)),
                "best_temperature": best_temperature,
                "presence_lock": "hard_and_soft_total_ge_0.1ml",
            })
            mlflow.log_metrics({
                **_flatten_summary_metrics("baseline", baseline_summary),
                **_flatten_summary_metrics(
                    "best_soft", candidate_summaries[best_temperature]
                ),
                "duration_s": duration,
                "peak_vram_gb": result["peak_vram_gb"],
            })
            mlflow.log_artifact(str(output_path), artifact_path="soft_volume_screen")
            mlflow.log_artifact(str(curve_path), artifact_path="soft_volume_screen")
            result["mlflow_run_id"] = run.info.run_id
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

        delta = result["candidate_minus_baseline"]
        notify_campaign(
            "success" if result["decision"] == "advance_to_crossfit_oof" else "info",
            (
                "غربال exp64 برای خوانش حجم نرم کامل شد. تحلیل کوتاه: Dice، AUC، "
                "FPR و F1 با قفل حضور باید دقیقاً ثابت بمانند؛ فقط اگر بهبود حجم "
                "به‌اندازهٔ کافی بزرگ، پایدار در temperature مجاور و بدون قربانی‌کردن "
                "MAE هیچ زیرنوع باشد، اجرای OOF مجاز می‌شود."
            ),
            run=args.run_name,
            decision=result["decision"],
            best_temperature=best_temperature,
            delta_checkpoint=f"{delta['checkpoint_score']:+.5f}",
            delta_mae_ml=f"{delta['total_volume_mae_ml']:+.3f}",
            delta_bias_ml=f"{delta['total_volume_bias_ml']:+.3f}",
            next_step=(
                "پیش‌ثبت cross-fitted OOF"
                if result["decision"] == "advance_to_crossfit_oof"
                else "رد readout نرم و رفتن سراغ معماری مستقل"
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        notify_campaign(
            "failure",
            "غربال exp64 به خطای فنی خورد؛ checkpoint و outer دست‌نخورده‌اند.",
            run=args.run_name,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise


if __name__ == "__main__":
    main()
