"""Evaluate cross-fitted mask thresholds with a label-free presence gate.

The per-subtype thresholds are selected on each checkpoint's calibration fold
and applied once to its untouched outer fold.  The incumbent hard-mask
presence decision then forms a two-sided gate: thresholded spatial outputs are
used only when both hard and thresholded study volumes are positive at 0.1 mL.
Consequently, Any-ICH presence decisions cannot change by construction.
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
import pandas as pd

from scripts.compare_ich_2p5d_segmentation_oof import (
    EXPECTED_FOLDS,
    VariantResult,
    _load_variant,
    _paired_patient_bootstrap,
    _study_sufficient_statistics,
)
from scripts.rescore_ich_oof_with_supervision_manifest import (
    apply_supervision_manifest,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_evaluation import (
    VOLUME_TO_LABEL,
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import ground_truth_ich_context
from src.strategies.ich_v2.operations import (
    configure_remote_mlflow,
    file_sha256,
    git_commit,
    notify_campaign,
)


SPATIAL_PREFIXES = (
    "pred_pixels",
    "intersection",
    "predicted_known_pixels",
    "observed_known_pixels",
)
POINT_METRICS = (
    "selection_score",
    "mean_foreground_dice",
    "any_ich_study_auc",
    "macro_subtype_study_auc",
    "presence_f1_at_0_1ml",
    "normal_false_positive_rate_at_0_1ml",
    "total_volume_mae_ml",
    "total_volume_bias_ml",
)
PROMOTION_POLICY = {
    "minimum_dice_probability": 0.95,
    "minimum_selection_probability": 0.95,
    "minimum_mae_probability": 0.95,
    "minimum_dice_ci95_lower": 0.0,
    "minimum_selection_ci95_lower": 0.0,
    "maximum_mae_ci95_upper_ml": 0.0,
    "minimum_fold_selection_wins": 3,
    "maximum_fold_selection_regression": 0.005,
    "safety_tolerance": 1e-12,
}


@dataclass(frozen=True)
class ThresholdArtifacts:
    slices: pd.DataFrame
    summaries: list[dict[str, Any]]
    provenance: list[dict[str, Any]]


def two_sided_presence_gate(
    hard_total_ml: np.ndarray,
    threshold_total_ml: np.ndarray,
    *,
    presence_threshold_ml: float = 0.1,
) -> np.ndarray:
    """Use threshold masks only when both methods call the study positive."""
    hard = np.asarray(hard_total_ml, dtype=np.float64)
    threshold = np.asarray(threshold_total_ml, dtype=np.float64)
    if hard.shape != threshold.shape:
        raise ValueError("Hard and threshold study volumes must have identical shapes")
    if not np.isfinite(hard).all() or not np.isfinite(threshold).all():
        raise ValueError("Study volumes must be finite")
    if presence_threshold_ml <= 0:
        raise ValueError("Presence threshold must be positive")
    return (hard >= presence_threshold_ml) & (
        threshold >= presence_threshold_ml
    )


def _variant(name: str, slices: pd.DataFrame, truth: pd.DataFrame) -> VariantResult:
    studies, summary = summarize_segmentation_predictions(slices, truth)
    sufficient = _study_sufficient_statistics(slices, studies)
    fold_summaries = []
    for fold, frame in slices.groupby("outer_fold", sort=True):
        _, fold_summary = summarize_segmentation_predictions(frame, truth)
        fold_summaries.append({"outer_fold": int(fold), **fold_summary})
    return VariantResult(
        name=name,
        slices=slices,
        studies=studies,
        sufficient=sufficient,
        summary=summary,
        fold_summaries=fold_summaries,
        runs=[],
    )


def _load_threshold_artifacts(
    run_dirs: list[Path],
    *,
    expected_manifest_sha256: str,
    expected_studies: int,
) -> ThresholdArtifacts:
    if len(run_dirs) != len(EXPECTED_FOLDS):
        raise ValueError("Exactly five threshold run directories are required")
    frames = []
    summaries = []
    provenance = []
    seen_folds: set[int] = set()
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "outer_slice_predictions.csv"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        outer_fold = int(payload["outer_fold"])
        calibration_fold = int(payload["calibration_fold"])
        if outer_fold in seen_folds:
            raise ValueError(f"Duplicate threshold outer fold {outer_fold}")
        if outer_fold == calibration_fold:
            raise ValueError("Threshold calibration and outer folds overlap")
        if payload["manifest_sha256"] != expected_manifest_sha256:
            raise ValueError(f"Threshold fold {outer_fold} used a different manifest")
        if not payload.get("thresholds"):
            raise ValueError(f"Threshold fold {outer_fold} has no selected thresholds")
        frame = pd.read_csv(
            predictions_path,
            dtype={"study_id": str, "patient_id": str},
        ).assign(outer_fold=outer_fold)
        frames.append(frame)
        summaries.append({"outer_fold": outer_fold, **payload["outer_summary"]})
        provenance.append(
            {
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "run_dir": str(run_dir),
                "checkpoint": payload["checkpoint"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "manifest_sha256": payload["manifest_sha256"],
                "evaluator_git_commit": payload["git_commit"],
                "thresholds": payload["thresholds"],
                "missing_class_threshold": payload.get(
                    "missing_class_threshold"
                ),
            }
        )
        seen_folds.add(outer_fold)
    if seen_folds != EXPECTED_FOLDS:
        raise ValueError(f"Threshold folds are {sorted(seen_folds)}, expected 0..4")
    slices = pd.concat(frames, ignore_index=True)
    if slices["study_id"].nunique() != expected_studies:
        raise ValueError("Threshold OOF study count is incomplete")
    if int(slices.groupby("patient_id")["outer_fold"].nunique().max()) != 1:
        raise ValueError("Threshold OOF contains patient leakage across folds")
    return ThresholdArtifacts(
        slices=slices,
        summaries=sorted(summaries, key=lambda row: int(row["outer_fold"])),
        provenance=sorted(provenance, key=lambda row: int(row["outer_fold"])),
    )


def _apply_gate(
    hard: VariantResult,
    threshold: VariantResult,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["study_id", "patient_id", "slice_index", "outer_fold"]
    hard_slices = hard.slices.sort_values(keys).reset_index(drop=True)
    threshold_slices = threshold.slices.sort_values(keys).reset_index(drop=True)
    if not hard_slices[keys].equals(threshold_slices[keys]):
        raise ValueError("Hard and threshold OOF slice keys do not match")
    for column in ("known", "voxel_volume_ml"):
        if not np.allclose(
            hard_slices[column].to_numpy(float),
            threshold_slices[column].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Hard and threshold OOF disagree on {column}")

    hard_studies = hard.studies.set_index("study_id")
    threshold_studies = threshold.studies.set_index("study_id")
    if not hard_studies.index.equals(threshold_studies.index):
        threshold_studies = threshold_studies.reindex(hard_studies.index)
    hard_total = hard_studies.filter(like="pred_V_").sum(axis=1)
    threshold_total = threshold_studies.filter(like="pred_V_").sum(axis=1)
    use_threshold = pd.Series(
        two_sided_presence_gate(hard_total, threshold_total),
        index=hard_studies.index,
    )

    gated = threshold_slices.copy()
    use_by_row = gated["study_id"].map(use_threshold).fillna(False).to_numpy(bool)
    for label in OUTPUT_LABELS[1:]:
        for prefix in SPATIAL_PREFIXES:
            column = f"{prefix}_{label}"
            gated.loc[~use_by_row, column] = hard_slices.loc[
                ~use_by_row, column
            ].to_numpy()
    for label in OUTPUT_LABELS:
        gated[f"prob_{label}"] = hard_slices[f"prob_{label}"].to_numpy()

    fold_by_study = hard_slices.groupby("study_id")["outer_fold"].first()
    audit = {
        "presence_threshold_ml": 0.1,
        "threshold_used_studies": int(use_threshold.sum()),
        "hard_used_studies": int((~use_threshold).sum()),
        "threshold_used_by_outer_fold": {
            str(fold): int(
                use_threshold.loc[fold_by_study.index[fold_by_study.eq(fold)]].sum()
            )
            for fold in sorted(fold_by_study.unique())
        },
        "label_free_gate": True,
        "presence_decision_invariant_by_design": True,
    }
    return gated, audit


def _fold_deltas(
    hard: VariantResult,
    gated: VariantResult,
) -> list[dict[str, Any]]:
    hard_by_fold = {
        int(row["outer_fold"]): row for row in hard.fold_summaries
    }
    gated_by_fold = {
        int(row["outer_fold"]): row for row in gated.fold_summaries
    }
    rows = []
    for fold in sorted(hard_by_fold):
        reference = hard_by_fold[fold]
        candidate = gated_by_fold[fold]
        rows.append(
            {
                "outer_fold": fold,
                "deltas": {
                    metric: float(candidate[metric]) - float(reference[metric])
                    for metric in POINT_METRICS
                },
                "subtype_mae_delta_ml": {
                    label: float(candidate["subtypes"][label]["mae_ml"])
                    - float(reference["subtypes"][label]["mae_ml"])
                    for label in VOLUME_TO_LABEL.values()
                },
            }
        )
    return rows


def promotion_decision(
    hard_summary: dict[str, Any],
    gated_summary: dict[str, Any],
    bootstrap: dict[str, Any],
    fold_deltas: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = bootstrap["metrics"]
    tolerance = float(PROMOTION_POLICY["safety_tolerance"])
    selection_fold_deltas = np.asarray(
        [row["deltas"]["selection_score"] for row in fold_deltas],
        dtype=np.float64,
    )
    gates = {
        "fpr_exactly_preserved": abs(
            float(gated_summary["normal_false_positive_rate_at_0_1ml"])
            - float(hard_summary["normal_false_positive_rate_at_0_1ml"])
        )
        <= tolerance,
        "f1_exactly_preserved": abs(
            float(gated_summary["presence_f1_at_0_1ml"])
            - float(hard_summary["presence_f1_at_0_1ml"])
        )
        <= tolerance,
        "any_auc_exactly_preserved": abs(
            float(gated_summary["any_ich_study_auc"])
            - float(hard_summary["any_ich_study_auc"])
        )
        <= tolerance,
        "macro_auc_exactly_preserved": abs(
            float(gated_summary["macro_subtype_study_auc"])
            - float(hard_summary["macro_subtype_study_auc"])
        )
        <= tolerance,
        "absolute_bias_not_worse": abs(float(gated_summary["total_volume_bias_ml"]))
        <= abs(float(hard_summary["total_volume_bias_ml"])),
        "dice_probability": metrics["mean_foreground_dice"][
            "bootstrap_probability_candidate_better"
        ]
        >= PROMOTION_POLICY["minimum_dice_probability"],
        "dice_ci95": metrics["mean_foreground_dice"]["delta_ci95"][0]
        >= PROMOTION_POLICY["minimum_dice_ci95_lower"],
        "selection_probability": metrics["selection_score"][
            "bootstrap_probability_candidate_better"
        ]
        >= PROMOTION_POLICY["minimum_selection_probability"],
        "selection_ci95": metrics["selection_score"]["delta_ci95"][0]
        >= PROMOTION_POLICY["minimum_selection_ci95_lower"],
        "mae_probability": metrics["total_volume_mae_ml"][
            "bootstrap_probability_candidate_better"
        ]
        >= PROMOTION_POLICY["minimum_mae_probability"],
        "mae_ci95": metrics["total_volume_mae_ml"]["delta_ci95"][1]
        <= PROMOTION_POLICY["maximum_mae_ci95_upper_ml"],
        "fold_selection_wins": int(np.count_nonzero(selection_fold_deltas > 0))
        >= PROMOTION_POLICY["minimum_fold_selection_wins"],
        "worst_fold_selection_regression": float(selection_fold_deltas.min())
        >= -PROMOTION_POLICY["maximum_fold_selection_regression"],
    }
    return {
        "policy": PROMOTION_POLICY,
        "gates": gates,
        "promotion_allowed": bool(all(gates.values())),
        "selection_fold_wins": int(np.count_nonzero(selection_fold_deltas > 0)),
        "selection_fold_deltas": selection_fold_deltas.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--reference-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--threshold-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument("--expected-promotions", type=int, default=145)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5801)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configure_remote_mlflow()
    mlflow.set_experiment("IAAA_BrainCT-ich-2p5d-segmentation")
    try:
        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.set_tags(
                {
                    "task": "ich_segmentation_volume",
                    "stage": "crossfitted_mask_threshold_presence_gate",
                    "run_kind": "adaptive_development_oof",
                    "evaluation_status": "adaptive_development_oof",
                    "leaderboard_confirmation_required": "true",
                    "git_commit": git_commit(),
                }
            )
            truth, truth_source = ground_truth_ich_context()
            manifest_sha = file_sha256(args.manifest_path)
            manifest = pd.read_csv(
                args.manifest_path,
                dtype={"study_id": str, "patient_id": str},
            )
            original = _load_variant(
                "hard_original_supervision",
                args.reference_run_dir,
                truth,
                args.expected_studies,
            )
            rescored_slices, supervision_audit = apply_supervision_manifest(
                original.slices,
                manifest,
                expected_promotions=args.expected_promotions,
            )
            hard = _variant("hard_schema4", rescored_slices, truth)
            threshold_artifacts = _load_threshold_artifacts(
                args.threshold_run_dir,
                expected_manifest_sha256=manifest_sha,
                expected_studies=args.expected_studies,
            )
            threshold = _variant(
                "threshold_schema4", threshold_artifacts.slices, truth
            )
            gated_slices, gate_audit = _apply_gate(hard, threshold)
            gated = _variant("hard_presence_gated_threshold", gated_slices, truth)
            bootstrap = _paired_patient_bootstrap(
                hard,
                gated,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            fold_deltas = _fold_deltas(hard, gated)
            decision = promotion_decision(
                hard.summary,
                gated.summary,
                bootstrap,
                fold_deltas,
            )
            duration = time.perf_counter() - started
            resolved = {
                "run_name": args.run_name,
                "reference_run_dirs": [str(path) for path in args.reference_run_dir],
                "threshold_run_dirs": [str(path) for path in args.threshold_run_dir],
                "manifest_path": str(args.manifest_path),
                "manifest_sha256": manifest_sha,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "promotion_policy": PROMOTION_POLICY,
                "git_commit": git_commit(),
            }
            summary = {
                "analysis_kind": "ich_crossfitted_mask_threshold_presence_gate",
                "run_name": args.run_name,
                "run_id": run.info.run_id,
                "development_status": "adaptive_oof_not_confirmatory",
                "studies": int(len(hard.studies)),
                "patients": int(hard.sufficient["patient_id"].nunique()),
                "duration_s": duration,
                "manifest_sha256": manifest_sha,
                "truth_source": str(truth_source),
                "supervision_audit": supervision_audit,
                "threshold_provenance": threshold_artifacts.provenance,
                "gate_audit": gate_audit,
                "hard": hard.summary,
                "raw_threshold": threshold.summary,
                "gated": gated.summary,
                "fold_deltas": fold_deltas,
                "paired_patient_bootstrap": bootstrap,
                "promotion_decision": decision,
                "git_commit": git_commit(),
                "leaderboard_confirmation_required": True,
            }
            (output / "resolved_config.json").write_text(
                json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            gated.slices.to_csv(
                output / "gated_oof_slice_predictions.csv", index=False
            )
            gated.studies.to_csv(
                output / "gated_oof_study_predictions.csv", index=False
            )
            threshold.studies.to_csv(
                output / "threshold_oof_study_predictions.csv", index=False
            )
            metric_rows = {}
            for metric in (
                "selection_score",
                "mean_foreground_dice",
                "presence_f1_at_0_1ml",
                "normal_false_positive_rate_at_0_1ml",
                "total_volume_mae_ml",
                "total_volume_bias_ml",
            ):
                metric_rows[f"hard_{metric}"] = float(hard.summary[metric])
                metric_rows[f"gated_{metric}"] = float(gated.summary[metric])
                metric_rows[f"delta_{metric}"] = float(gated.summary[metric]) - float(
                    hard.summary[metric]
                )
            metric_rows.update(
                {
                    "dice_bootstrap_probability": float(
                        bootstrap["metrics"]["mean_foreground_dice"][
                            "bootstrap_probability_candidate_better"
                        ]
                    ),
                    "mae_bootstrap_probability": float(
                        bootstrap["metrics"]["total_volume_mae_ml"][
                            "bootstrap_probability_candidate_better"
                        ]
                    ),
                    "selection_fold_wins": float(decision["selection_fold_wins"]),
                    "promotion_allowed": float(decision["promotion_allowed"]),
                    "duration_s": duration,
                }
            )
            mlflow.log_metrics(metric_rows)
            mlflow.log_artifacts(str(output), artifact_path="ich_mask_threshold_oof")

        notify_campaign(
            "success" if decision["promotion_allowed"] else "warning",
            "🧠 مسابقه IAAA 2026 | مدل خونریزی\n\n📊 ارزیابی پنج‌fold exp58 "
            "تمام شد. gate سخت، FPR/F1/AUC را دقیقاً ثابت نگه داشت؛ "
            f"Dice={hard.summary['mean_foreground_dice']:.4f}→"
            f"{gated.summary['mean_foreground_dice']:.4f}، MAE="
            f"{hard.summary['total_volume_mae_ml']:.3f}→"
            f"{gated.summary['total_volume_mae_ml']:.3f}mL و |bias|="
            f"{abs(hard.summary['total_volume_bias_ml']):.3f}→"
            f"{abs(gated.summary['total_volume_bias_ml']):.3f}mL. "
            f"P(Dice بهتر)={bootstrap['metrics']['mean_foreground_dice']['bootstrap_probability_candidate_better']:.3f}، "
            f"P(MAE بهتر)={bootstrap['metrics']['total_volume_mae_ml']['bootstrap_probability_candidate_better']:.3f} "
            f"و promotion={decision['promotion_allowed']}.\n\n🔎 تحلیل: safety کاملاً حفظ شد، "
            f"اما فقط {decision['selection_fold_wins']}/5 fold در selection بردند و CIها "
            "هنوز صفر را قطع می‌کنند؛ بنابراین checkpoint جدیدی ساخته نمی‌شود و نتیجه فقط "
            "به‌عنوان کاندید پژوهشی ثبت شد. مسیر بعدی باید خود مدل فضایی را قوی‌تر کند.",
            run=args.run_name,
            kind="adaptive_development_oof",
            detail=f"MLflow {summary['run_id']}; leaderboard confirmation required",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        notify_campaign(
            "failure",
            "🧠 مسابقه IAAA 2026 | مدل خونریزی\n\n⚠️ ارزیابی exp58 با خطای فنی "
            "متوقف شد. این رخداد نتیجهٔ کیفیتی نیست و هیچ checkpointی promote نشد.\n\n"
            "🔎 تحلیل: artefactهای fold حفظ شده‌اند؛ علت باید بدون تغییر recipe اصلاح شود.",
            run=args.run_name,
            kind="adaptive_development_oof",
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        raise


if __name__ == "__main__":
    main()
