"""Audit patient-safe ICH fold shift for rare subtype interpretation.

The script is deliberately label-only: it never reads model predictions and
therefore cannot tune a model to an outer fold.  It quantifies whether unstable
IVH/SAH metrics can be explained by rare positives, lesion-volume shift, or
known spatial-supervision exclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.strategies.ich_v2.evaluation import (
    ground_truth_ich_context,
    load_slice_metadata,
)


SUBTYPE_COLUMNS = {
    "IVH": ("V_IVH", "IntraventricularHemorrhage"),
    "IPH": ("V_IPH", "IntraparenchymalHemorrhage"),
    "SDH": ("V_SDH", "SubduralHemorrhage"),
    "EDH": ("V_EDH", "EpiduralHemorrhage"),
    "SAH": ("V_SAH", "SubarachnoidHemorrhage"),
}
SMALL_VOLUME_THRESHOLDS_ML = (0.1, 0.5, 1.0, 2.0)
STUDY_BALANCE_POWERS = (0.0, 0.5, 0.75, 1.0)


def _finite_quantile(values: pd.Series, quantile: float) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, quantile)) if len(array) else None


def _positive_summary(
    frame: pd.DataFrame,
    *,
    subtype: str,
    volume_column: str,
    positive_slice_column: str,
) -> dict[str, Any]:
    positive = frame.loc[frame[volume_column] > 0].copy()
    other_volume_columns = [
        f"gt_{volume_key}"
        for label, (volume_key, _) in SUBTYPE_COLUMNS.items()
        if label != subtype
    ]
    result: dict[str, Any] = {
        "studies": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "positive_studies": int(len(positive)),
        "prevalence": float(len(positive) / len(frame)) if len(frame) else 0.0,
        "isolated_positive_studies": int(
            (
                (positive[other_volume_columns].sum(axis=1) <= 0)
                if len(positive)
                else pd.Series(dtype=bool)
            ).sum()
        ),
        "volume_ml_min": _finite_quantile(positive[volume_column], 0.0),
        "volume_ml_q25": _finite_quantile(positive[volume_column], 0.25),
        "volume_ml_median": _finite_quantile(positive[volume_column], 0.5),
        "volume_ml_q75": _finite_quantile(positive[volume_column], 0.75),
        "volume_ml_max": _finite_quantile(positive[volume_column], 1.0),
        "volume_ml_mean": (
            float(positive[volume_column].mean()) if len(positive) else None
        ),
        "positive_slices_min": _finite_quantile(
            positive[positive_slice_column], 0.0
        ),
        "positive_slices_median": _finite_quantile(
            positive[positive_slice_column], 0.5
        ),
        "positive_slices_max": _finite_quantile(
            positive[positive_slice_column], 1.0
        ),
    }
    for threshold in SMALL_VOLUME_THRESHOLDS_ML:
        key = str(threshold).replace(".", "p")
        result[f"positive_fraction_below_{key}ml"] = (
            float((positive[volume_column] < threshold).mean())
            if len(positive)
            else None
        )
    return result


def _bootstrap_fold2_median_delta(
    fold2: np.ndarray,
    other: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not len(fold2) or not len(other):
        return {
            "fold2_minus_other_median_ml": None,
            "bootstrap_ci95_ml": [None, None],
            "bootstrap_probability_fold2_larger": None,
        }
    point = float(np.median(fold2) - np.median(other))
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        fold2_sample = rng.choice(fold2, size=len(fold2), replace=True)
        other_sample = rng.choice(other, size=len(other), replace=True)
        deltas[index] = np.median(fold2_sample) - np.median(other_sample)
    return {
        "fold2_minus_other_median_ml": point,
        "bootstrap_ci95_ml": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "bootstrap_probability_fold2_larger": float(
            np.mean(deltas > 0) + 0.5 * np.mean(deltas == 0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-path", type=Path, default=Path("config/folds.csv"))
    parser.add_argument(
        "--slice-audit-path",
        type=Path,
        default=Path("tmp/ich_supervision_audit_dicom/slice_audit.csv"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--focus-outer-fold", type=int, default=2)
    parser.add_argument("--focus-calibration-fold", type=int, default=1)
    parser.add_argument(
        "--focus-subtype", choices=tuple(SUBTYPE_COLUMNS), default="IVH"
    )
    args = parser.parse_args()
    if args.expected_studies <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("expected-studies and bootstrap-samples must be positive")
    if args.focus_outer_fold == args.focus_calibration_fold:
        raise ValueError("focus outer and calibration folds must differ")

    folds = pd.read_csv(
        args.folds_path,
        dtype={"study_id": str, "patient_id": str},
    )
    required_fold_columns = {"study_id", "patient_id", "fold"}
    missing_fold_columns = required_fold_columns - set(folds)
    if missing_fold_columns:
        raise ValueError(f"Fold table lacks columns: {sorted(missing_fold_columns)}")
    if folds["study_id"].duplicated().any():
        raise ValueError("Every study must occur exactly once in the fold table")
    if folds["study_id"].nunique() != args.expected_studies:
        raise ValueError(
            f"Expected {args.expected_studies} studies, got {folds['study_id'].nunique()}"
        )
    if set(folds["fold"].astype(int)) != set(range(5)):
        raise ValueError("ICH audit expects exactly folds 0..4")
    patient_fold_counts = folds.groupby("patient_id")["fold"].nunique()
    if int(patient_fold_counts.max()) != 1:
        raise ValueError("Patient leakage detected in fold assignments")

    truth, truth_source = ground_truth_ich_context()
    truth["study_id"] = truth["study_id"].astype(str)
    merged = folds[["study_id", "patient_id", "fold"]].merge(
        truth,
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != args.expected_studies:
        raise ValueError("Fold assignments and reconstructed ICH truth are misaligned")

    slice_metadata, slice_metadata_source = load_slice_metadata()
    slice_metadata = slice_metadata.copy()
    slice_metadata["study_id"] = slice_metadata["dicom_series.id"].astype(str)
    positive_slice_columns: dict[str, str] = {}
    for subtype, (_, metadata_column) in SUBTYPE_COLUMNS.items():
        positive_slice_column = f"positive_slices_{subtype}"
        positive_slice_columns[subtype] = positive_slice_column
        slice_metadata[metadata_column] = (
            slice_metadata[metadata_column].fillna(False).astype(bool)
        )
    positive_slices = (
        slice_metadata.groupby("study_id", as_index=False)
        .agg(
            **{
                positive_slice_columns[subtype]: (metadata_column, "sum")
                for subtype, (_, metadata_column) in SUBTYPE_COLUMNS.items()
            }
        )
    )
    merged = merged.merge(
        positive_slices,
        on="study_id",
        how="left",
        validate="one_to_one",
    )
    for column in positive_slice_columns.values():
        merged[column] = merged[column].fillna(0).astype(int)

    # Reconstruct the current subtype-aware sampler for the focus experiment.
    # It weights positive *slices*, so a study with many positive slices receives
    # more expected draws than a study with an equally important tiny lesion.
    slice_with_folds = slice_metadata.merge(
        folds[["study_id", "fold"]],
        on="study_id",
        how="inner",
        validate="many_to_one",
    )
    focus_training = slice_with_folds.loc[
        ~slice_with_folds["fold"].isin(
            [args.focus_outer_fold, args.focus_calibration_fold]
        )
    ].copy()
    metadata_subtype_columns = [
        metadata_column for _, metadata_column in SUBTYPE_COLUMNS.values()
    ]
    positive_matrix = focus_training[metadata_subtype_columns].to_numpy(
        dtype=np.float64
    )
    positive_counts = positive_matrix.sum(axis=0)
    if np.any(positive_counts <= 0):
        raise ValueError("Focus training split lacks at least one ICH subtype")
    rare_weights = np.clip(
        np.sqrt(float(positive_counts.max()) / positive_counts), 1.0, 5.0
    )
    current_sample_weight = np.ones(len(focus_training), dtype=np.float64)
    for row_index, row in enumerate(positive_matrix):
        active = rare_weights[row > 0]
        if len(active):
            current_sample_weight[row_index] = 2.0 + float(active.max())
    focus_training["current_sampler_weight"] = current_sample_weight
    positive_rows = positive_matrix.any(axis=1)
    current_positive_mass = float(current_sample_weight[positive_rows].sum())
    balance_strategy_columns: dict[str, str] = {"current": "current_sampler_weight"}
    for power in STUDY_BALANCE_POWERS[1:]:
        candidate_matrix = np.zeros_like(positive_matrix, dtype=np.float64)
        for subtype_index, metadata_column in enumerate(metadata_subtype_columns):
            study_positive_counts = focus_training.groupby("study_id")[
                metadata_column
            ].transform("sum").to_numpy(dtype=np.float64)
            active = positive_matrix[:, subtype_index] > 0
            candidate_matrix[active, subtype_index] = (
                2.0 + rare_weights[subtype_index]
            ) / np.power(study_positive_counts[active], power)
        raw_positive_weights = candidate_matrix.max(axis=1)
        normalizer = current_positive_mass / float(raw_positive_weights.sum())
        balanced = np.ones(len(focus_training), dtype=np.float64)
        balanced[positive_rows] = raw_positive_weights[positive_rows] * normalizer
        strategy = f"study_balance_p{str(power).replace('.', 'p')}"
        column = f"{strategy}_sampler_weight"
        focus_training[column] = balanced
        balance_strategy_columns[strategy] = column
    focus_metadata_column = SUBTYPE_COLUMNS[args.focus_subtype][1]
    focus_positive = focus_training.loc[
        focus_training[focus_metadata_column].astype(bool)
    ]
    focus_volume_column = f"gt_{SUBTYPE_COLUMNS[args.focus_subtype][0]}"
    exposure_frames: list[pd.DataFrame] = []
    exposure_by_size_frames: list[pd.DataFrame] = []
    exposure_strategy_summary: dict[str, Any] = {}
    for strategy, weight_column in balance_strategy_columns.items():
        study_exposure = (
            focus_positive.groupby("study_id", as_index=False)
            .agg(
                positive_slices=(focus_metadata_column, "sum"),
                positive_slice_weight_mass=(weight_column, "sum"),
            )
            .merge(
                merged[
                    [
                        "study_id",
                        "patient_id",
                        "fold",
                        focus_volume_column,
                    ]
                ],
                on="study_id",
                how="inner",
                validate="one_to_one",
            )
        )
        study_exposure["expected_positive_draws_per_epoch"] = (
            len(focus_training)
            * study_exposure["positive_slice_weight_mass"]
            / float(focus_training[weight_column].sum())
        )
        study_exposure["volume_bin"] = pd.cut(
            study_exposure[focus_volume_column],
            bins=[-np.inf, 2.0, 10.0, np.inf],
            labels=["small_le_2ml", "medium_2_to_10ml", "large_gt_10ml"],
        ).astype(str)
        study_exposure.insert(0, "strategy", strategy)
        total_focus_mass = float(
            study_exposure["positive_slice_weight_mass"].sum()
        )
        exposure_by_size = (
            study_exposure.groupby(
                ["strategy", "volume_bin"], as_index=False, observed=False
            )
            .agg(
                studies=("study_id", "nunique"),
                positive_slices=("positive_slices", "sum"),
                positive_slice_weight_mass=("positive_slice_weight_mass", "sum"),
                expected_positive_draws_per_epoch=(
                    "expected_positive_draws_per_epoch",
                    "sum",
                ),
                median_expected_draws_per_study=(
                    "expected_positive_draws_per_epoch",
                    "median",
                ),
                median_volume_ml=(focus_volume_column, "median"),
            )
        )
        exposure_by_size["positive_sampling_mass_fraction"] = (
            exposure_by_size["positive_slice_weight_mass"] / total_focus_mass
        )
        correlation = float(
            study_exposure[
                [focus_volume_column, "expected_positive_draws_per_epoch"]
            ].corr(method="spearman").iloc[0, 1]
        )
        exposure_strategy_summary[strategy] = {
            "spearman_volume_vs_expected_draws": correlation,
            "maximum_slice_weight": float(focus_training[weight_column].max()),
            "total_positive_weight_mass": float(
                focus_training.loc[positive_rows, weight_column].sum()
            ),
        }
        exposure_frames.append(study_exposure)
        exposure_by_size_frames.append(exposure_by_size)
    all_study_exposure = pd.concat(exposure_frames, ignore_index=True)
    all_exposure_by_size = pd.concat(exposure_by_size_frames, ignore_index=True)

    subtype_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_frame = merged.loc[merged["fold"] == fold]
        for subtype, (volume_key, _) in SUBTYPE_COLUMNS.items():
            summary = _positive_summary(
                fold_frame,
                subtype=subtype,
                volume_column=f"gt_{volume_key}",
                positive_slice_column=positive_slice_columns[subtype],
            )
            subtype_rows.append({"fold": fold, "subtype": subtype, **summary})
    subtype_distribution = pd.DataFrame(subtype_rows)

    audit = pd.read_csv(args.slice_audit_path, dtype={"study_id": str})
    required_audit_columns = {
        "study_id",
        "spatial_supervision_safe",
        "positive_without_spatial_mask",
    }
    missing_audit_columns = required_audit_columns - set(audit)
    if missing_audit_columns:
        raise ValueError(f"Slice audit lacks columns: {sorted(missing_audit_columns)}")
    audit = audit.merge(
        folds[["study_id", "fold"]],
        on="study_id",
        how="inner",
        validate="many_to_one",
    )
    if audit["study_id"].nunique() != args.expected_studies:
        raise ValueError("Slice audit does not cover every folded study")
    quality_rows = []
    for fold in range(5):
        fold_studies = merged.loc[merged["fold"] == fold]
        fold_audit = audit.loc[audit["fold"] == fold]
        gt_total = fold_studies[
            [f"gt_{volume_key}" for volume_key, _ in SUBTYPE_COLUMNS.values()]
        ].sum(axis=1)
        unsafe = fold_audit["spatial_supervision_safe"] <= 0
        mismatch = fold_audit["positive_without_spatial_mask"] > 0
        quality_rows.append(
            {
                "fold": fold,
                "studies": int(len(fold_studies)),
                "patients": int(fold_studies["patient_id"].nunique()),
                "normal_studies": int((gt_total <= 0).sum()),
                "ich_positive_studies": int((gt_total > 0).sum()),
                "audited_slices": int(len(fold_audit)),
                "unsafe_spatial_slices": int(unsafe.sum()),
                "unsafe_spatial_studies": int(
                    fold_audit.loc[unsafe, "study_id"].nunique()
                ),
                "positive_without_spatial_mask_slices": int(mismatch.sum()),
                "positive_without_spatial_mask_studies": int(
                    fold_audit.loc[mismatch, "study_id"].nunique()
                ),
            }
        )
    quality_distribution = pd.DataFrame(quality_rows)

    fold2_shift: dict[str, Any] = {}
    for subtype, (volume_key, _) in SUBTYPE_COLUMNS.items():
        volume_column = f"gt_{volume_key}"
        fold2 = merged.loc[
            (merged["fold"] == 2) & (merged[volume_column] > 0), volume_column
        ].to_numpy(dtype=np.float64)
        other = merged.loc[
            (merged["fold"] != 2) & (merged[volume_column] > 0), volume_column
        ].to_numpy(dtype=np.float64)
        fold2_shift[subtype] = {
            "fold2_positive_studies": int(len(fold2)),
            "other_positive_studies": int(len(other)),
            "fold2_median_ml": float(np.median(fold2)) if len(fold2) else None,
            "other_median_ml": float(np.median(other)) if len(other) else None,
            **_bootstrap_fold2_median_delta(
                fold2,
                other,
                samples=args.bootstrap_samples,
                seed=args.seed + list(SUBTYPE_COLUMNS).index(subtype),
            ),
        }

    payload = {
        "evaluation_scope": "ich_labels_only_no_model_predictions_no_outer_tuning",
        "truth_source": str(truth_source),
        "slice_metadata_source": str(slice_metadata_source),
        "folds_path": str(args.folds_path),
        "slice_audit_path": str(args.slice_audit_path),
        "studies": int(merged["study_id"].nunique()),
        "patients": int(merged["patient_id"].nunique()),
        "patient_disjoint_folds": True,
        "fold2_positive_volume_shift": fold2_shift,
        "focus_training_sampler_exposure": {
            "outer_fold": args.focus_outer_fold,
            "calibration_fold": args.focus_calibration_fold,
            "training_folds": sorted(
                int(value) for value in focus_training["fold"].unique()
            ),
            "subtype": args.focus_subtype,
            "training_slices": int(len(focus_training)),
            "positive_studies": int(len(study_exposure)),
            "positive_slices": int(len(focus_positive)),
            "strategies": exposure_strategy_summary,
            "interpretation": (
                "positive-slice-weighted sampler exposure before any proposed "
                "study balancing"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subtype_distribution.to_csv(
        args.output_dir / "fold_subtype_distribution.csv", index=False
    )
    quality_distribution.to_csv(
        args.output_dir / "fold_quality_distribution.csv", index=False
    )
    all_study_exposure.sort_values(
        ["strategy", focus_volume_column, "study_id"]
    ).to_csv(args.output_dir / "focus_training_study_exposure.csv", index=False)
    all_exposure_by_size.to_csv(
        args.output_dir / "focus_training_exposure_by_size.csv", index=False
    )
    (args.output_dir / "fold_shift_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
