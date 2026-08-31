"""Evaluate a deployable incumbent + detector-snapshot fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.evaluate_fracture_mil_deployable_blend import _empirical_cdf
from scripts.evaluate_fracture_mil_oof import _interval, _macro_paired_bootstrap
from scripts.train_fracture_smooth_attention_mil import _study_bags
from src.fracture.pooling import aggregate_study_scores


IDENTITY_COLUMNS = [
    "study_id",
    "patient_id",
    "study_fracture",
    "outer_fold",
    "slice_index",
]


def _load_snapshot_scores(
    epoch10_cache: Path,
    epoch15_cache: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    first = pd.read_csv(
        epoch10_cache / "slices.csv", dtype={"study_id": str, "patient_id": str}
    )
    second = pd.read_csv(
        epoch15_cache / "slices.csv", dtype={"study_id": str, "patient_id": str}
    )
    missing = set(IDENTITY_COLUMNS).difference(first.columns).union(
        set(IDENTITY_COLUMNS).difference(second.columns)
    )
    if missing:
        raise ValueError(f"Cache slice tables are missing columns: {sorted(missing)}")
    if not first[IDENTITY_COLUMNS].equals(second[IDENTITY_COLUMNS]):
        raise RuntimeError("Epoch10/epoch15 cache slice identities differ")
    first_scores = np.load(
        epoch10_cache / "slice_scores.npy", allow_pickle=False
    ).astype(np.float64)
    second_scores = np.load(
        epoch15_cache / "slice_scores.npy", allow_pickle=False
    ).astype(np.float64)
    if first_scores.shape != second_scores.shape or first_scores.shape != (len(first),):
        raise RuntimeError("Snapshot score arrays do not match slice tables")
    return first, 0.5 * (first_scores + second_scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch10-cache-root", type=Path, required=True)
    parser.add_argument("--epoch15-cache-root", type=Path, required=True)
    parser.add_argument("--incumbent-predictions", type=Path, required=True)
    parser.add_argument("--screening-predictions", type=Path, required=True)
    parser.add_argument("--fixed-weight", type=float, default=0.4)
    parser.add_argument("--parity-tolerance", type=float, default=0.0005)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.fixed_weight <= 1.0:
        raise ValueError("fixed-weight must be in [0, 1]")
    incumbent = pd.read_csv(
        args.incumbent_predictions, dtype={"study_id": str, "patient_id": str}
    )
    screening = pd.read_csv(args.screening_predictions, dtype={"study_id": str})
    if "fold" in screening and "outer_fold" not in screening:
        screening = screening.rename(columns={"fold": "outer_fold"})
    output_rows: list[pd.DataFrame] = []
    calibration_manifest: list[dict[str, object]] = []
    parity_by_fold: dict[str, float] = {}
    for fold in range(5):
        slices, snapshot_slice_scores = _load_snapshot_scores(
            args.epoch10_cache_root / f"fold_{fold}",
            args.epoch15_cache_root / f"fold_{fold}",
        )
        bags = _study_bags(slices)
        study = pd.DataFrame(
            {
                "study_id": [bag.study_id for bag in bags],
                "patient_id": [bag.patient_id for bag in bags],
                "truth": [bag.truth for bag in bags],
                "outer_fold": [bag.outer_fold for bag in bags],
                "snapshot_raw_score": [
                    aggregate_study_scores(snapshot_slice_scores[bag.indices])[
                        "top5_mean"
                    ]
                    for bag in bags
                ],
            }
        )
        training = study["outer_fold"].ne(fold)
        validation = study["outer_fold"].eq(fold)
        if int(validation.sum()) == 0:
            raise ValueError(f"Fold {fold} validation set is empty")
        validation_frame = study.loc[validation].copy()
        validation_frame["snapshot_train_cdf"] = _empirical_cdf(
            study.loc[training, "snapshot_raw_score"].to_numpy(float),
            validation_frame["snapshot_raw_score"].to_numpy(float),
        )
        incumbent_fold = incumbent.loc[
            incumbent["outer_fold"].eq(fold),
            ["study_id", "truth", "outer_fold", "deployable_blend_score"],
        ]
        validation_frame = validation_frame.merge(
            incumbent_fold,
            on=["study_id", "truth", "outer_fold"],
            validate="one_to_one",
        )
        screening_fold = screening.loc[
            screening["outer_fold"].eq(fold),
            ["study_id", "truth", "outer_fold", "prob_final_snapshot_ensemble"],
        ]
        parity = validation_frame.merge(
            screening_fold,
            on=["study_id", "truth", "outer_fold"],
            validate="one_to_one",
        )
        maximum_difference = float(
            np.max(
                np.abs(
                    parity["snapshot_raw_score"].to_numpy(float)
                    - parity["prob_final_snapshot_ensemble"].to_numpy(float)
                )
            )
        )
        if maximum_difference > args.parity_tolerance:
            raise RuntimeError(
                f"Fold {fold} snapshot parity {maximum_difference} exceeds "
                f"{args.parity_tolerance}"
            )
        parity_by_fold[str(fold)] = maximum_difference
        validation_frame["deployable_snapshot_fusion_score"] = (
            (1.0 - args.fixed_weight)
            * validation_frame["deployable_blend_score"].to_numpy(float)
            + args.fixed_weight
            * validation_frame["snapshot_train_cdf"].to_numpy(float)
        )
        output_rows.append(validation_frame)
        calibration_manifest.append(
            {
                "fold": fold,
                "n_training_studies": int(training.sum()),
                "snapshot_training_scores": study.loc[
                    training, "snapshot_raw_score"
                ].tolist(),
            }
        )

    predictions = pd.concat(output_rows, ignore_index=True)
    per_fold: list[dict[str, float | int]] = []
    for fold, frame in predictions.groupby("outer_fold", sort=True):
        reference_auc = float(
            roc_auc_score(frame["truth"], frame["deployable_blend_score"])
        )
        fusion_auc = float(
            roc_auc_score(
                frame["truth"], frame["deployable_snapshot_fusion_score"]
            )
        )
        per_fold.append(
            {
                "fold": int(fold),
                "reference_auc": reference_auc,
                "fusion_auc": fusion_auc,
                "difference": fusion_auc - reference_auc,
            }
        )
    reference_bootstrap, fusion_bootstrap = _macro_paired_bootstrap(
        predictions,
        "deployable_blend_score",
        "deployable_snapshot_fusion_score",
        iterations=args.iterations,
        seed=args.seed,
    )
    difference = fusion_bootstrap - reference_bootstrap
    reference_values = np.asarray([row["reference_auc"] for row in per_fold])
    fusion_values = np.asarray([row["fusion_auc"] for row in per_fold])
    payload = {
        "method": "epoch10_epoch15_slice_average_top5_train_cdf_fusion",
        "fixed_weight": args.fixed_weight,
        "per_fold": per_fold,
        "reference_macro_auc": float(reference_values.mean()),
        "fusion_macro_auc": float(fusion_values.mean()),
        "macro_difference": float(fusion_values.mean() - reference_values.mean()),
        "reference_worst_fold_auc": float(reference_values.min()),
        "fusion_worst_fold_auc": float(fusion_values.min()),
        "snapshot_parity_maximum_absolute_difference_by_fold": parity_by_fold,
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            "difference_95": _interval(difference),
            "probability_fusion_not_better": float(np.mean(difference <= 0.0)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "private_oof_predictions.csv", index=False)
    (args.output / "private_calibration_manifest.json").write_text(
        json.dumps(calibration_manifest, indent=2) + "\n", encoding="utf-8"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
