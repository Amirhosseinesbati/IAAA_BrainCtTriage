"""Re-score frozen ICH OOF predictions under a corrected supervision manifest.

The model outputs are never changed.  For newly-known strict clean-negative
slices, the existing foreground prediction is added to the known-pixel Dice
denominator while the observed foreground and intersection remain zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.compare_ich_2p5d_segmentation_oof import _load_variant
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import ground_truth_ich_context


KEY_COLUMNS = ["study_id", "slice_index"]
TARGET_COLUMNS = ["any_ich", *OUTPUT_LABELS[1:]]


def apply_supervision_manifest(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    expected_promotions: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand known-pixel accounting using a validated newer manifest."""
    prediction_required = {"study_id", "slice_index", "known"}
    for label in OUTPUT_LABELS[1:]:
        prediction_required.update({
            f"pred_pixels_{label}",
            f"intersection_{label}",
            f"predicted_known_pixels_{label}",
            f"observed_known_pixels_{label}",
        })
    manifest_required = {
        "study_id",
        "slice_index",
        "segmentation_known",
        "classification_known",
        "supervision_type",
        *TARGET_COLUMNS,
    }
    missing_predictions = prediction_required - set(predictions)
    missing_manifest = manifest_required - set(manifest)
    if missing_predictions:
        raise ValueError(
            f"Predictions are missing columns: {sorted(missing_predictions)}"
        )
    if missing_manifest:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_manifest)}")

    frame = predictions.copy()
    supervision = manifest.loc[
        :, [*KEY_COLUMNS, "segmentation_known", "classification_known",
            "supervision_type", *TARGET_COLUMNS]
    ].copy()
    for table in (frame, supervision):
        table["study_id"] = table["study_id"].astype(str)
        table["slice_index"] = table["slice_index"].astype(int)
    if supervision.duplicated(KEY_COLUMNS).any():
        raise ValueError("Supervision manifest contains duplicate study/slice keys")

    before_rows = len(frame)
    frame = frame.merge(
        supervision,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if len(frame) != before_rows or frame["segmentation_known"].isna().any():
        raise ValueError("Predictions and supervision manifest do not match one-to-one")

    old_known = frame["known"].astype(int)
    new_known = frame["segmentation_known"].astype(int)
    if not old_known.isin([0, 1]).all() or not new_known.isin([0, 1]).all():
        raise ValueError("Known flags must be binary")
    demoted = (old_known == 1) & (new_known == 0)
    if demoted.any():
        raise ValueError(
            f"Corrected manifest unexpectedly demotes {int(demoted.sum())} slices"
        )
    promoted = (old_known == 0) & (new_known == 1)
    promoted_count = int(promoted.sum())
    if expected_promotions is not None and promoted_count != expected_promotions:
        raise ValueError(
            f"Expected {expected_promotions} promoted slices, got {promoted_count}"
        )
    if promoted.any():
        promoted_rows = frame.loc[promoted]
        invalid_type = promoted_rows["supervision_type"].ne("clean_negative")
        if invalid_type.any():
            raise ValueError("Only strict clean-negative slices may be promoted")
        if promoted_rows[TARGET_COLUMNS].astype(int).to_numpy().any():
            raise ValueError("A promoted clean-negative slice has a positive target")
        if promoted_rows["classification_known"].astype(int).ne(1).any():
            raise ValueError("Promoted spatial supervision is not classification-known")

    for label in OUTPUT_LABELS[1:]:
        zero_columns = [
            f"intersection_{label}",
            f"predicted_known_pixels_{label}",
            f"observed_known_pixels_{label}",
        ]
        if promoted.any() and frame.loc[promoted, zero_columns].to_numpy().any():
            raise ValueError(
                f"Previously-unknown {label} sufficient statistics are not zero"
            )
        frame.loc[promoted, f"intersection_{label}"] = 0
        frame.loc[promoted, f"observed_known_pixels_{label}"] = 0
        frame.loc[promoted, f"predicted_known_pixels_{label}"] = frame.loc[
            promoted, f"pred_pixels_{label}"
        ].astype(int)
    frame["known"] = new_known

    audit: dict[str, Any] = {
        "prediction_slices": int(len(frame)),
        "old_known_slices": int(old_known.sum()),
        "new_known_slices": int(new_known.sum()),
        "promoted_slices": promoted_count,
        "demoted_slices": int(demoted.sum()),
        "promoted_studies": int(frame.loc[promoted, "study_id"].nunique()),
        "promotion_policy": "strict_clean_negative_only",
        "model_predictions_changed": False,
    }
    if "outer_fold" in frame:
        audit["promoted_slices_by_outer_fold"] = {
            str(int(fold)): int(count)
            for fold, count in frame.loc[promoted].groupby("outer_fold").size().items()
        }
    return frame.drop(
        columns=["segmentation_known", "classification_known", "supervision_type",
                 *TARGET_COLUMNS]
    ), audit


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    metrics = (
        "selection_score",
        "mean_foreground_dice",
        "any_ich_study_auc",
        "macro_subtype_study_auc",
        "presence_f1_at_0_1ml",
        "normal_false_positive_rate_at_0_1ml",
        "total_volume_mae_ml",
        "total_volume_bias_ml",
    )
    return {
        metric: float(after[metric]) - float(before[metric])
        for metric in metrics
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument("--expected-promotions", type=int, default=145)
    args = parser.parse_args()

    truth, metadata_source = ground_truth_ich_context()
    original = _load_variant(
        "frozen_incumbent_original_supervision",
        args.run_dir,
        truth,
        args.expected_studies,
    )
    manifest = pd.read_csv(
        args.cache_manifest,
        dtype={"study_id": str, "patient_id": str},
    )
    rescored_slices, audit = apply_supervision_manifest(
        original.slices,
        manifest,
        expected_promotions=args.expected_promotions,
    )
    rescored_studies, rescored_summary = summarize_segmentation_predictions(
        rescored_slices, truth
    )
    fold_summaries: list[dict[str, Any]] = []
    for outer_fold, fold_frame in rescored_slices.groupby("outer_fold", sort=True):
        _, fold_summary = summarize_segmentation_predictions(fold_frame, truth)
        fold_summaries.append({"outer_fold": int(outer_fold), **fold_summary})

    payload = {
        "evaluation_scope": "standalone_ich_patient_disjoint_oof",
        "analysis_kind": "frozen_predictions_supervision_rescore",
        "metadata_source": str(metadata_source),
        "cache_manifest": str(args.cache_manifest),
        "audit": audit,
        "original_summary": original.summary,
        "rescored_summary": rescored_summary,
        "rescored_minus_original": _metric_delta(
            original.summary, rescored_summary
        ),
        "rescored_folds": fold_summaries,
        "interpretation": (
            "Only known-pixel spatial accounting changed. Classification scores, "
            "predicted volumes, FPR, F1, and volume errors must remain invariant."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rescored_slices.to_csv(
        args.output_dir / "rescored_oof_slice_predictions.csv", index=False
    )
    rescored_studies.to_csv(
        args.output_dir / "rescored_oof_study_predictions.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
