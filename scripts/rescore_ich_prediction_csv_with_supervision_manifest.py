"""Re-score one frozen ICH prediction CSV under a corrected supervision manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.rescore_ich_oof_with_supervision_manifest import (
    _metric_delta,
    apply_supervision_manifest,
)
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import ground_truth_ich_context


INVARIANT_METRICS = (
    "any_ich_study_auc",
    "macro_subtype_study_auc",
    "presence_f1_at_0_1ml",
    "normal_false_positive_rate_at_0_1ml",
    "total_volume_mae_ml",
    "total_volume_bias_ml",
)


def validate_prediction_invariants(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    """Ensure the supervision-only rescore did not alter model outputs."""
    changed = {
        metric: (before[metric], after[metric])
        for metric in INVARIANT_METRICS
        if not np.isclose(
            float(before[metric]), float(after[metric]), rtol=0.0, atol=1e-12
        )
    }
    if changed:
        raise ValueError(f"Prediction-derived invariant metrics changed: {changed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int)
    parser.add_argument("--expected-promotions", type=int)
    args = parser.parse_args()

    truth, metadata_source = ground_truth_ich_context()
    predictions = pd.read_csv(
        args.prediction_csv,
        dtype={"study_id": str, "patient_id": str},
    )
    manifest = pd.read_csv(
        args.cache_manifest,
        dtype={"study_id": str, "patient_id": str},
    )
    original_studies, original_summary = summarize_segmentation_predictions(
        predictions, truth
    )
    if args.expected_studies is not None and len(original_studies) != args.expected_studies:
        raise ValueError(
            f"Expected {args.expected_studies} studies, got {len(original_studies)}"
        )
    rescored_slices, audit = apply_supervision_manifest(
        predictions,
        manifest,
        expected_promotions=args.expected_promotions,
    )
    rescored_studies, rescored_summary = summarize_segmentation_predictions(
        rescored_slices, truth
    )
    validate_prediction_invariants(original_summary, rescored_summary)

    payload = {
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "analysis_kind": "frozen_prediction_csv_supervision_rescore",
        "metadata_source": str(metadata_source),
        "prediction_csv": str(args.prediction_csv),
        "cache_manifest": str(args.cache_manifest),
        "audit": audit,
        "original_summary": original_summary,
        "rescored_summary": rescored_summary,
        "rescored_minus_original": _metric_delta(
            original_summary, rescored_summary
        ),
        "invariant_metrics_verified": list(INVARIANT_METRICS),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rescored_slices.to_csv(
        args.output_dir / "rescored_slice_predictions.csv", index=False
    )
    rescored_studies.to_csv(
        args.output_dir / "rescored_study_predictions.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
