"""Evaluate a patient-safe OOF hybrid of hard-pixel presence and reference masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compare_ich_2p5d_segmentation_oof import (
    VariantResult,
    _load_variant,
    _paired_patient_bootstrap,
    _study_sufficient_statistics,
    _write_variant,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context


ALIGNMENT_KEYS = ["study_id", "patient_id", "slice_index", "outer_fold"]
SPATIAL_PREFIXES = ("pred_pixels", "intersection", "predicted_known_pixels")


def build_presence_gated_reference(
    reference: VariantResult,
    candidate: VariantResult,
    truth: pd.DataFrame,
) -> VariantResult:
    """Keep reference masks only when candidate study volume reaches 0.1 mL."""
    reference_slices = reference.slices.sort_values(ALIGNMENT_KEYS).reset_index(drop=True)
    candidate_slices = candidate.slices.sort_values(ALIGNMENT_KEYS).reset_index(drop=True)
    if not reference_slices[ALIGNMENT_KEYS].equals(candidate_slices[ALIGNMENT_KEYS]):
        raise ValueError("Reference and candidate OOF slices are not exactly aligned")

    candidate_studies = candidate.studies.copy()
    candidate_studies["gate_positive"] = (
        candidate_studies[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1) >= 0.1
    )
    gate_by_study = candidate_studies.set_index("study_id")["gate_positive"]

    hybrid = reference_slices.copy()
    hybrid["prob_any_ich"] = candidate_slices["prob_any_ich"].to_numpy()
    for label in OUTPUT_LABELS[1:]:
        hybrid[f"prob_{label}"] = candidate_slices[f"prob_{label}"].to_numpy()

    gate = hybrid["study_id"].astype(str).map(gate_by_study).fillna(False).to_numpy(bool)
    for label in OUTPUT_LABELS[1:]:
        for prefix in SPATIAL_PREFIXES:
            column = f"{prefix}_{label}"
            hybrid.loc[~gate, column] = 0

    studies, summary = summarize_segmentation_predictions(hybrid, truth)
    summary = {
        **summary,
        "patients": int(hybrid["patient_id"].nunique()),
        "outer_folds": sorted(int(value) for value in hybrid["outer_fold"].unique()),
        "patient_disjoint_outer_folds": True,
        "gate": "candidate_total_volume_at_least_0.1ml",
        "spatial_source": reference.name,
        "score_source": candidate.name,
    }
    fold_summaries: list[dict[str, Any]] = []
    for outer_fold, fold_frame in hybrid.groupby("outer_fold", sort=True):
        _, fold_summary = summarize_segmentation_predictions(fold_frame, truth)
        fold_summaries.append({"outer_fold": int(outer_fold), **fold_summary})
    sufficient = _study_sufficient_statistics(hybrid, studies)
    return VariantResult(
        name="candidate_presence_gated_reference_spatial",
        slices=hybrid,
        studies=studies,
        sufficient=sufficient,
        summary=summary,
        fold_summaries=fold_summaries,
        runs=[
            {
                "kind": "deterministic_oof_hybrid",
                "gate": "candidate_total_volume_at_least_0.1ml",
                "reference_variant": reference.name,
                "candidate_variant": candidate.name,
            }
        ],
    )


def _confusion(studies: pd.DataFrame) -> dict[str, int]:
    gt_total = studies[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1).to_numpy(float)
    pred_total = studies[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1).to_numpy(float)
    truth = gt_total > 0
    predicted = pred_total >= 0.1
    return {
        "true_positive": int(np.sum(truth & predicted)),
        "false_positive": int(np.sum(~truth & predicted)),
        "true_negative": int(np.sum(~truth & ~predicted)),
        "false_negative": int(np.sum(truth & ~predicted)),
    }


def _diagnostics(
    reference: VariantResult,
    candidate: VariantResult,
    hybrid: VariantResult,
) -> pd.DataFrame:
    identity = (
        candidate.slices[["study_id", "patient_id", "outer_fold"]]
        .drop_duplicates()
        .astype({"study_id": str, "patient_id": str, "outer_fold": int})
    )
    frames: list[pd.DataFrame] = []
    for variant in (reference, candidate, hybrid):
        frame = variant.studies.copy()
        predicted_columns = [f"pred_{key}" for key in VOLUME_KEYS]
        truth_columns = [f"gt_{key}" for key in VOLUME_KEYS]
        frame[f"pred_total_{variant.name}"] = frame[predicted_columns].sum(axis=1)
        frame[f"abs_error_total_{variant.name}"] = (
            frame[predicted_columns].sum(axis=1) - frame[truth_columns].sum(axis=1)
        ).abs()
        frame[f"pred_any_{variant.name}"] = (
            frame[f"pred_total_{variant.name}"] >= 0.1
        )
        keep = [
            "study_id",
            f"pred_total_{variant.name}",
            f"abs_error_total_{variant.name}",
            f"pred_any_{variant.name}",
        ]
        if not frames:
            frame["gt_total"] = frame[truth_columns].sum(axis=1)
            frame["gt_any"] = frame["gt_total"] > 0
            keep.extend(["gt_total", "gt_any"])
        frames.append(frame[keep])
    result = identity
    for frame in frames:
        result = result.merge(frame, on="study_id", how="inner", validate="one_to_one")
    result["candidate_minus_reference_abs_error"] = (
        result[f"abs_error_total_{candidate.name}"]
        - result[f"abs_error_total_{reference.name}"]
    )
    result["hybrid_minus_candidate_abs_error"] = (
        result[f"abs_error_total_{hybrid.name}"]
        - result[f"abs_error_total_{candidate.name}"]
    )
    return result.sort_values(
        ["candidate_minus_reference_abs_error", "study_id"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--candidate-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--reference-name", default="pixel_baseline")
    parser.add_argument("--candidate-name", default="hardpixel_fprselect")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    truth, metadata_source = ground_truth_ich_context()
    reference = _load_variant(
        args.reference_name, args.reference_run_dir, truth, args.expected_studies
    )
    candidate = _load_variant(
        args.candidate_name, args.candidate_run_dir, truth, args.expected_studies
    )
    hybrid = build_presence_gated_reference(reference, candidate, truth)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_variant(args.output_dir, hybrid)
    diagnostics = _diagnostics(reference, candidate, hybrid)
    diagnostics.to_csv(args.output_dir / "paired_study_diagnostics.csv", index=False)
    payload = {
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "metadata_source": str(metadata_source),
        "reference": reference.summary,
        "candidate": candidate.summary,
        "hybrid": hybrid.summary,
        "confusion": {
            reference.name: _confusion(reference.studies),
            candidate.name: _confusion(candidate.studies),
            hybrid.name: _confusion(hybrid.studies),
        },
        "bootstrap_reference_vs_hybrid": _paired_patient_bootstrap(
            reference, hybrid, samples=args.bootstrap_samples, seed=args.seed
        ),
        "bootstrap_candidate_vs_hybrid": _paired_patient_bootstrap(
            candidate, hybrid, samples=args.bootstrap_samples, seed=args.seed + 1
        ),
    }
    (args.output_dir / "hybrid_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
