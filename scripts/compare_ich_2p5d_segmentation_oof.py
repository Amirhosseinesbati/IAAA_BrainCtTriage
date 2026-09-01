"""Compare two five-fold ICH segmentation variants on patient-disjoint OOF data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.strategies.ich_2p5d.cache import OUTPUT_LABELS
from src.strategies.ich_2p5d.segmentation_evaluation import (
    VOLUME_TO_LABEL,
    summarize_segmentation_predictions,
)
from src.strategies.ich_v2.evaluation import VOLUME_KEYS, ground_truth_ich_context


EXPECTED_FOLDS = set(range(5))
HIGHER_IS_BETTER = {
    "selection_score": True,
    "mean_foreground_dice": True,
    "any_ich_study_auc": True,
    "macro_subtype_study_auc": True,
    "presence_f1_at_0_1ml": True,
    "normal_false_positive_rate_at_0_1ml": False,
    "total_volume_mae_ml": False,
}


@dataclass(frozen=True)
class VariantResult:
    name: str
    slices: pd.DataFrame
    studies: pd.DataFrame
    sufficient: pd.DataFrame
    summary: dict[str, Any]
    fold_summaries: list[dict[str, Any]]
    runs: list[dict[str, Any]]


@dataclass(frozen=True)
class RunArtifacts:
    outer_fold: int
    predictions_path: Path
    provenance: dict[str, Any]


def _load_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Resolve either a trained fold or a locked channel-hybrid fold."""
    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary = json.loads(
            (run_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        return RunArtifacts(
            outer_fold=int(config["outer_fold"]),
            predictions_path=run_dir / "outer_slice_predictions.csv",
            provenance={
                "artifact_kind": "trained_fold",
                "run_id": summary.get("run_id"),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "best_epoch": summary.get("best_epoch"),
                "segmentation_class_weight_basis": config.get(
                    "segmentation_class_weight_basis", "slice"
                ),
            },
        )

    hybrid_path = run_dir / "hybrid_summary.json"
    if not hybrid_path.exists():
        raise FileNotFoundError(
            f"{run_dir} contains neither resolved_config.json nor hybrid_summary.json"
        )
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
    if hybrid.get("evaluation_split") != "outer_fold":
        raise ValueError(f"{run_dir}: hybrid is not an outer-fold evaluation")
    if hybrid.get("outer_fold") is None:
        raise ValueError(f"{run_dir}: hybrid outer_fold provenance is missing")
    return RunArtifacts(
        outer_fold=int(hybrid["outer_fold"]),
        predictions_path=run_dir / "hybrid_slice_predictions.csv",
        provenance={
            "artifact_kind": "locked_channel_hybrid",
            "run_id": None,
            "checkpoint_sha256": None,
            "best_epoch": None,
            "segmentation_class_weight_basis": "channel_hybrid",
            "reference_labels": hybrid.get("reference_labels"),
            "candidate_labels": hybrid.get("candidate_labels"),
            "any_ich_source": hybrid.get("any_ich_source"),
            "reference": hybrid.get("reference"),
            "candidate": hybrid.get("candidate"),
        },
    )


def _safe_weighted_auc(
    truth: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    active = weights > 0
    if len(np.unique(truth[active])) < 2:
        return None
    return float(roc_auc_score(truth, score, sample_weight=weights))


def _study_sufficient_statistics(
    slices: pd.DataFrame,
    studies: pd.DataFrame,
) -> pd.DataFrame:
    aggregation: dict[str, str] = {
        "patient_id": "first",
        "outer_fold": "first",
    }
    for label in OUTPUT_LABELS[1:]:
        aggregation[f"intersection_{label}"] = "sum"
        aggregation[f"predicted_known_pixels_{label}"] = "sum"
        aggregation[f"observed_known_pixels_{label}"] = "sum"
    sufficient = (
        slices.groupby("study_id", as_index=False)
        .agg(aggregation)
        .merge(studies, on="study_id", how="inner", validate="one_to_one")
    )
    if len(sufficient) != slices["study_id"].nunique():
        raise ValueError("Study sufficient statistics do not match the OOF slices")
    return sufficient.sort_values("study_id").reset_index(drop=True)


def _metric_vector(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(frame) or float(weights.sum()) <= 0:
        raise ValueError("Bootstrap weights must cover at least one study")

    subtype_aucs: list[float] = []
    dice_values: list[float] = []
    for volume_key, label in VOLUME_TO_LABEL.items():
        intersection = float(np.sum(
            frame[f"intersection_{label}"].to_numpy(float) * weights
        ))
        predicted = float(np.sum(
            frame[f"predicted_known_pixels_{label}"].to_numpy(float) * weights
        ))
        observed = float(np.sum(
            frame[f"observed_known_pixels_{label}"].to_numpy(float) * weights
        ))
        if observed > 0:
            dice_values.append((2.0 * intersection) / max(1.0, predicted + observed))
        truth = (frame[f"gt_{volume_key}"].to_numpy(float) > 0).astype(np.uint8)
        auc = _safe_weighted_auc(
            truth,
            frame[f"score_{label}"].to_numpy(float),
            weights,
        )
        if auc is not None:
            subtype_aucs.append(auc)

    gt_total = frame[[f"gt_{key}" for key in VOLUME_KEYS]].sum(axis=1).to_numpy(float)
    pred_total = frame[[f"pred_{key}" for key in VOLUME_KEYS]].sum(axis=1).to_numpy(float)
    gt_any = gt_total > 0
    pred_any = pred_total >= 0.1
    any_auc = _safe_weighted_auc(
        gt_any.astype(np.uint8),
        frame["score_any_ich"].to_numpy(float),
        weights,
    )
    if any_auc is None:
        raise ValueError("A bootstrap sample contains only one Any-ICH class")
    mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
    macro_auc = float(np.mean(subtype_aucs)) if subtype_aucs else 0.0
    normal = ~gt_any
    normal_weight = float(weights[normal].sum())
    fpr = (
        float(np.sum(weights[normal] * pred_any[normal]) / normal_weight)
        if normal_weight > 0
        else 0.0
    )
    return {
        "selection_score": 0.55 * mean_dice + 0.30 * any_auc + 0.15 * macro_auc,
        "mean_foreground_dice": mean_dice,
        "any_ich_study_auc": any_auc,
        "macro_subtype_study_auc": macro_auc,
        "presence_f1_at_0_1ml": float(
            f1_score(gt_any, pred_any, sample_weight=weights, zero_division=0)
        ),
        "normal_false_positive_rate_at_0_1ml": fpr,
        "total_volume_mae_ml": float(
            np.average(np.abs(pred_total - gt_total), weights=weights)
        ),
    }


def _load_variant(
    name: str,
    run_dirs: list[Path],
    truth: pd.DataFrame,
    expected_studies: int,
) -> VariantResult:
    if len(run_dirs) != 5:
        raise ValueError(f"{name}: exactly five run directories are required")
    seen_folds: set[int] = set()
    fold_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        artifacts = _load_run_artifacts(run_dir)
        outer_fold = artifacts.outer_fold
        if outer_fold in seen_folds:
            raise ValueError(f"{name}: duplicate outer fold {outer_fold}")
        seen_folds.add(outer_fold)
        frame = pd.read_csv(
            artifacts.predictions_path,
            dtype={"study_id": str, "patient_id": str},
        ).assign(outer_fold=outer_fold)
        fold_studies, fold_summary = summarize_segmentation_predictions(frame, truth)
        fold_summary = {"outer_fold": outer_fold, **fold_summary}
        fold_summaries.append(fold_summary)
        fold_frames.append(frame)
        runs.append({
            "outer_fold": outer_fold,
            "run_dir": str(run_dir),
            "studies": int(len(fold_studies)),
            **artifacts.provenance,
        })
    if seen_folds != EXPECTED_FOLDS:
        raise ValueError(f"{name}: expected folds 0..4, got {sorted(seen_folds)}")

    slices = pd.concat(fold_frames, ignore_index=True)
    study_fold_counts = slices.groupby("study_id")["outer_fold"].nunique()
    if int(study_fold_counts.max()) != 1:
        raise ValueError(f"{name}: at least one study occurs in multiple outer folds")
    patient_fold_counts = slices.groupby("patient_id")["outer_fold"].nunique()
    if int(patient_fold_counts.max()) != 1:
        raise ValueError(f"{name}: at least one patient occurs in multiple outer folds")
    if slices["study_id"].nunique() != expected_studies:
        raise ValueError(
            f"{name}: expected {expected_studies} studies, got "
            f"{slices['study_id'].nunique()}"
        )
    studies, summary = summarize_segmentation_predictions(slices, truth)
    summary = {
        **summary,
        "patients": int(slices["patient_id"].nunique()),
        "outer_folds": sorted(seen_folds),
        "patient_disjoint_outer_folds": True,
    }
    sufficient = _study_sufficient_statistics(slices, studies)
    return VariantResult(
        name=name,
        slices=slices,
        studies=studies,
        sufficient=sufficient,
        summary=summary,
        fold_summaries=sorted(fold_summaries, key=lambda row: int(row["outer_fold"])),
        runs=sorted(runs, key=lambda row: int(row["outer_fold"])),
    )


def _paired_patient_bootstrap(
    reference: VariantResult,
    candidate: VariantResult,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    reference_frame = reference.sufficient.sort_values("study_id").reset_index(drop=True)
    candidate_frame = candidate.sufficient.sort_values("study_id").reset_index(drop=True)
    if not reference_frame["study_id"].equals(candidate_frame["study_id"]):
        raise ValueError("Reference and candidate OOF study identifiers do not match")
    if not reference_frame["patient_id"].equals(candidate_frame["patient_id"]):
        raise ValueError("Reference and candidate patient identifiers do not match")

    patient_ids = reference_frame["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patient_ids)
    patient_to_index = {patient: index for index, patient in enumerate(unique_patients)}
    study_patient_indices = np.asarray(
        [patient_to_index[patient] for patient in patient_ids], dtype=np.int64
    )
    unit_weights = np.ones(len(reference_frame), dtype=np.float64)
    reference_point = _metric_vector(reference_frame, unit_weights)
    candidate_point = _metric_vector(candidate_frame, unit_weights)

    rng = np.random.default_rng(seed)
    deltas = {metric: [] for metric in HIGHER_IS_BETTER}
    accepted = 0
    for _ in range(samples):
        sampled = rng.integers(0, len(unique_patients), size=len(unique_patients))
        patient_weights = np.bincount(sampled, minlength=len(unique_patients)).astype(float)
        weights = patient_weights[study_patient_indices]
        try:
            reference_metrics = _metric_vector(reference_frame, weights)
            candidate_metrics = _metric_vector(candidate_frame, weights)
        except ValueError:
            continue
        accepted += 1
        for metric in deltas:
            deltas[metric].append(candidate_metrics[metric] - reference_metrics[metric])
    if accepted < max(1, int(samples * 0.95)):
        raise ValueError(f"Only {accepted}/{samples} bootstrap samples were valid")

    result: dict[str, Any] = {
        "resampling_unit": "patient",
        "requested_samples": samples,
        "accepted_samples": accepted,
        "seed": seed,
        "metrics": {},
    }
    for metric, higher_is_better in HIGHER_IS_BETTER.items():
        values = np.asarray(deltas[metric], dtype=np.float64)
        if higher_is_better:
            probability = float(np.mean(values > 0) + 0.5 * np.mean(values == 0))
        else:
            probability = float(np.mean(values < 0) + 0.5 * np.mean(values == 0))
        result["metrics"][metric] = {
            "reference": reference_point[metric],
            "candidate": candidate_point[metric],
            "candidate_minus_reference": candidate_point[metric] - reference_point[metric],
            "delta_ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "higher_is_better": higher_is_better,
            "bootstrap_probability_candidate_better": probability,
        }
    return result


def _write_variant(output_dir: Path, variant: VariantResult) -> None:
    destination = output_dir / variant.name
    destination.mkdir(parents=True, exist_ok=True)
    variant.studies.to_csv(destination / "oof_study_predictions.csv", index=False)
    (destination / "summary.json").write_text(
        json.dumps({
            "summary": variant.summary,
            "folds": variant.fold_summaries,
            "runs": variant.runs,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--candidate-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-studies", type=int, default=338)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    truth, metadata_source = ground_truth_ich_context()
    reference = _load_variant(
        args.reference_name,
        args.reference_run_dir,
        truth,
        args.expected_studies,
    )
    candidate = _load_variant(
        args.candidate_name,
        args.candidate_run_dir,
        truth,
        args.expected_studies,
    )
    comparison = _paired_patient_bootstrap(
        reference,
        candidate,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = {
        "evaluation_scope": "ich_only_no_mls_no_fracture_no_triage",
        "metadata_source": str(metadata_source),
        "reference": reference.summary,
        "candidate": candidate.summary,
        "paired_patient_bootstrap": comparison,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_variant(args.output_dir, reference)
    _write_variant(args.output_dir, candidate)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
