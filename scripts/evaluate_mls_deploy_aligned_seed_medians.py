"""Canonical triage comparison of baseline and candidate three-seed OOF medians.

Only saved CUDA predictions are consumed here. The exact frozen Champion ICH
and fracture branches are evaluated alongside an oracle branch context. Raw
study rows stay private; the aggregate JSON contains every promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import load_fold_manifest
from src.evaluation.triage import triage_from_intermediates
from src.config import FOLD_MANIFEST_PATH


VOLUME_KEYS = ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH")
LABELS = (0, 1, 2)
LABEL_NAMES = ("Normal", "Urgent", "Critical")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _parse_fold_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("fold input must be FOLD=CSV")
    raw_fold, raw_path = value.split("=", 1)
    fold = int(raw_fold)
    if fold < 0:
        raise argparse.ArgumentTypeError(f"unsupported fold: {fold}")
    return fold, Path(raw_path).expanduser()


def _load_oof(
    inputs: list[tuple[int, Path]],
    prefix: str,
    fold_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected_folds = {fold for fold, _ in inputs}
    if len(inputs) != len(selected_folds) or not selected_folds:
        raise ValueError(f"{prefix} requires one unique file per selected fold")
    available_folds = set(fold_manifest["fold"].astype(int).unique())
    if not selected_folds.issubset(available_folds):
        raise ValueError(
            f"{prefix} requested unavailable folds: {sorted(selected_folds - available_folds)}"
        )
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for fold, path in sorted(inputs):
        path = path.resolve()
        frame = pd.read_csv(path, dtype={"study_id": str, "patient_id": str})
        required = {"study_id", "patient_id", "triage_class", "gt_MLS_mm", "median_MLS_mm"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"{path} missing columns: {missing}")
        expected = fold_manifest.loc[
            fold_manifest["fold"].astype(int) == fold,
            ["study_id", "patient_id", "triage_class"],
        ].copy()
        expected["study_id"] = expected["study_id"].astype(str)
        expected["patient_id"] = expected["patient_id"].astype(str)
        frame["study_id"] = frame["study_id"].astype(str)
        frame["patient_id"] = frame["patient_id"].astype(str)
        if len(frame) != len(expected) or frame["study_id"].duplicated().any():
            raise ValueError(f"Invalid held-out fold file: {path}")
        if set(frame["study_id"]) != set(expected["study_id"]):
            raise ValueError(f"Held-out fold membership differs from immutable manifest: {path}")
        contract = frame[["study_id", "patient_id", "triage_class"]].merge(
            expected,
            on="study_id",
            suffixes=("_actual", "_expected"),
            validate="one_to_one",
        )
        if not (
            contract["patient_id_actual"].eq(contract["patient_id_expected"]).all()
            and contract["triage_class_actual"].astype(int).eq(
                contract["triage_class_expected"].astype(int)
            ).all()
        ):
            raise ValueError(f"Patient or class contract differs from fold manifest: {path}")
        frame = frame[[
            "study_id", "patient_id", "triage_class", "gt_MLS_mm", "median_MLS_mm",
        ]].copy()
        frame["fold"] = fold
        frame = frame.rename(columns={"median_MLS_mm": f"{prefix}_MLS_mm"})
        frames.append(frame)
        sources.append({
            "fold": fold,
            "path": str(path),
            "sha256": _sha256(path),
            "studies": len(frame),
        })
    output = pd.concat(frames, ignore_index=True)
    expected_total = int(
        fold_manifest["fold"].astype(int).isin(selected_folds).sum()
    )
    if len(output) != expected_total or output["study_id"].duplicated().any():
        raise ValueError(
            f"{prefix} OOF contract expected {expected_total} unique studies"
        )
    return output, sources


def _classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=LABELS, zero_division=0,
    )
    matrix = confusion_matrix(truth, prediction, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(LABEL_NAMES)
        },
        "catastrophic_errors": {
            "normal_to_critical": int(matrix[0, 2]),
            "critical_to_normal": int(matrix[2, 0]),
        },
    }


def _threshold_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        f"f1_{threshold}mm": float(f1_score(
            truth >= threshold, prediction >= threshold, zero_division=0,
        ))
        for threshold in (1, 3, 5)
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    truth_column: str,
    baseline_column: str,
    candidate_column: str,
    *,
    samples: int = 10_000,
    seed: int = 20260902,
) -> dict[str, float]:
    patients = frame["patient_id"].drop_duplicates().to_numpy(str)
    grouped = {
        patient: frame.index[frame["patient_id"].astype(str) == patient].to_numpy(int)
        for patient in patients
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for iteration in range(samples):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([grouped[patient] for patient in sampled])
        truth = frame.loc[indices, truth_column].to_numpy(int)
        baseline = frame.loc[indices, baseline_column].to_numpy(int)
        candidate = frame.loc[indices, candidate_column].to_numpy(int)
        deltas[iteration] = (
            f1_score(truth, candidate, labels=LABELS, average="macro", zero_division=0)
            - f1_score(truth, baseline, labels=LABELS, average="macro", zero_division=0)
        )
    return {
        "samples": samples,
        "seed": seed,
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_of_improvement": float(np.mean(deltas > 0.0)),
    }


def _triage(frame: pd.DataFrame, prefix: str, mls_column: str) -> np.ndarray:
    predictions: list[int] = []
    for row in frame.itertuples(index=False):
        values = {key: float(getattr(row, f"{prefix}_{key}")) for key in VOLUME_KEYS}
        values["fracture_prob"] = float(getattr(row, f"{prefix}_fracture_prob"))
        values["MLS_mm"] = float(getattr(row, mls_column))
        predictions.append(int(triage_from_intermediates(values)))
    return np.asarray(predictions, dtype=int)


def _evaluate_context(frame: pd.DataFrame, prefix: str, name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    truth = frame["triage_class"].to_numpy(int)
    baseline = _triage(frame, prefix, "baseline_MLS_mm")
    candidate = _triage(frame, prefix, "candidate_MLS_mm")
    local = frame[["fold", "study_id", "patient_id", "triage_class"]].copy()
    local[f"{name}_baseline_triage"] = baseline
    local[f"{name}_candidate_triage"] = candidate
    bootstrap_frame = local.rename(columns={
        f"{name}_baseline_triage": "baseline",
        f"{name}_candidate_triage": "candidate",
    })
    baseline_metrics = _classification_metrics(truth, baseline)
    candidate_metrics = _classification_metrics(truth, candidate)
    per_fold: dict[str, Any] = {}
    for fold in sorted(int(value) for value in frame["fold"].unique()):
        selected = frame["fold"].to_numpy(int) == fold
        fold_baseline = _classification_metrics(truth[selected], baseline[selected])
        fold_candidate = _classification_metrics(truth[selected], candidate[selected])
        per_fold[str(fold)] = {
            "baseline": fold_baseline,
            "candidate": fold_candidate,
            "delta_macro_f1": fold_candidate["macro_f1"] - fold_baseline["macro_f1"],
        }
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "macro_f1": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "accuracy": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
            "urgent_f1": (
                candidate_metrics["per_class"]["Urgent"]["f1"]
                - baseline_metrics["per_class"]["Urgent"]["f1"]
            ),
        },
        "paired_patient_bootstrap": _cluster_bootstrap(
            bootstrap_frame, "triage_class", "baseline", "candidate",
        ),
        "per_fold": per_fold,
        "changed_triage_decisions": int(np.sum(baseline != candidate)),
    }, local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-fold", action="append", type=_parse_fold_path, required=True)
    parser.add_argument("--candidate-fold", action="append", type=_parse_fold_path, required=True)
    parser.add_argument("--fold-manifest", type=Path, default=None)
    parser.add_argument("--frozen-champion-predictions", type=Path, required=True)
    parser.add_argument(
        "--truth-table", type=Path,
        default=Path("reports/eda/deep/deep_series_table.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fold_manifest_path = Path(args.fold_manifest or FOLD_MANIFEST_PATH).resolve()
    fold_manifest = load_fold_manifest(fold_manifest_path)
    available_folds = sorted(int(value) for value in fold_manifest["fold"].unique())
    selected_folds = sorted({fold for fold, _ in args.baseline_fold})
    if set(selected_folds) != {fold for fold, _ in args.candidate_fold}:
        raise ValueError("Baseline and candidate must cover the same folds")
    baseline, baseline_sources = _load_oof(
        args.baseline_fold, "baseline", fold_manifest,
    )
    candidate, candidate_sources = _load_oof(
        args.candidate_fold, "candidate", fold_manifest,
    )
    expected_studies = int(
        fold_manifest["fold"].astype(int).isin(selected_folds).sum()
    )
    keys = ["fold", "study_id", "patient_id", "triage_class", "gt_MLS_mm"]
    frame = baseline.merge(candidate, on=keys, validate="one_to_one")
    if len(frame) != expected_studies:
        raise ValueError(
            "Baseline/candidate OOF intersection does not match selected fold manifest"
        )

    frozen_path = args.frozen_champion_predictions.resolve()
    frozen = pd.read_csv(frozen_path, dtype={"study_id": str})
    required_frozen = {"study_id", "fracture_prob", *VOLUME_KEYS}
    if missing := sorted(required_frozen - set(frozen.columns)):
        raise ValueError(f"Frozen Champion predictions missing columns: {missing}")
    if frozen["study_id"].duplicated().any():
        raise ValueError("Frozen Champion predictions contain duplicate studies")
    selected_studies = set(frame["study_id"].astype(str))
    if not selected_studies.issubset(set(frozen["study_id"].astype(str))):
        raise ValueError("Frozen Champion predictions do not cover selected folds")
    frozen = frozen.loc[frozen["study_id"].astype(str).isin(selected_studies)].copy()
    frozen = frozen[["study_id", "fracture_prob", *VOLUME_KEYS]].rename(columns={
        key: f"frozen_{key}" for key in ("fracture_prob", *VOLUME_KEYS)
    })
    frame = frame.merge(frozen, on="study_id", validate="one_to_one")

    truth_path = args.truth_table.resolve()
    truth = pd.read_csv(truth_path, dtype={"dicom_series.id": str}).rename(
        columns={"dicom_series.id": "study_id", "fracture_prob": "oracle_fracture_prob"},
    )
    if truth["study_id"].duplicated().any():
        raise ValueError("Truth table contains duplicate studies")
    truth = truth.loc[truth["study_id"].astype(str).isin(selected_studies), [
        "study_id", "oracle_fracture_prob", *VOLUME_KEYS,
    ]].rename(
        columns={key: f"oracle_{key}" for key in VOLUME_KEYS},
    )
    frame = frame.merge(truth, on="study_id", validate="one_to_one")
    if len(frame) != expected_studies:
        raise ValueError("Branch contexts do not cover the selected-fold OOF")

    contexts: dict[str, Any] = {}
    raw = frame[keys + ["baseline_MLS_mm", "candidate_MLS_mm"]].copy()
    for prefix, name in (("frozen", "frozen_champion"), ("oracle", "oracle")):
        summary, local = _evaluate_context(frame, prefix, name)
        contexts[name] = summary
        raw = raw.merge(local, on=["fold", "study_id", "patient_id", "triage_class"])

    truth_mls = frame["gt_MLS_mm"].to_numpy(float)
    baseline_mls = frame["baseline_MLS_mm"].to_numpy(float)
    candidate_mls = frame["candidate_MLS_mm"].to_numpy(float)
    baseline_thresholds = _threshold_metrics(truth_mls, baseline_mls)
    candidate_thresholds = _threshold_metrics(truth_mls, candidate_mls)
    frozen_summary = contexts["frozen_champion"]
    oracle_summary = contexts["oracle"]
    frozen_base = frozen_summary["baseline"]
    frozen_candidate = frozen_summary["candidate"]
    performance_gates = {
        "macro_f1_improved": frozen_summary["delta"]["macro_f1"] > 0.0,
        "macro_f1_preferred_margin_plus_0p01": frozen_summary["delta"]["macro_f1"] >= 0.01,
        "accuracy_noninferior": frozen_candidate["accuracy"] >= frozen_base["accuracy"],
        "urgent_f1_noninferior": (
            frozen_candidate["per_class"]["Urgent"]["f1"]
            >= frozen_base["per_class"]["Urgent"]["f1"]
        ),
        "normal_f1_not_below_minus_0p01": (
            frozen_candidate["per_class"]["Normal"]["f1"]
            >= frozen_base["per_class"]["Normal"]["f1"] - 0.01
        ),
        "critical_f1_not_below_minus_0p01": (
            frozen_candidate["per_class"]["Critical"]["f1"]
            >= frozen_base["per_class"]["Critical"]["f1"] - 0.01
        ),
        "no_fold_macro_drop_below_minus_0p01": all(
            row["delta_macro_f1"] >= -0.01
            for row in frozen_summary["per_fold"].values()
        ),
        "bootstrap_probability_at_least_0p95": (
            frozen_summary["paired_patient_bootstrap"]["probability_of_improvement"] >= 0.95
        ),
        "f1_3mm_noninferior": candidate_thresholds["f1_3mm"] >= baseline_thresholds["f1_3mm"],
        "f1_5mm_noninferior": candidate_thresholds["f1_5mm"] >= baseline_thresholds["f1_5mm"],
        "normal_to_critical_not_worse": (
            frozen_candidate["catastrophic_errors"]["normal_to_critical"]
            <= frozen_base["catastrophic_errors"]["normal_to_critical"]
        ),
        "critical_to_normal_not_worse": (
            frozen_candidate["catastrophic_errors"]["critical_to_normal"]
            <= frozen_base["catastrophic_errors"]["critical_to_normal"]
        ),
        "oracle_and_frozen_macro_direction_consistent": (
            frozen_summary["delta"]["macro_f1"] >= 0.0
            and oracle_summary["delta"]["macro_f1"] >= 0.0
        ),
        "oracle_and_frozen_urgent_direction_consistent": (
            frozen_summary["delta"]["urgent_f1"] >= 0.0
            and oracle_summary["delta"]["urgent_f1"] >= 0.0
        ),
    }
    performance_hard_gate_names = [
        name for name in performance_gates
        if name != "macro_f1_preferred_margin_plus_0p01"
    ]
    full_fold_coverage = selected_folds == available_folds
    gates = {
        **performance_gates,
        "full_immutable_fold_coverage": full_fold_coverage,
    }
    hard_gate_names = [*performance_hard_gate_names, "full_immutable_fold_coverage"]
    development_gate_passed = all(
        performance_gates[name] for name in performance_hard_gate_names
    )
    payload = {
        "schema_version": 1,
        "protocol": "deploy_aligned_fixed_three_seed_median_canonical_triage",
        "evaluation_scope": "full_oof" if full_fold_coverage else "development_oof_subset",
        "development_gate_passed": development_gate_passed,
        "promotion_eligible": all(gates[name] for name in hard_gate_names),
        "selected_folds": selected_folds,
        "available_folds": available_folds,
        "full_fold_coverage": full_fold_coverage,
        "sources": {
            "baseline_folds": baseline_sources,
            "candidate_folds": candidate_sources,
            "frozen_champion_predictions": {
                "path": str(frozen_path), "sha256": _sha256(frozen_path), "studies": len(frozen),
            },
            "truth_table": {"path": str(truth_path), "sha256": _sha256(truth_path)},
            "fold_manifest": {
                "path": str(fold_manifest_path),
                "sha256": _sha256(fold_manifest_path),
                "studies": len(fold_manifest),
            },
        },
        "studies": len(frame),
        "threshold_metrics": {
            "baseline": baseline_thresholds,
            "candidate": candidate_thresholds,
            "delta": {
                key: candidate_thresholds[key] - baseline_thresholds[key]
                for key in baseline_thresholds
            },
        },
        "contexts": contexts,
        "promotion_gates": gates,
        "failed_hard_gates": [name for name in hard_gate_names if not gates[name]],
    }
    output_dir = args.output_dir.resolve()
    aggregate_path = output_dir / "aggregate_summary.json"
    private_path = output_dir / "per_study_private.csv"
    _atomic_text(aggregate_path, json.dumps(payload, indent=2) + "\n")
    _atomic_csv(raw, private_path)
    print(json.dumps({
        "promotion_eligible": payload["promotion_eligible"],
        "development_gate_passed": payload["development_gate_passed"],
        "evaluation_scope": payload["evaluation_scope"],
        "failed_hard_gates": payload["failed_hard_gates"],
        "frozen_delta": frozen_summary["delta"],
        "bootstrap": frozen_summary["paired_patient_bootstrap"],
    }))


if __name__ == "__main__":
    main()
