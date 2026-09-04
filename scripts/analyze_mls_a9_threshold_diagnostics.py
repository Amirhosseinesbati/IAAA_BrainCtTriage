"""Aggregate-only diagnostic for A9's threshold-specific resource-screen failure.

The two private 70-study prediction files remain on the server.  This script
aligns them internally, writes no study ID, MLS value, slice prediction, or
row-level result, and emits only aggregate counts/statistics for A10 design.
No model inference or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _metrics, _sha256


BASE = Path("/workspace/iaaa_artifacts/mls_deploy_aligned_20260902")
CANDIDATE = BASE / "a9_frozen_baseline_refiner_20260904/canonical_audit/candidate/study_predictions_private.json"
BASELINE = BASE / "reference_refinement_baseline_qualification_20260904/study_predictions_private.json"
OUT = BASE / "a9_threshold_diagnostic_20260905.json"
THRESHOLDS = (1.0, 3.0, 5.0)


def _finite_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite private prediction field: {key}")
    return values


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Private prediction file is not a nonempty row list: {path}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Private prediction row is not an object")
    required = {"study_id", "input_fingerprint", "MLS_mm", "gt_MLS_mm", "slice_predictions"}
    if any(required - set(row) for row in rows):
        raise ValueError("Private prediction schema changed")
    indexed = {str(row["study_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Private prediction study IDs are not unique")
    return indexed


def _f1_confusion(truth: np.ndarray, prediction: np.ndarray, threshold: float) -> dict[str, int | float]:
    actual = truth >= threshold
    predicted = prediction >= threshold
    tp = int(np.count_nonzero(actual & predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    fn = int(np.count_nonzero(actual & ~predicted))
    tn = int(np.count_nonzero(~actual & ~predicted))
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": float(0.0 if denominator == 0 else 2 * tp / denominator)}


def _threshold_transition(truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, threshold: float) -> dict[str, Any]:
    actual = truth >= threshold
    old = baseline >= threshold
    new = candidate >= threshold
    return {
        "threshold_mm": threshold,
        "baseline": _f1_confusion(truth, baseline, threshold),
        "candidate": _f1_confusion(truth, candidate, threshold),
        "delta_f1": _f1_confusion(truth, candidate, threshold)["f1"] - _f1_confusion(truth, baseline, threshold)["f1"],
        "classification_changed": int(np.count_nonzero(old != new)),
        "true_positive_gained": int(np.count_nonzero(actual & ~old & new)),
        "true_positive_lost": int(np.count_nonzero(actual & old & ~new)),
        "false_positive_introduced": int(np.count_nonzero(~actual & ~old & new)),
        "false_positive_removed": int(np.count_nonzero(~actual & old & ~new)),
        "baseline_prediction_within_1mm_of_threshold": int(np.count_nonzero(np.abs(baseline - threshold) <= 1.0)),
        "candidate_prediction_within_1mm_of_threshold": int(np.count_nonzero(np.abs(candidate - threshold) <= 1.0)),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "mean_abs": float(np.abs(values).mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "max_abs": float(np.abs(values).max()),
    }


def _truth_strata(truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    intervals: list[tuple[str, float, float | None]] = [
        ("lt_1mm", -math.inf, 1.0), ("1_to_lt_3mm", 1.0, 3.0),
        ("3_to_lt_5mm", 3.0, 5.0), ("gte_5mm", 5.0, None),
    ]
    result: dict[str, Any] = {}
    for name, lower, upper in intervals:
        mask = truth >= lower
        if upper is not None:
            mask &= truth < upper
        count = int(mask.sum())
        if not count:
            result[name] = {"studies": 0}
            continue
        result[name] = {
            "studies": count,
            "baseline_mae_mm": float(np.abs(baseline[mask] - truth[mask]).mean()),
            "candidate_mae_mm": float(np.abs(candidate[mask] - truth[mask]).mean()),
            "candidate_minus_baseline_mae_mm": float(
                np.abs(candidate[mask] - truth[mask]).mean() - np.abs(baseline[mask] - truth[mask]).mean()
            ),
            "mean_candidate_minus_baseline_mm": float((candidate[mask] - baseline[mask]).mean()),
        }
    return result


def run() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite A9 threshold diagnostic: {OUT}")
    baseline_rows, candidate_rows = _load(BASELINE), _load(CANDIDATE)
    if set(baseline_rows) != set(candidate_rows) or len(baseline_rows) != 70:
        raise ValueError("Baseline/candidate private coverage differs from the 70-study canonical screen")
    keys = sorted(baseline_rows)
    if any(baseline_rows[key]["input_fingerprint"] != candidate_rows[key]["input_fingerprint"] for key in keys):
        raise ValueError("Baseline/candidate input fingerprint differs")
    baseline_list = [baseline_rows[key] for key in keys]
    candidate_list = [candidate_rows[key] for key in keys]
    truth = _finite_array(baseline_list, "gt_MLS_mm")
    candidate_truth = _finite_array(candidate_list, "gt_MLS_mm")
    if not np.array_equal(truth, candidate_truth):
        raise ValueError("Baseline/candidate ground truth differs")
    baseline = _finite_array(baseline_list, "MLS_mm")
    candidate = _finite_array(candidate_list, "MLS_mm")
    correction = candidate - baseline
    transitions = {str(threshold): _threshold_transition(truth, baseline, candidate, threshold) for threshold in THRESHOLDS}
    result = {
        "status": "completed",
        "scope": "a9_vs_qualified_runtime_baseline_aggregate_threshold_diagnostic",
        "studies": int(len(keys)),
        "model_inference_performed": False,
        "model_training_performed": False,
        "private_rows_exported": False,
        "candidate_model_promotion": False,
        "input_alignment": {
            "study_coverage_equal": True,
            "input_fingerprints_equal": True,
            "ground_truth_equal": True,
            "baseline_private_sha256": _sha256(BASELINE),
            "candidate_private_sha256": _sha256(CANDIDATE),
        },
        "metrics": {
            "baseline": _metrics(truth, baseline),
            "candidate": _metrics(truth, candidate),
        },
        "threshold_transitions": transitions,
        "correction_mm": _summary(correction),
        "truth_strata": _truth_strata(truth, baseline, candidate),
        "interpretation_guard": "Aggregate diagnostics guide a pre-registered next hypothesis only; no threshold, pooling, or checkpoint selection may be tuned on this screen.",
        "source_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(OUT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run()
    print(json.dumps({
        "status": result["status"], "studies": result["studies"],
        "f1_3mm_delta": result["threshold_transitions"]["3.0"]["delta_f1"],
        "private_rows_exported": result["private_rows_exported"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
