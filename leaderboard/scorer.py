"""
scorer.py — Scoring utilities for the personal leaderboard.

Computes Macro-F1 — the metric stated in the official challenge guide — plus
QWK and supplementary diagnostics. QWK remains available for continuity with
older local reports but is not treated as the primary selection criterion.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Versioned output — preserves previous results by appending _1, _2, ...
# ---------------------------------------------------------------------------
def versioned_output_path(path: Path) -> Path:
    """Return the next available version of *path* without overwriting.

    If *path* does not exist it is returned as-is.
    Otherwise ``_1``, ``_2``, … is inserted before the extension.
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    # If the stem already ends with _N, start from N+1 so we don't
    # produce results_1, results_2_1, etc.
    m = re.search(r"_(\d+)$", stem)
    start = int(m.group(1)) + 1 if m else 1

    for version in range(start, start + 10_000):
        new_stem = re.sub(r"_\d+$", f"_{version}", stem) if m else f"{stem}_{version}"
        candidate = parent / f"{new_stem}{suffix}"
        if not candidate.exists():
            return candidate

    # Fallback — should never happen with 10k attempts
    raise FileExistsError(f"Cannot find free version number for {path}")


def versioned_output_pair(csv_path: Path, json_path: Path) -> Tuple[Path, Path]:
    """Version both CSV and JSON output paths with the *same* version number."""
    if not csv_path.exists() and not json_path.exists():
        return csv_path, json_path

    # Derive a candidate stem from the CSV path
    csv_stem = csv_path.stem
    csv_suffix = csv_path.suffix
    json_stem = json_path.stem
    json_suffix = json_path.suffix
    parent = csv_path.parent

    m = re.search(r"_(\d+)$", csv_stem)
    start = int(m.group(1)) + 1 if m else 1

    for version in range(start, start + 10_000):
        new_csv_stem = re.sub(r"_\d+$", f"_{version}", csv_stem) if m else f"{csv_stem}_{version}"
        new_json_stem = re.sub(r"_\d+$", f"_{version}", json_stem) if m else f"{json_stem}_{version}"
        candidate_csv = parent / f"{new_csv_stem}{csv_suffix}"
        candidate_json = json_path.parent / f"{new_json_stem}{json_suffix}"
        if not candidate_csv.exists() and not candidate_json.exists():
            return candidate_csv, candidate_json

    raise FileExistsError(f"Cannot find free version for {{{csv_path}, {json_path}}}")

TRIAGE_LABELS = {
    0: "Normal    (0)",
    1: "Emergency (1)",
    2: "Critical  (2)",
}


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa.

    Args:
        y_true: Ground truth triage labels (0, 1, 2).
        y_pred: Predicted triage labels (0, 1, 2).

    Returns:
        QWK score in range [-1, 1]. Higher = better agreement than chance.
        - 1.0 = perfect agreement
        - 0.0 = chance-level agreement
        - < 0 = worse than chance
    """
    return float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )


def compute_metrics(
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intermediates: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Compute full suite of evaluation metrics.

    Args:
        study_ids: List of study identifiers (same order as y_true/y_pred).
        y_true: Ground truth triage labels.
        y_pred: Predicted triage labels.
        intermediates: Optional list of intermediate dicts (7 values) per study.

    Returns:
        Metrics dictionary suitable for JSON serialization.
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No samples to evaluate."}

    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
    qwk = compute_qwk(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Per-class metrics via classification_report
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=["Normal", "Emergency", "Critical"],
        output_dict=True,
        zero_division=0,
    )

    # Count distribution
    gt_dist = {int(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))}
    pred_dist = {int(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))}

    # Fill missing classes with 0
    for c in [0, 1, 2]:
        gt_dist.setdefault(c, 0)
        pred_dist.setdefault(c, 0)

    metrics = {
        "n_samples": n,
        "official_metric": "macro_f1",
        "macro_f1": round(macro_f1, 6),
        "qwk": round(qwk, 6),
        "accuracy": round(acc, 6),
        "confusion_matrix": cm.tolist(),
        "ground_truth_distribution": gt_dist,
        "prediction_distribution": pred_dist,
        "per_class": {
            str(k): {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1_score": round(v["f1-score"], 4),
                "support": int(v["support"]),
            }
            for k, v in report.items()
            if k in ["Normal", "Emergency", "Critical"]
        },
    }

    return metrics


def print_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics to console.

    Args:
        metrics: Dict returned by :func:`compute_metrics`.
    """
    if "error" in metrics:
        print(f"\n  ⚠️  {metrics['error']}")
        return

    qwk = metrics["qwk"]
    acc = metrics["accuracy"]
    n = metrics["n_samples"]

    # ---- Color-code QWK ---------------------------------------------------
    if qwk >= 0.81:
        qwk_color = "\033[92m"  # green
    elif qwk >= 0.61:
        qwk_color = "\033[93m"  # yellow
    else:
        qwk_color = "\033[91m"  # red
    reset = "\033[0m"

    # ---- Header ------------------------------------------------------------
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🏆  PERSONAL LEADERBOARD — IAAA 2026 Brain CT Triage".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Studies evaluated: {n:<38}║")
    print(f"║  QWK Score:  {qwk_color}{qwk:>8.4f}{reset}                            ║")
    print(f"║  Accuracy:   {acc:>8.4f}  ({acc*100:.1f}%)                   ║")
    print("╠" + "═" * 58 + "╣")

    # ---- QWK Interpretation ------------------------------------------------
    if qwk >= 0.81:
        interpretation = "Almost perfect agreement"
    elif qwk >= 0.61:
        interpretation = "Substantial agreement"
    elif qwk >= 0.41:
        interpretation = "Moderate agreement"
    elif qwk >= 0.21:
        interpretation = "Fair agreement"
    elif qwk >= 0.01:
        interpretation = "Slight agreement"
    else:
        interpretation = "Poor / worse than chance"
    print(f"║  Interpretation: {interpretation:<40}║")
    print("╠" + "═" * 58 + "╣")

    # ---- Confusion Matrix --------------------------------------------------
    cm = np.array(metrics["confusion_matrix"])
    print("║  Confusion Matrix (rows=GT, cols=Pred):                         ║")
    print(f"║           Pred N   Pred E   Pred C                              ║")
    labels_short = ["GT Normal ", "GT Emerg  ", "GT Critical"]
    for i, label in enumerate(labels_short):
        print(
            f"║  {label}  {cm[i][0]:>6}   {cm[i][1]:>6}   {cm[i][2]:>6}                           ║"
        )
    print("╠" + "═" * 58 + "╣")

    # ---- Per-class metrics -------------------------------------------------
    print("║  Per-Class Metrics:                                              ║")
    print("║           Precision  Recall    F1       Support                  ║")
    class_names = ["Normal", "Emergency", "Critical"]
    for cls_name in class_names:
        stats = metrics["per_class"].get(cls_name, {})
        if stats:
            print(
                f"║  {cls_name:<8}  "
                f"{stats['precision']:.4f}    {stats['recall']:.4f}    "
                f"{stats['f1_score']:.4f}    {stats['support']:>5}                    ║"
            )
    print("╠" + "═" * 58 + "╣")

    # ---- Distribution ------------------------------------------------------
    gt = metrics["ground_truth_distribution"]
    pr = metrics["prediction_distribution"]
    print("║  Class Distribution (GT → Pred):                                 ║")
    for cls_int in [0, 1, 2]:
        cls_name = ["Normal", "Emergency", "Critical"][cls_int]
        print(
            f"║    {cls_name:<8}: {gt.get(cls_int, 0):>4} → {pr.get(cls_int, 0):>4}"
            + " " * 24 + "║"
        )
    print("╚" + "═" * 58 + "╝")
    print()

    # ---- Verbose per-class details ------------------------------------------
    for cls_name in class_names:
        stats = metrics["per_class"].get(cls_name, {})
        if stats:
            logger.info(
                "%s — Precision: %.4f | Recall: %.4f | F1: %.4f | Support: %d",
                cls_name,
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                stats["support"],
            )


def export_results_csv(
    output_path: str | Path,
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: Optional[List[str]] = None,
    ground_truth_info: Optional[Dict[str, Dict[str, Any]]] = None,
    intermediates_list: Optional[List[Dict[str, float]]] = None,
) -> Path:
    """Export per-study predictions and ground truth to CSV.

    Args:
        output_path: Destination CSV path.
        study_ids: Study identifiers.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        patient_ids: Optional patient identifiers.
        ground_truth_info: Optional full ground truth dict (from load_study_labels).
        intermediates_list: Optional list of model intermediates per study.

    Returns:
        Path to the saved CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, study_id in enumerate(study_ids):
        row = {
            "study_id": study_id,
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "correct": int(y_true[i] == y_pred[i]),
        }
        if patient_ids:
            row["patient_id"] = patient_ids[i]
        elif ground_truth_info and study_id in ground_truth_info:
            row["patient_id"] = ground_truth_info[study_id].get("patient_id", "")

        # Append ground truth details if available
        if ground_truth_info and study_id in ground_truth_info:
            gt = ground_truth_info[study_id]
            for key in ["MLS_mm", "SkullFracture", "total_volume_ml"]:
                row[f"gt_{key}"] = gt.get(key, "")

        # Append model intermediates if available
        if intermediates_list and i < len(intermediates_list):
            inter = intermediates_list[i]
            for key in ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
                        "fracture_prob", "MLS_mm"]:
                row[f"pred_{key}"] = inter.get(key, "")

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Per-study results exported to %s (%d rows)", output_path, len(df))
    return output_path


def export_metrics_json(metrics: Dict[str, Any], output_path: str | Path) -> Path:
    """Export metrics dict as JSON.

    Args:
        metrics: Dict from :func:`compute_metrics`.
        output_path: Destination JSON path.

    Returns:
        Path to the saved JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Metrics exported to %s", output_path)
    return output_path
