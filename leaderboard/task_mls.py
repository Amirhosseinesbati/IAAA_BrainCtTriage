"""
task_mls.py — Task-specific leaderboard for midline shift (MLS) estimation.

Evaluates the MLS_mm prediction against ground truth MidlineShiftMM from
training_df.csv, independently of the triage pipeline.

Key metrics:
  - Regression: MAE, MSE, RMSE, R², Pearson / Spearman correlation
  - Clinical threshold classification:
      - Urgent threshold (>= 3 mm): accuracy, precision, recall, F1
      - Critical threshold (>= 5 mm): accuracy, precision, recall, F1
  - Bland-Altman analysis (bias, 95% limits of agreement)

Usage (CSV mode)::

    python -m leaderboard.task_mls
    python -m leaderboard.task_mls --input-csv path/to/results.csv

Usage (inference mode)::

    python -m leaderboard.task_mls --run-inference
    python -m leaderboard.task_mls --run-inference --device cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_project_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# Clinical thresholds (matches triage.py)
MLS_URGENT_THRESHOLD = 3.0   # mm
MLS_CRITICAL_THRESHOLD = 5.0  # mm


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth(csv_path: str | Path) -> Dict[str, float]:
    """Load MidlineShiftMM ground truth aggregated to study level (max).

    Args:
        csv_path: Path to training_df.csv.

    Returns:
        Dict mapping study_id → MLS_mm (float).
    """
    csv_path = Path(csv_path)
    logger.info("Loading MLS ground truth from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    series = df.groupby("dicom_series.id")["MidlineShiftMM"].max()
    result = {str(k): float(v) for k, v in series.items()}
    logger.info("Loaded %d MLS ground-truth values.", len(result))
    return result


# ---------------------------------------------------------------------------
# Prediction loading (CSV mode)
# ---------------------------------------------------------------------------
def load_predictions_from_csv(csv_path: str | Path) -> Dict[str, float]:
    """Load MLS predictions from a pre-computed results CSV.

    Args:
        csv_path: Path to results.csv.

    Returns:
        Dict mapping study_id → predicted MLS_mm.
    """
    csv_path = Path(csv_path)
    logger.info("Loading MLS predictions from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    result: Dict[str, float] = {}
    for _, row in df.iterrows():
        result[str(row["study_id"])] = float(row["pred_MLS_mm"])
    logger.info("Loaded %d MLS predictions.", len(result))
    return result


# ---------------------------------------------------------------------------
# Prediction loading (inference mode)
# ---------------------------------------------------------------------------
def discover_study_dirs(data_dir: Path) -> Dict[str, Path]:
    from leaderboard.evaluate import discover_study_dirs as _discover
    return _discover(data_dir)


def run_inference(
    study_dirs: Dict[str, Path],
    models_dir: Path,
    device: str = "cuda",
    ich_strategy: str = "nnunet",
) -> Dict[str, float]:
    """Run the full model pipeline and extract MLS_mm values.

    Args:
        study_dirs: Dict mapping study_id → directory path.
        models_dir: Path to submission/models/.
        device: 'cuda' or 'cpu'.
        ich_strategy: ICH segmentation strategy name.

    Returns:
        Dict mapping study_id → predicted MLS_mm.
    """
    _ensure_project_on_path()
    from submission.model import load_models, predict as model_predict

    logger.info(
        "Loading models (ich_strategy=%s, device=%s) ...", ich_strategy, device
    )
    models = load_models(str(models_dir), device=device, ich_strategy=ich_strategy)
    logger.info("Models loaded. Running inference on %d studies ...", len(study_dirs))

    predictions: Dict[str, float] = {}
    errors: List[str] = []

    try:
        from tqdm import tqdm
        progress = tqdm(sorted(study_dirs.items()), desc="MLS inference", unit="study")
    except ImportError:
        progress = sorted(study_dirs.items())

    for study_id, study_dir in progress:
        try:
            intermediates = model_predict(str(study_dir), models=models)
            predictions[study_id] = float(intermediates["MLS_mm"])
        except Exception as exc:
            errors.append(f"{study_id}: {exc}")

    if errors:
        logger.warning("%d study(s) failed during inference.", len(errors))
        for err in errors[:10]:
            logger.warning("  - %s", err)

    logger.info("Inference complete: %d/%d studies.", len(predictions), len(study_dirs))
    return predictions


# ---------------------------------------------------------------------------
# Bland-Altman analysis
# ---------------------------------------------------------------------------
def bland_altman_analysis(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute Bland-Altman statistics.

    Returns:
        Dict with 'bias', 'sd_of_differences', 'upper_loa', 'lower_loa',
        and 'loa_range'.
    """
    differences = y_pred - y_true
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    return {
        "bias": round(mean_diff, 4),
        "sd_of_differences": round(sd_diff, 4),
        "upper_loa_95": round(upper_loa, 4),
        "lower_loa_95": round(lower_loa, 4),
        "loa_range": round(upper_loa - lower_loa, 4),
    }


# ---------------------------------------------------------------------------
# Threshold classification helper
# ---------------------------------------------------------------------------
def _threshold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float, label: str
) -> Dict[str, Any]:
    """Compute binary classification metrics at a given MLS threshold.

    Args:
        y_true: Ground truth MLS values.
        y_pred: Predicted MLS values.
        threshold: MLS threshold in mm.
        label: Short label for the threshold (e.g., "urgent" or "critical").

    Returns:
        Metrics dict.
    """
    true_bin = (y_true >= threshold).astype(int)
    pred_bin = (y_pred >= threshold).astype(int)

    return {
        "threshold_mm": threshold,
        "label": label,
        "accuracy": round(float(accuracy_score(true_bin, pred_bin)), 6),
        "precision": round(float(precision_score(true_bin, pred_bin, zero_division=0)), 4),
        "recall": round(float(recall_score(true_bin, pred_bin, zero_division=0)), 4),
        "f1_score": round(float(f1_score(true_bin, pred_bin, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(true_bin, pred_bin, labels=[0, 1]).tolist(),
        "prevalence": float(np.mean(true_bin)),
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute full suite of MLS evaluation metrics.

    Args:
        study_ids: List of study identifiers.
        y_true: Ground truth MLS_mm values.
        y_pred: Predicted MLS_mm values.

    Returns:
        Metrics dictionary.
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No samples to evaluate."}

    # -- Regression metrics --------------------------------------------------
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Pearson / Spearman correlation
    if n >= 3:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    # -- Bland-Altman --------------------------------------------------------
    ba = bland_altman_analysis(y_true, y_pred)

    # -- Threshold classification --------------------------------------------
    thresh_urgent = _threshold_metrics(y_true, y_pred, MLS_URGENT_THRESHOLD, "urgent")
    thresh_critical = _threshold_metrics(y_true, y_pred, MLS_CRITICAL_THRESHOLD, "critical")

    # -- Error distribution --------------------------------------------------
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    metrics = {
        "task": "midline_shift",
        "n_samples": n,
        "regression": {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p_value": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p_value": round(float(spearman_p), 6),
        },
        "bland_altman": ba,
        "classification_at_thresholds": [thresh_urgent, thresh_critical],
        "error_distribution": {
            "min_error": round(float(np.min(errors)), 4),
            "max_error": round(float(np.max(errors)), 4),
            "mean_error": round(float(np.mean(errors)), 4),
            "std_error": round(float(np.std(errors, ddof=1)), 4),
            "median_abs_error": round(float(np.median(abs_errors)), 4),
            "percent_within_1mm": round(float(np.mean(abs_errors <= 1.0) * 100), 2),
            "percent_within_2mm": round(float(np.mean(abs_errors <= 2.0) * 100), 2),
            "percent_within_5mm": round(float(np.mean(abs_errors <= 5.0) * 100), 2),
        },
        "ground_truth_distribution": {
            "min": round(float(np.min(y_true)), 4),
            "max": round(float(np.max(y_true)), 4),
            "mean": round(float(np.mean(y_true)), 4),
            "median": round(float(np.median(y_true)), 4),
            "std": round(float(np.std(y_true, ddof=1)), 4),
        },
        "prediction_distribution": {
            "min": round(float(np.min(y_pred)), 4),
            "max": round(float(np.max(y_pred)), 4),
            "mean": round(float(np.mean(y_pred)), 4),
            "median": round(float(np.median(y_pred)), 4),
            "std": round(float(np.std(y_pred, ddof=1)), 4),
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print MLS evaluation metrics."""
    if "error" in metrics:
        print(f"\n  ⚠️  {metrics['error']}")
        return

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  📏  MIDLINE SHIFT — Task Leaderboard".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Studies evaluated: {metrics['n_samples']:<38}║")
    print("╠" + "═" * 58 + "╣")

    # Regression
    reg = metrics["regression"]
    print("║  Regression Metrics:                                           ║")
    print(f"║    MAE  : {reg['mae']:>8.4f} mm                                   ║")
    print(f"║    RMSE : {reg['rmse']:>8.4f} mm                                   ║")
    print(f"║    R²   : {reg['r2_score']:>8.4f}                                    ║")
    print(f"║    Pearson r : {reg['pearson_r']:>8.4f}  (p={reg['pearson_p_value']:.4f})            ║")
    print(f"║    Spearman  : {reg['spearman_r']:>8.4f}  (p={reg['spearman_p_value']:.4f})            ║")

    print("╠" + "═" * 58 + "╣")

    # Bland-Altman
    ba = metrics["bland_altman"]
    print("║  Bland-Altman Analysis:                                        ║")
    print(f"║    Bias (mean diff) : {ba['bias']:>8.4f} mm                          ║")
    print(f"║    SD of differences: {ba['sd_of_differences']:>8.4f} mm                          ║")
    print(f"║    95% LOA         : [{ba['lower_loa_95']:.2f}, {ba['upper_loa_95']:.2f}] mm              ║")

    print("╠" + "═" * 58 + "╣")

    # Threshold classification
    print("║  Classification at Clinical Thresholds:                        ║")
    for t in metrics["classification_at_thresholds"]:
        label = t["label"].upper()
        print(f"║  ── {label} (≥ {t['threshold_mm']} mm) ───────────────────────────────║")
        print(f"║    Accuracy : {t['accuracy']:.4f}  ({t['accuracy']*100:.1f}%)             ║")
        print(f"║    Precision: {t['precision']:.4f}                               ║")
        print(f"║    Recall   : {t['recall']:.4f}                               ║")
        print(f"║    F1-score : {t['f1_score']:.4f}                               ║")
        print(f"║    Prevalence: {t['prevalence']:.4f} ({t['prevalence']*100:.1f}%)            ║")

    print("╠" + "═" * 58 + "╣")

    # Error distribution
    ed = metrics["error_distribution"]
    print("║  Error Distribution:                                           ║")
    print(f"║    Mean error     : {ed['mean_error']:>8.4f} mm                          ║")
    print(f"║    Median |error| : {ed['median_abs_error']:>8.4f} mm                          ║")
    print(f"║    Within 1mm     : {ed['percent_within_1mm']:>6.2f}%                              ║")
    print(f"║    Within 2mm     : {ed['percent_within_2mm']:>6.2f}%                              ║")
    print(f"║    Within 5mm     : {ed['percent_within_5mm']:>6.2f}%                              ║")

    print("╠" + "═" * 58 + "╣")

    # Distribution comparison
    gtd = metrics["ground_truth_distribution"]
    pd_ = metrics["prediction_distribution"]
    print("║  Distribution (GT vs Pred):                                    ║")
    print(f"║    GT :  mean={gtd['mean']:.2f}  median={gtd['median']:.2f}  std={gtd['std']:.2f}              ║")
    print(f"║    Pred: mean={pd_['mean']:.2f}  median={pd_['median']:.2f}  std={pd_['std']:.2f}              ║")

    print("╚" + "═" * 58 + "╝")
    print()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_results_csv(
    output_path: str | Path,
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    """Export per-study MLS predictions to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "study_id": sid,
            "gt_MLS_mm": round(float(gt), 4),
            "pred_MLS_mm": round(float(pred), 4),
            "abs_error": round(abs(float(pred - gt)), 4),
            "urgent_gt": int(gt >= MLS_URGENT_THRESHOLD),
            "urgent_pred": int(pred >= MLS_URGENT_THRESHOLD),
            "critical_gt": int(gt >= MLS_CRITICAL_THRESHOLD),
            "critical_pred": int(pred >= MLS_CRITICAL_THRESHOLD),
        }
        for sid, gt, pred in zip(study_ids, y_true, y_pred)
    ]
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("MLS results exported to %s (%d rows)", output_path, len(df))
    return output_path


def export_metrics_json(metrics: Dict[str, Any], output_path: str | Path) -> Path:
    """Export metrics dict as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("MLS metrics exported to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------
def _align(
    gt: Dict[str, float], preds: Dict[str, float]
) -> tuple[List[str], np.ndarray, np.ndarray]:
    """Align ground truth and predictions on shared study IDs."""
    matched = sorted(set(gt.keys()) & set(preds.keys()))
    only_gt = set(gt.keys()) - set(preds.keys())
    only_pred = set(preds.keys()) - set(gt.keys())

    if only_gt:
        logger.warning("%d studies in GT but not predictions (skipped).", len(only_gt))
    if only_pred:
        logger.warning("%d studies in predictions but not GT (skipped).", len(only_pred))

    if not matched:
        raise ValueError("No matching studies between ground truth and predictions.")

    y_true = np.array([gt[sid] for sid in matched])
    y_pred = np.array([preds[sid] for sid in matched])
    return matched, y_true, y_pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    default_results = PROJECT_ROOT / "leaderboard" / "results.csv"
    default_gt_csv = PROJECT_ROOT / "Data" / "metadata" / "training_df.csv"
    default_models = PROJECT_ROOT / "submission" / "models"
    default_data = PROJECT_ROOT / "Data" / "raw" / "training"
    default_out_csv = PROJECT_ROOT / "leaderboard" / "mls_results.csv"
    default_out_json = PROJECT_ROOT / "leaderboard" / "mls_metrics.json"

    parser = argparse.ArgumentParser(
        description="📏 Midline Shift (MLS) Estimation — Task Leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leaderboard.task_mls                                         # from default results.csv
  python -m leaderboard.task_mls --input-csv results_monai.csv           # from custom CSV
  python -m leaderboard.task_mls --run-inference                         # run model pipeline
  python -m leaderboard.task_mls --run-inference --device cpu            # inference on CPU
        """,
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_results,
        help=f"Path to results CSV from evaluate.py (default: {default_results})",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=default_gt_csv,
        help=f"Path to training_df.csv (default: {default_gt_csv})",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run the full model pipeline instead of reading a pre-computed CSV.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models,
        help=f"Models directory (default: {default_models})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help=f"DICOM data directory (default: {default_data})",
    )
    parser.add_argument(
        "--ich-strategy",
        type=str,
        default="nnunet",
        help="ICH segmentation strategy for model loading (default: nnunet).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_out_csv,
        help=f"Output per-study CSV (default: {default_out_csv})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_out_json,
        help=f"Output metrics JSON (default: {default_out_json})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting CSV and JSON files.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the MLS task leaderboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    print()
    print("=" * 60)
    print("  📏  MIDLINE SHIFT — Task Leaderboard")
    print("=" * 60)

    # ---- Load ground truth ------------------------------------------------
    logger.info("Loading MLS ground truth ...")
    gt = load_ground_truth(args.gt_csv)

    # ---- Load predictions -------------------------------------------------
    if args.run_inference:
        _ensure_project_on_path()

        if args.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU.")
                    args.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not found, falling back to CPU.")
                args.device = "cpu"

        study_dirs = discover_study_dirs(args.data_dir)
        logger.info("Found %d DICOM study directories.", len(study_dirs))
        preds = run_inference(study_dirs, args.models_dir, args.device, args.ich_strategy)
        print(f"  Inference mode  : {args.ich_strategy} ({args.device})")
    else:
        logger.info("Reading predictions from %s ...", args.input_csv)
        preds = load_predictions_from_csv(args.input_csv)
        print(f"  CSV mode        : {args.input_csv}")

    print(f"  Ground truth CSV: {args.gt_csv}")
    print(f"  Device          : {args.device}")
    print("=" * 60)

    # ---- Align + compute metrics ------------------------------------------
    study_ids, y_true, y_pred = _align(gt, preds)
    metrics = compute_metrics(study_ids, y_true, y_pred)
    print_report(metrics)

    # ---- Export (with auto-versioning) ------------------------------------
    if not args.no_export:
        from leaderboard.scorer import versioned_output_pair
        csv_path, json_path = versioned_output_pair(args.output_csv, args.output_json)
        logger.info("Output CSV  : %s", csv_path)
        logger.info("Output JSON : %s", json_path)
        export_results_csv(csv_path, study_ids, y_true, y_pred)
        export_metrics_json(metrics, json_path)
        print(f"📄 Per-study results : {csv_path.resolve()}")
        print(f"📄 Metrics JSON      : {json_path.resolve()}")
    else:
        print(f"📄 Per-study results : {args.output_csv.resolve()} (skipped — --no-export)")
        print(f"📄 Metrics JSON      : {args.output_json.resolve()} (skipped — --no-export)")

    print("✅ MLS task evaluation complete!\n")

    return metrics


if __name__ == "__main__":
    main()
