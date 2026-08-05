"""
task_hemorrhage.py — Task-specific leaderboard for hemorrhage detection & volume estimation.

Evaluates ICH predictions (V_IVH, V_IPH, V_SDH, V_EDH, V_SAH volumes) against
ground truth from training_df.csv at TWO levels:

  1. Binary detection (AnyICH + per-type): AUC-ROC, accuracy, precision, recall, F1
  2. Volume regression (per-type + total): MAE, MSE, RMSE, R²

Supports comparing across all ICH strategies (--compare-all).

Usage (CSV mode)::

    python -m leaderboard.task_hemorrhage
    python -m leaderboard.task_hemorrhage --input-csv path/to/results.csv

Usage (single strategy inference)::

    python -m leaderboard.task_hemorrhage --run-inference --ich-strategy nnunet

Usage (compare all strategies)::

    python -m leaderboard.task_hemorrhage --run-inference --compare-all
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
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_project_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hemorrhage type keys (as in model output and ground_truth.py)
HEM_TYPES = ["IVH", "IPH", "SDH", "EDH", "SAH"]
VOL_KEYS = [f"V_{t}" for t in HEM_TYPES]  # V_IVH, V_IPH, V_SDH, V_EDH, V_SAH

# Column names in the raw CSV for boolean hemorrhage flags
BOOL_COLUMNS = {
    "IVH": "IntraventricularHemorrhage",
    "IPH": "IntraparenchymalHemorrhage",
    "SDH": "SubduralHemorrhage",
    "EDH": "EpiduralHemorrhage",
    "SAH": "SubarachnoidHemorrhage",
}

# Volume threshold for binary detection (matches triage.py EPS_VOLUME)
EPS_VOLUME = 0.1  # mL

# ICH strategies (for --compare-all)
ALL_STRATEGIES = ["nnunet", "smp", "monai", "yolo_seg"]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth(csv_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load hemorrhage ground truth at study level.

    Returns a dict mapping study_id → info with:
      - bool_IVH, bool_IPH, bool_SDH, bool_EDH, bool_SAH: binary flags
      - bool_AnyICH: True if any hemorrhage type present
      - V_IVH, V_IPH, V_SDH, V_EDH, V_SAH: volumes in mL
      - total_volume_ml: sum of all volumes
    """
    csv_path = Path(csv_path)
    logger.info("Loading hemorrhage ground truth from %s ...", csv_path)

    df = pd.read_csv(csv_path)

    # ---- Aggregate boolean flags (max = OR across slices) -----------------
    bool_agg = {col: "max" for col in BOOL_COLUMNS.values()}
    agg_dict: Dict[str, Any] = {
        **bool_agg,
        "IntraventricularHemorrhage_Area": "sum",
        "IntraparenchymalHemorrhage_Area": "sum",
        "SubarachnoidHemorrhage_Area": "sum",
        "EpiduralHemorrhage_Area": "sum",
        "SubduralHemorrhage_Area": "sum",
        "dicom_series.PixelSpacing0": "first",
        "dicom_series.PixelSpacing1": "first",
    }

    series_df = df.groupby("dicom_series.id").agg(agg_dict).reset_index()

    # ---- Compute volumes --------------------------------------------------
    spacing_x = series_df["dicom_series.PixelSpacing0"]
    spacing_y = series_df["dicom_series.PixelSpacing1"]
    thickness_map = df.groupby("dicom_series.id")["dicom_series.SliceThickness"].first()
    thickness = series_df["dicom_series.id"].map(thickness_map)
    factor = spacing_x * spacing_y * thickness / 1000.0

    area_to_vol = {
        "IntraventricularHemorrhage_Area": "V_IVH",
        "IntraparenchymalHemorrhage_Area": "V_IPH",
        "SubarachnoidHemorrhage_Area": "V_SAH",
        "EpiduralHemorrhage_Area": "V_EDH",
        "SubduralHemorrhage_Area": "V_SDH",
    }

    # Compute volumes on the whole dataframe (avoids scalar/Series mismatch)
    for area_col, vol_key in area_to_vol.items():
        series_df[vol_key] = series_df[area_col] * factor

    series_df["total_volume_ml"] = series_df[
        list(area_to_vol.values())
    ].sum(axis=1)

    result: Dict[str, Dict[str, Any]] = {}
    for _, row in series_df.iterrows():
        study_id = str(row["dicom_series.id"])

        # Boolean flags
        entry: Dict[str, Any] = {}
        for hem_type, col in BOOL_COLUMNS.items():
            entry[f"bool_{hem_type}"] = bool(row[col])

        entry["bool_AnyICH"] = any(entry[f"bool_{t}"] for t in HEM_TYPES)

        # Volumes
        for vol_key in area_to_vol.values():
            entry[vol_key] = float(row[vol_key])

        entry["total_volume_ml"] = float(row["total_volume_ml"])
        result[study_id] = entry

    logger.info("Loaded hemorrhage ground truth for %d studies.", len(result))
    return result


# ---------------------------------------------------------------------------
# Prediction loading (CSV mode)
# ---------------------------------------------------------------------------
def load_predictions_from_csv(csv_path: str | Path) -> Dict[str, Dict[str, float]]:
    """Load ICH predictions from a pre-computed results CSV.

    Returns dict mapping study_id → {V_IVH, V_IPH, V_SDH, V_EDH, V_SAH}.
    """
    csv_path = Path(csv_path)
    logger.info("Loading ICH predictions from %s ...", csv_path)
    df = pd.read_csv(csv_path)

    result: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        sid = str(row["study_id"])
        result[sid] = {k: float(row[f"pred_{k}"]) for k in VOL_KEYS}

    logger.info("Loaded ICH predictions for %d studies.", len(result))
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
) -> Dict[str, Dict[str, float]]:
    """Run the full model pipeline and extract ICH volume predictions.

    Returns dict mapping study_id → {V_IVH, V_IPH, V_SDH, V_EDH, V_SAH}.
    """
    _ensure_project_on_path()
    from submission.model import load_models, predict as model_predict

    logger.info(
        "Loading models (ich_strategy=%s, device=%s) ...", ich_strategy, device
    )
    models = load_models(str(models_dir), device=device, ich_strategy=ich_strategy)
    logger.info("Models loaded. Running inference on %d studies ...", len(study_dirs))

    predictions: Dict[str, Dict[str, float]] = {}
    errors: List[str] = []

    try:
        from tqdm import tqdm
        progress = tqdm(sorted(study_dirs.items()), desc=f"ICH {ich_strategy}", unit="study")
    except ImportError:
        progress = sorted(study_dirs.items())

    for study_id, study_dir in progress:
        try:
            intermediates = model_predict(str(study_dir), models=models)
            predictions[study_id] = {k: float(intermediates[k]) for k in VOL_KEYS}
        except Exception as exc:
            errors.append(f"{study_id}: {exc}")

    if errors:
        logger.warning("%d study(s) failed during inference.", len(errors))
        for err in errors[:10]:
            logger.warning("  - %s", err)

    logger.info(
        "Inference complete: %d/%d studies.", len(predictions), len(study_dirs)
    )
    return predictions


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _binary_metrics(
    y_true: np.ndarray,
    y_pred_score: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    """Compute binary classification metrics for a hemorrhage type.

    Args:
        y_true: Binary ground truth (0 or 1).
        y_pred_score: Predicted volume (used as score for AUC) or binary pred.
        label: Display label for this hemorrhage type.

    Returns:
        Metrics dict.
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No samples", "label": label}

    # AUC-ROC
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_pred_score))
        fpr, tpr, _ = roc_curve(y_true, y_pred_score)
    else:
        auc = float("nan")
        fpr = tpr = None

    # Binary at EPS_VOLUME threshold
    y_pred_bin = (y_pred_score >= EPS_VOLUME).astype(int)

    acc = float(accuracy_score(y_true, y_pred_bin))
    prec = float(precision_score(y_true, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).tolist()
    prevalence = float(np.mean(y_true))

    return {
        "label": label,
        "n_samples": n,
        "prevalence": round(prevalence, 4),
        "auc_roc": round(auc, 6),
        "threshold_ml": EPS_VOLUME,
        "accuracy": round(acc, 6),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    """Compute regression metrics for hemorrhage volume.

    Args:
        y_true: Ground truth volumes (mL).
        y_pred: Predicted volumes (mL).
        label: Display label.

    Returns:
        Metrics dict.
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No samples", "label": label}

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Only compute correlation if enough non-constant values
    if n >= 3 and np.std(y_true) > 1e-10 and np.std(y_pred) > 1e-10:
        from scipy.stats import pearsonr
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    else:
        pearson_r = pearson_p = float("nan")

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    return {
        "label": label,
        "n_samples": n,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p_value": round(float(pearson_p), 6),
        "mean_error": round(float(np.mean(errors)), 4),
        "median_abs_error": round(float(np.median(abs_errors)), 4),
        "gt_mean": round(float(np.mean(y_true)), 4),
        "pred_mean": round(float(np.mean(y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# Metric computation (full)
# ---------------------------------------------------------------------------
def compute_metrics(
    study_ids: List[str],
    gt_dict: Dict[str, Dict[str, Any]],
    pred_dict: Dict[str, Dict[str, float]],
    strategy_label: str = "",
) -> Dict[str, Any]:
    """Compute full suite of hemorrhage metrics.

    Args:
        study_ids: List of matched study IDs.
        gt_dict: Ground truth dict (from load_ground_truth).
        pred_dict: Predictions dict (from CSV or inference).
        strategy_label: Optional label for the ICH strategy.

    Returns:
        Metrics dictionary.
    """
    n = len(study_ids)
    if n == 0:
        return {"error": "No samples to evaluate.", "strategy": strategy_label}

    # Build aligned arrays
    binary_metrics_list: List[Dict[str, Any]] = []
    regression_metrics_list: List[Dict[str, Any]] = []

    # -- Per-type metrics ---------------------------------------------------
    for hem_type in HEM_TYPES:
        vol_key = f"V_{hem_type}"
        y_true_bin = np.array([int(gt_dict[sid][f"bool_{hem_type}"]) for sid in study_ids])
        y_true_vol = np.array([gt_dict[sid][vol_key] for sid in study_ids])
        y_pred_vol = np.array([pred_dict[sid][vol_key] for sid in study_ids])

        binary_metrics_list.append(
            _binary_metrics(y_true_bin, y_pred_vol, hem_type)
        )
        regression_metrics_list.append(
            _regression_metrics(y_true_vol, y_pred_vol, hem_type)
        )

    # -- AnyICH detection ----------------------------------------------------
    y_true_any = np.array([int(gt_dict[sid]["bool_AnyICH"]) for sid in study_ids])
    y_pred_total = np.array([
        sum(pred_dict[sid][k] for k in VOL_KEYS)
        for sid in study_ids
    ])
    anyich_metrics = _binary_metrics(y_true_any, y_pred_total, "AnyICH")

    # -- Total volume regression ---------------------------------------------
    y_true_total = np.array([gt_dict[sid]["total_volume_ml"] for sid in study_ids])
    total_vol_metrics = _regression_metrics(y_true_total, y_pred_total, "Total")

    # -- Overall detection summary -------------------------------------------
    det_summary = {m["label"]: {"auc_roc": m["auc_roc"], "f1_score": m["f1_score"]}
                    for m in binary_metrics_list}
    det_summary["AnyICH"] = {"auc_roc": anyich_metrics["auc_roc"],
                              "f1_score": anyich_metrics["f1_score"]}

    metrics = {
        "task": "hemorrhage",
        "strategy": strategy_label,
        "n_samples": n,
        "binary_detection": {
            "per_type": binary_metrics_list,
            "any_ich": anyich_metrics,
        },
        "volume_regression": {
            "per_type": regression_metrics_list,
            "total_volume": total_vol_metrics,
        },
        "detection_summary": det_summary,
    }

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_binary_table(bin_metrics_list: List[Dict[str, Any]],
                        anyich: Dict[str, Any]) -> None:
    """Print a compact binary detection table."""
    print("║  Detection Metrics (threshold: volume >= 0.1 mL):                ║")
    print("║  Type         AUC-ROC   Acc      Prec     Rec      F1       Prev ║")
    print("║  ────────────────────────────────────────────────────────────────║")

    for m in bin_metrics_list:
        auc_str = f"{m['auc_roc']:.4f}" if not np.isnan(m['auc_roc']) else "  N/A  "
        print(f"║  {m['label']:<8}  {auc_str}  {m['accuracy']:.4f}  "
              f"{m['precision']:.4f}  {m['recall']:.4f}  {m['f1_score']:.4f}  "
              f"{m['prevalence']:.3f}       ║")

    auc_str = f"{anyich['auc_roc']:.4f}" if not np.isnan(anyich['auc_roc']) else "  N/A  "
    print(f"║  ────────────────────────────────────────────────────────────────║")
    print(f"║  AnyICH    {auc_str}  {anyich['accuracy']:.4f}  "
          f"{anyich['precision']:.4f}  {anyich['recall']:.4f}  "
          f"{anyich['f1_score']:.4f}  {anyich['prevalence']:.3f}       ║")


def _print_regression_table(reg_metrics_list: List[Dict[str, Any]],
                            total_vol: Dict[str, Any]) -> None:
    """Print a compact volume regression table."""
    print("║  Volume Regression Metrics (mL):                                ║")
    print("║  Type         MAE      RMSE     R²       Pearson  GT-mean  Pred-mean║")
    print("║  ────────────────────────────────────────────────────────────────║")

    for m in reg_metrics_list:
        r_str = f"{m['pearson_r']:.4f}" if not np.isnan(m['pearson_r']) else "  N/A  "
        print(f"║  {m['label']:<8}  {m['mae']:>7.4f}  {m['rmse']:>7.4f}  "
              f"{m['r2_score']:>7.4f}  {r_str}  "
              f"{m['gt_mean']:>7.4f}  {m['pred_mean']:>7.4f}    ║")

    r_str = f"{total_vol['pearson_r']:.4f}" if not np.isnan(total_vol['pearson_r']) else "  N/A  "
    print(f"║  ────────────────────────────────────────────────────────────────║")
    print(f"║  Total     {total_vol['mae']:>7.4f}  {total_vol['rmse']:>7.4f}  "
          f"{total_vol['r2_score']:>7.4f}  {r_str}  "
          f"{total_vol['gt_mean']:>7.4f}  {total_vol['pred_mean']:>7.4f}    ║")


def print_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print hemorrhage evaluation metrics."""
    if "error" in metrics:
        print(f"\n  ⚠️  {metrics['error']}")
        return

    strategy = metrics.get("strategy", "")
    strat_tag = f"  [{strategy}]" if strategy else ""

    print()
    print("╔" + "═" * 66 + "╗")
    print("║" + f"  🩸  HEMORRHAGE — Task Leaderboard{strat_tag}".ljust(66) + "║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  Studies evaluated: {metrics['n_samples']:<44}║")
    print("╠" + "═" * 66 + "╣")

    _print_binary_table(
        metrics["binary_detection"]["per_type"],
        metrics["binary_detection"]["any_ich"],
    )
    print("╠" + "═" * 66 + "╣")

    _print_regression_table(
        metrics["volume_regression"]["per_type"],
        metrics["volume_regression"]["total_volume"],
    )

    print("╠" + "═" * 66 + "╣")
    det = metrics["detection_summary"]
    best_type = max(det, key=lambda k: det[k]["f1_score"] if not np.isnan(det[k].get("f1_score", 0)) else -1)
    best_auc = max(det, key=lambda k: det[k].get("auc_roc", -1) if not np.isnan(det[k].get("auc_roc", -1)) else -1)
    print(f"║  Best detection F1   : {best_type} ({det[best_type]['f1_score']:.4f})            ║")
    print(f"║  Best detection AUC  : {best_auc} ({det[best_auc]['auc_roc']:.4f})            ║")
    print("╚" + "═" * 66 + "╝")
    print()


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------
def _print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print side-by-side comparison of hemorrhage metrics across strategies."""
    print()
    print("╔" + "═" * 72 + "╗")
    print("║" + "  📊  HEMORRHAGE STRATEGY COMPARISON".ljust(70) + "║")
    print("╠" + "═" * 72 + "╣")

    # Header
    print("║  {:<12s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}║".format(
        "Strategy", "AnyICH", "AnyICH", "AnyICH", "Vol-MAE", "Vol-RMSE", "Vol-R²", "Time(s)",
    ))
    print("║  {:>12s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}║".format(
        "", "AUC", "F1", "Acc", "", "", "", ""
    ))
    print("║" + "  " + "─" * 68 + "  ║")

    best_f1 = -1.0
    best_name = ""
    for r in results:
        if "error" in r:
            continue
        anyich = r.get("binary_detection", {}).get("any_ich", {})
        total_vol = r.get("volume_regression", {}).get("total_volume", {})

        auc = anyich.get("auc_roc", 0)
        f1 = anyich.get("f1_score", 0)
        acc = anyich.get("accuracy", 0)
        mae = total_vol.get("mae", 0)
        rmse = total_vol.get("rmse", 0)
        r2 = total_vol.get("r2_score", 0)
        t = r.get("time_s", 0)

        if f1 > best_f1:
            best_f1 = f1
            best_name = r.get("strategy", "")

        marker = "👑" if r.get("strategy", "") == best_name else "  "
        print("║ {}{:<10s}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>6.1f}s  ║".format(
            marker, r.get("strategy", ""), auc, f1, acc, mae, rmse, r2, t,
        ))

    print("╠" + "═" * 72 + "╣")
    print("║" + f"  👑 Best AnyICH F1: {best_name} = {best_f1:.4f}".ljust(70) + "║")
    print("╚" + "═" * 72 + "╝")
    print()


def _run_single_strategy(
    strategy_name: str,
    models_dir: Path,
    device: str,
    study_dirs: Dict[str, Path],
    gt: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run hemorrhage evaluation for a single ICH strategy."""
    logger.info("─" * 60)
    logger.info("🔄  Hemorrhage evaluation — strategy: %s", strategy_name)
    logger.info("─" * 60)

    start_t = time.time()
    preds = run_inference(study_dirs, models_dir, device, strategy_name)
    elapsed = time.time() - start_t

    matched = sorted(set(gt.keys()) & set(preds.keys()))
    if not matched:
        return {"strategy": strategy_name, "error": "No matched studies"}

    metrics = compute_metrics(matched, gt, preds, strategy_label=strategy_name)
    metrics["time_s"] = round(elapsed, 1)
    return metrics


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_results_csv(
    output_path: str | Path,
    study_ids: List[str],
    gt_dict: Dict[str, Dict[str, Any]],
    pred_dict: Dict[str, Dict[str, float]],
) -> Path:
    """Export per-study hemorrhage predictions to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for sid in study_ids:
        gt = gt_dict[sid]
        pred = pred_dict[sid]
        row = {"study_id": sid}

        # Ground truth booleans + volumes
        for hem_type in HEM_TYPES:
            row[f"gt_bool_{hem_type}"] = int(gt[f"bool_{hem_type}"])
            row[f"gt_{hem_type}_vol"] = round(gt[f"V_{hem_type}"], 4)

        row["gt_bool_AnyICH"] = int(gt["bool_AnyICH"])
        row["gt_total_vol"] = round(gt["total_volume_ml"], 4)

        # Predictions
        for vol_key in VOL_KEYS:
            row[f"pred_{vol_key}"] = round(pred[vol_key], 4)

        row["pred_total_vol"] = round(sum(pred[k] for k in VOL_KEYS), 4)
        row["pred_AnyICH"] = int(row["pred_total_vol"] >= EPS_VOLUME)

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Hemorrhage results exported to %s (%d rows)", output_path, len(df))
    return output_path


def export_metrics_json(metrics: Dict[str, Any], output_path: str | Path) -> Path:
    """Export metrics dict as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle NaN values for JSON serialization
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    sanitized = _sanitize(metrics)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)
    logger.info("Hemorrhage metrics exported to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------
def _align(
    gt: Dict[str, Dict[str, Any]],
    preds: Dict[str, Dict[str, float]],
) -> List[str]:
    """Align ground truth and predictions, returning matched study IDs."""
    matched = sorted(set(gt.keys()) & set(preds.keys()))
    only_gt = set(gt.keys()) - set(preds.keys())
    only_pred = set(preds.keys()) - set(gt.keys())

    if only_gt:
        logger.warning("%d studies in GT but not predictions (skipped).", len(only_gt))
    if only_pred:
        logger.warning("%d studies in predictions but not GT (skipped).", len(only_pred))

    if not matched:
        raise ValueError("No matching studies between ground truth and predictions.")

    return matched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    default_results = PROJECT_ROOT / "leaderboard" / "results.csv"
    default_gt_csv = PROJECT_ROOT / "Data" / "metadata" / "training_df.csv"
    default_models = PROJECT_ROOT / "submission" / "models"
    default_data = PROJECT_ROOT / "Data" / "raw" / "training"
    default_out_csv = PROJECT_ROOT / "leaderboard" / "hemorrhage_results.csv"
    default_out_json = PROJECT_ROOT / "leaderboard" / "hemorrhage_metrics.json"

    parser = argparse.ArgumentParser(
        description="🩸 Hemorrhage Detection & Volume — Task Leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leaderboard.task_hemorrhage                                     # from default results.csv
  python -m leaderboard.task_hemorrhage --input-csv results_nnunet.csv      # from custom CSV
  python -m leaderboard.task_hemorrhage --run-inference --ich-strategy monai
  python -m leaderboard.task_hemorrhage --run-inference --compare-all       # all strategies
  python -m leaderboard.task_hemorrhage --compare-all --device cpu
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
        help="ICH segmentation strategy (default: nnunet). Ignored with --compare-all.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run ALL available ICH strategies and compare side-by-side.",
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
    """Run the hemorrhage task leaderboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- Load ground truth once -------------------------------------------
    logger.info("Loading hemorrhage ground truth ...")
    gt = load_ground_truth(args.gt_csv)

    # ---- Determine mode ---------------------------------------------------
    if args.run_inference or args.compare_all:
        _ensure_project_on_path()

        # CUDA check (only in inference/compare-all mode)
        if args.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU.")
                    args.device = "cpu"
            except (ImportError, OSError):
                logger.warning("PyTorch/CUDA not available, falling back to CPU.")
                args.device = "cpu"

    if args.run_inference and args.compare_all:
        # ── Compare-all mode ────────────────────────────────────────────────
        print()
        print("=" * 70)
        print("  🧪  HEMORRHAGE COMPARE-ALL: Evaluating every ICH strategy")
        print("=" * 70)
        print(f"  Models dir : {args.models_dir}")
        print(f"  Data dir   : {args.data_dir}")
        print(f"  Device     : {args.device}")
        print("=" * 70)

        study_dirs = discover_study_dirs(args.data_dir)
        logger.info("Found %d DICOM study directories.", len(study_dirs))

        results: List[Dict[str, Any]] = []
        for strategy in ALL_STRATEGIES:
            strategy_dir = args.models_dir / strategy
            if not strategy_dir.is_dir():
                logger.warning("Skipping '%s' — directory not found.", strategy)
                continue
            real_files = [f for f in strategy_dir.iterdir()
                          if f.is_file() and f.name != ".gitkeep"]
            if not real_files:
                logger.warning("Skipping '%s' — no model files.", strategy)
                continue

            try:
                metrics = _run_single_strategy(
                    strategy, args.models_dir, args.device, study_dirs, gt
                )
                results.append(metrics)
            except Exception as exc:
                logger.error("  ❌ %s failed: %s", strategy, exc)
                results.append({"strategy": strategy, "error": str(exc)})

        _print_comparison_table(results)

        # Export best (with auto-versioning)
        best = max(
            (r for r in results if "error" not in r),
            key=lambda r: r.get("binary_detection", {}).get("any_ich", {}).get("f1_score", -1),
            default=None,
        )
        if best and not args.no_export:
            from leaderboard.scorer import versioned_output_pair
            _, json_path = versioned_output_pair(args.output_csv, args.output_json)
            comparison_data = {
                "task": "hemorrhage",
                "type": "strategy_comparison",
                "results": results,
            }
            export_metrics_json(comparison_data, json_path)
            logger.info("Strategy comparison exported to %s", json_path)

        return {"comparison": results}

    elif args.run_inference:
        # ── Single-strategy inference mode ──────────────────────────────────
        print()
        print("=" * 60)
        print("  🩸  HEMORRHAGE — Task Leaderboard")
        print("=" * 60)

        study_dirs = discover_study_dirs(args.data_dir)
        logger.info("Found %d DICOM study directories.", len(study_dirs))

        preds = run_inference(study_dirs, args.models_dir, args.device, args.ich_strategy)
        print(f"  Inference mode  : {args.ich_strategy} ({args.device})")

        matched = _align(gt, preds)
        metrics = compute_metrics(matched, gt, preds, strategy_label=args.ich_strategy)
        print_report(metrics)

        if not args.no_export:
            from leaderboard.scorer import versioned_output_pair
            csv_path, json_path = versioned_output_pair(args.output_csv, args.output_json)
            logger.info("Output CSV  : %s", csv_path)
            logger.info("Output JSON : %s", json_path)
            export_results_csv(csv_path, matched, gt, preds)
            export_metrics_json(metrics, json_path)
            print(f"📄 Per-study results : {csv_path.resolve()}")
            print(f"📄 Metrics JSON      : {json_path.resolve()}")
        else:
            print(f"📄 Per-study results : {args.output_csv.resolve()} (skipped)")
            print(f"📄 Metrics JSON      : {args.output_json.resolve()} (skipped)")
        print("✅ Hemorrhage task evaluation complete!\n")
        return metrics

    else:
        # ── CSV mode (default) ─────────────────────────────────────────────
        print()
        print("=" * 60)
        print("  🩸  HEMORRHAGE — Task Leaderboard")
        print("=" * 60)

        logger.info("Reading predictions from %s ...", args.input_csv)
        preds = load_predictions_from_csv(args.input_csv)
        print(f"  CSV mode        : {args.input_csv}")

        matched = _align(gt, preds)
        metrics = compute_metrics(matched, gt, preds)
        print_report(metrics)

        if not args.no_export:
            from leaderboard.scorer import versioned_output_pair
            csv_path, json_path = versioned_output_pair(args.output_csv, args.output_json)
            logger.info("Output CSV  : %s", csv_path)
            logger.info("Output JSON : %s", json_path)
            export_results_csv(csv_path, matched, gt, preds)
            export_metrics_json(metrics, json_path)
            print(f"📄 Per-study results : {csv_path.resolve()}")
            print(f"📄 Metrics JSON      : {json_path.resolve()}")
        else:
            print(f"📄 Per-study results : {args.output_csv.resolve()} (skipped)")
            print(f"📄 Metrics JSON      : {args.output_json.resolve()} (skipped)")
        print("✅ Hemorrhage task evaluation complete!\n")
        return metrics


if __name__ == "__main__":
    main()
