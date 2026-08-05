"""
task_fracture.py — Task-specific leaderboard for skull fracture detection.

Evaluates the skull fracture prediction (fracture_prob) against ground truth
SkullFracture labels from training_df.csv, independently of the triage pipeline.

Key metrics:
  - AUC-ROC (primary)
  - Binary classification at threshold 0.5: accuracy, precision, recall, F1
  - Optimal threshold via Youden's J statistic
  - Confusion matrix + classification report

Usage (CSV mode — requires a pre-computed results.csv from evaluate.py)::

    python -m leaderboard.task_fracture
    python -m leaderboard.task_fracture --input-csv path/to/results.csv

Usage (inference mode — runs the full model pipeline)::

    python -m leaderboard.task_fracture --run-inference
    python -m leaderboard.task_fracture --run-inference --ich-strategy monai --device cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    classification_report,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers (mirrors evaluate.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_project_on_path() -> None:
    """Add project root to sys.path so submission/ is importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth(csv_path: str | Path) -> Dict[str, bool]:
    """Load SkullFracture ground truth aggregated to study level.

    Args:
        csv_path: Path to training_df.csv.

    Returns:
        Dict mapping study_id → bool (True = fracture present in any slice).
    """
    csv_path = Path(csv_path)
    logger.info("Loading fracture ground truth from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    series = df.groupby("dicom_series.id")["SkullFracture"].max()
    result = {str(k): bool(v) for k, v in series.items()}
    logger.info("Loaded %d fracture ground-truth labels.", len(result))
    return result


# ---------------------------------------------------------------------------
# Prediction loading (CSV mode)
# ---------------------------------------------------------------------------
def load_predictions_from_csv(csv_path: str | Path) -> Dict[str, float]:
    """Load fracture predictions from a pre-computed results CSV.

    Args:
        csv_path: Path to results.csv (produced by evaluate.py).

    Returns:
        Dict mapping study_id → fracture_prob (float).
    """
    csv_path = Path(csv_path)
    logger.info("Loading fracture predictions from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    result: Dict[str, float] = {}
    for _, row in df.iterrows():
        sid = str(row["study_id"])
        result[sid] = float(row["pred_fracture_prob"])
    logger.info("Loaded %d fracture predictions.", len(result))
    return result


# ---------------------------------------------------------------------------
# Prediction loading (inference mode)
# ---------------------------------------------------------------------------
def discover_study_dirs(data_dir: Path) -> Dict[str, Path]:
    """Find all study directories containing .dcm files."""
    from leaderboard.evaluate import discover_study_dirs as _discover
    return _discover(data_dir)


def run_inference(
    study_dirs: Dict[str, Path],
    models_dir: Path,
    device: str = "cuda",
    ich_strategy: str = "nnunet",
) -> Dict[str, float]:
    """Run the full model pipeline and extract fracture probabilities.

    Args:
        study_dirs: Dict mapping study_id → directory path.
        models_dir: Path to submission/models/.
        device: 'cuda' or 'cpu'.
        ich_strategy: ICH segmentation strategy name.

    Returns:
        Dict mapping study_id → fracture_prob (float).
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
        progress = tqdm(sorted(study_dirs.items()), desc="Fracture inference", unit="study")
    except ImportError:
        progress = sorted(study_dirs.items())

    for study_id, study_dir in progress:
        try:
            intermediates = model_predict(str(study_dir), models=models)
            predictions[study_id] = float(intermediates["fracture_prob"])
        except Exception as exc:
            errors.append(f"{study_id}: {exc}")

    if errors:
        logger.warning("%d study(s) failed during inference.", len(errors))
        for err in errors[:10]:
            logger.warning("  - %s", err)

    logger.info("Inference complete: %d/%d studies.", len(predictions), len(study_dirs))
    return predictions


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute full suite of fracture-detection metrics.

    Args:
        study_ids: List of study identifiers.
        y_true: Binary ground truth (0 or 1).
        y_pred_prob: Predicted fracture probability in [0, 1].

    Returns:
        Metrics dictionary.
    """
    n = len(y_true)
    if n == 0:
        return {"error": "No samples to evaluate."}

    # -- AUC-ROC ------------------------------------------------------------
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        logger.warning("Only one class present in ground truth — AUC is undefined.")
    else:
        auc = float(roc_auc_score(y_true, y_pred_prob))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # -- Binary classification at default threshold 0.5 ----------------------
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    acc_50 = float(accuracy_score(y_true, y_pred_bin))
    prec_50 = float(precision_score(y_true, y_pred_bin, zero_division=0))
    rec_50 = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1_50 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    cm_50 = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).tolist()
    report_50 = classification_report(
        y_true, y_pred_bin,
        labels=[0, 1],
        target_names=["No Fracture", "Fracture"],
        output_dict=True,
        zero_division=0,
    )

    # -- Optimal threshold via Youden's J ------------------------------------
    if len(thresholds) > 1:
        youden_j = tpr - fpr
        best_idx = int(np.argmax(youden_j))
        best_threshold = float(thresholds[best_idx])
        y_pred_opt = (y_pred_prob >= best_threshold).astype(int)
        acc_opt = float(accuracy_score(y_true, y_pred_opt))
        prec_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
        rec_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
        f1_opt = float(f1_score(y_true, y_pred_opt, zero_division=0))
        cm_opt = confusion_matrix(y_true, y_pred_opt, labels=[0, 1]).tolist()
    else:
        best_threshold = 0.5
        acc_opt, prec_opt, rec_opt, f1_opt = acc_50, prec_50, rec_50, f1_50
        cm_opt = cm_50

    # -- Distribution of predictions per class -------------------------------
    neg_preds = y_pred_prob[y_true == 0]
    pos_preds = y_pred_prob[y_true == 1]

    # -- Counts --------------------------------------------------------------
    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos

    metrics = {
        "task": "skull_fracture",
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "auc_roc": round(auc, 6),
        "default_threshold_0.5": {
            "accuracy": round(acc_50, 6),
            "precision": round(prec_50, 4),
            "recall": round(rec_50, 4),
            "f1_score": round(f1_50, 4),
            "confusion_matrix": cm_50,
        },
        "optimal_threshold": {
            "threshold": round(best_threshold, 6),
            "youden_j": float(youden_j[best_idx]) if len(thresholds) > 1 else None,
            "accuracy": round(acc_opt, 6),
            "precision": round(prec_opt, 4),
            "recall": round(rec_opt, 4),
            "f1_score": round(f1_opt, 4),
            "confusion_matrix": cm_opt,
        },
        "per_class_report": {
            str(k): {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1_score": round(v["f1-score"], 4),
                "support": int(v["support"]),
            }
            for k, v in report_50.items()
            if k in ["No Fracture", "Fracture"]
        },
        "prediction_distribution": {
            "negative_mean": float(np.mean(neg_preds)) if len(neg_preds) > 0 else None,
            "negative_std": float(np.std(neg_preds)) if len(neg_preds) > 0 else None,
            "positive_mean": float(np.mean(pos_preds)) if len(pos_preds) > 0 else None,
            "positive_std": float(np.std(pos_preds)) if len(pos_preds) > 0 else None,
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print fracture evaluation metrics."""
    if "error" in metrics:
        print(f"\n  ⚠️  {metrics['error']}")
        return

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🦴  FRACTURE DETECTION — Task Leaderboard".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Studies evaluated: {metrics['n_samples']:<38}║")
    print(f"║  Prevalence: {metrics['n_positive']}/{metrics['n_samples']} "
          f"({100*metrics['n_positive']/metrics['n_samples']:.1f}%)          ║")
    print(f"║  AUC-ROC:    {metrics['auc_roc']:>8.4f}                                ║")

    print("╠" + "═" * 58 + "╣")
    print("║  At default threshold (0.5):                                   ║")
    d = metrics["default_threshold_0.5"]
    print(f"║    Accuracy : {d['accuracy']:.4f}  ({d['accuracy']*100:.1f}%)             ║")
    print(f"║    Precision: {d['precision']:.4f}                               ║")
    print(f"║    Recall   : {d['recall']:.4f}                               ║")
    print(f"║    F1-score : {d['f1_score']:.4f}                               ║")

    cm = d["confusion_matrix"]
    print("║    Confusion Matrix (rows=GT, cols=Pred):                    ║")
    print(f"║              Pred Neg   Pred Pos                              ║")
    print(f"║    GT Neg    {cm[0][0]:>6}     {cm[0][1]:>6}                             ║")
    print(f"║    GT Pos    {cm[1][0]:>6}     {cm[1][1]:>6}                             ║")

    print("╠" + "═" * 58 + "╣")
    print("║  At optimal threshold:                                        ║")
    o = metrics["optimal_threshold"]
    print(f"║    Threshold: {o['threshold']:.4f}  (Youden J={o.get('youden_j', 'N/A')})        ║")
    print(f"║    Accuracy : {o['accuracy']:.4f}  ({o['accuracy']*100:.1f}%)             ║")
    print(f"║    Precision: {o['precision']:.4f}                               ║")
    print(f"║    Recall   : {o['recall']:.4f}                               ║")
    print(f"║    F1-score : {o['f1_score']:.4f}                               ║")

    print("╠" + "═" * 58 + "╣")
    pd_dist = metrics["prediction_distribution"]
    print("║  Prediction distribution:                                      ║")
    neg_mean = pd_dist.get("negative_mean")
    pos_mean = pd_dist.get("positive_mean")
    if neg_mean is not None:
        print(f"║    Negative class: mean={neg_mean:.4f}  std={pd_dist['negative_std']:.4f}        ║")
    if pos_mean is not None:
        print(f"║    Positive class: mean={pos_mean:.4f}  std={pd_dist['positive_std']:.4f}        ║")
    print("╚" + "═" * 58 + "╝")
    print()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_results_csv(
    output_path: str | Path,
    study_ids: List[str],
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> Path:
    """Export per-study fracture predictions to CSV.

    Args:
        output_path: Destination CSV path.
        study_ids: Study identifiers.
        y_true: Binary ground truth.
        y_pred_prob: Predicted probabilities.

    Returns:
        Path to the saved CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "study_id": sid,
            "gt_fracture": int(gt),
            "pred_fracture_prob": round(float(pred), 6),
            "pred_fracture_binary_0.5": int(pred >= 0.5),
            "correct_0.5": int(gt == (pred >= 0.5)),
        }
        for sid, gt, pred in zip(study_ids, y_true, y_pred_prob)
    ]
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Fracture results exported to %s (%d rows)", output_path, len(df))
    return output_path


def export_metrics_json(metrics: Dict[str, Any], output_path: str | Path) -> Path:
    """Export metrics dict as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Fracture metrics exported to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------
def _align_ground_truth_and_predictions(
    gt: Dict[str, bool],
    preds: Dict[str, float],
) -> tuple[List[str], np.ndarray, np.ndarray]:
    """Align ground truth and predictions on shared study IDs."""
    matched = sorted(set(gt.keys()) & set(preds.keys()))
    only_gt = set(gt.keys()) - set(preds.keys())
    only_pred = set(preds.keys()) - set(gt.keys())

    if only_gt:
        logger.warning("%d studies in GT but not in predictions (skipped).", len(only_gt))
    if only_pred:
        logger.warning("%d studies in predictions but not in GT (skipped).", len(only_pred))

    if not matched:
        raise ValueError("No matching studies between ground truth and predictions.")

    y_true = np.array([int(gt[sid]) for sid in matched], dtype=int)
    y_pred_prob = np.array([float(preds[sid]) for sid in matched])
    return matched, y_true, y_pred_prob


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    default_results = PROJECT_ROOT / "leaderboard" / "results.csv"
    default_gt_csv = PROJECT_ROOT / "Data" / "metadata" / "training_df.csv"
    default_models = PROJECT_ROOT / "submission" / "models"
    default_data = PROJECT_ROOT / "Data" / "raw" / "training"
    default_out_csv = PROJECT_ROOT / "leaderboard" / "fracture_results.csv"
    default_out_json = PROJECT_ROOT / "leaderboard" / "fracture_metrics.json"

    parser = argparse.ArgumentParser(
        description="🦴 Skull Fracture Detection — Task Leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leaderboard.task_fracture                                  # from default results.csv
  python -m leaderboard.task_fracture --input-csv results_monai.csv    # from custom CSV
  python -m leaderboard.task_fracture --run-inference                  # run model pipeline
  python -m leaderboard.task_fracture --run-inference --device cpu     # inference on CPU
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
    """Run the fracture task leaderboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    print()
    print("=" * 60)
    print("  🦴  FRACTURE DETECTION — Task Leaderboard")
    print("=" * 60)

    # ---- Load ground truth ------------------------------------------------
    logger.info("Loading fracture ground truth ...")
    gt = load_ground_truth(args.gt_csv)

    # ---- Load predictions -------------------------------------------------
    if args.run_inference:
        _ensure_project_on_path()

        # CUDA check
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
    study_ids, y_true, y_pred_prob = _align_ground_truth_and_predictions(gt, preds)
    metrics = compute_metrics(study_ids, y_true, y_pred_prob)
    print_report(metrics)

    # ---- Export (with auto-versioning) ------------------------------------
    if not args.no_export:
        from leaderboard.scorer import versioned_output_pair
        csv_path, json_path = versioned_output_pair(args.output_csv, args.output_json)
        logger.info("Output CSV  : %s", csv_path)
        logger.info("Output JSON : %s", json_path)
        export_results_csv(csv_path, study_ids, y_true, y_pred_prob)
        export_metrics_json(metrics, json_path)
        print(f"📄 Per-study results : {csv_path.resolve()}")
        print(f"📄 Metrics JSON      : {json_path.resolve()}")
    else:
        print(f"📄 Per-study results : {args.output_csv.resolve()} (skipped — --no-export)")
        print(f"📄 Metrics JSON      : {args.output_json.resolve()} (skipped — --no-export)")

    print("✅ Fracture task evaluation complete!\n")

    return metrics


if __name__ == "__main__":
    main()
