"""
evaluate.py — Main entry point for the personal leaderboard.

Evaluates trained models (placed in submission/models/) against ground truth
from Data/metadata/training_df.csv using Quadratic Weighted Kappa (QWK).

Usage:
    python -m leaderboard.evaluate
    python -m leaderboard.evaluate --device cpu
    python -m leaderboard.evaluate --models-dir submission/models --device cuda
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Pre-import pyarrow to prevent WinError 6714 crash during nnU-Net import chain.
# When pyarrow is cached in sys.modules, subsequent pandas→pyarrow imports
# triggered by nnU-Net will skip the faulty _fill_cache scan on Windows.
import pyarrow  # noqa: F401

import numpy as np

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("leaderboard")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _get_project_root() -> Path:
    """Find the project root (parent of leaderboard/)."""
    return Path(__file__).resolve().parent.parent


def _add_submission_to_path() -> None:
    """Add submission/ directory to sys.path so we can import from it."""
    project_root = _get_project_root()
    submission_dir = project_root / "submission"
    if str(submission_dir) not in sys.path:
        sys.path.insert(0, str(submission_dir))


# ---------------------------------------------------------------------------
# Study discovery
# ---------------------------------------------------------------------------
def discover_study_dirs(data_dir: Path) -> Dict[str, Path]:
    """Find all study directories under data_dir.

    Args:
        data_dir: Path to directory containing one sub-directory per study
                  (each with *.dcm files).

    Returns:
        Dict mapping study_id (folder name) → full path.
    """
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    studies: Dict[str, Path] = {}
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Check that it contains at least one .dcm file
        dcm_files = list(entry.glob("*.dcm"))
        if dcm_files:
            studies[entry.name] = entry
        else:
            logger.debug("Skipping %s — no .dcm files found.", entry.name)

    logger.info("Found %d study directories with DICOM files in %s",
                len(studies), data_dir)
    return studies


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------
def run_inference(
    study_dirs: Dict[str, Path],
    models: dict,
    ground_truth: Dict[str, Dict[str, Any]],
) -> tuple[List[str], np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Run model inference on all matched studies.

    Args:
        study_dirs: Dict mapping study_id → directory path.
        models: Loaded models dict (from submission.model.load_models).
        ground_truth: Ground truth labels dict (from ground_truth.load_study_labels).

    Returns:
        Tuple of (study_ids, y_true array, y_pred array, intermediates list).
    """
    from submission.model import predict as model_predict
    from submission.triage import triage_from_intermediates

    # Intersect study IDs present in BOTH the data dir and the CSV
    csv_ids = set(ground_truth.keys())
    dir_ids = set(study_dirs.keys())
    matched_ids = sorted(csv_ids & dir_ids)
    only_csv = csv_ids - dir_ids
    only_dir = dir_ids - csv_ids

    logger.info(
        "Study matching: %d in CSV, %d on disk, %d matched.",
        len(csv_ids), len(dir_ids), len(matched_ids),
    )
    if only_csv:
        logger.info(
            "  ⚠️  %d studies in CSV but NOT on disk (skipped): %s",
            len(only_csv),
            ", ".join(sorted(only_csv)[:10])
            + ("..." if len(only_csv) > 10 else ""),
        )
    if only_dir:
        logger.info(
            "  ⚠️  %d study dirs on disk but NOT in CSV (skipped): %s",
            len(only_dir),
            ", ".join(sorted(only_dir)[:10])
            + ("..." if len(only_dir) > 10 else ""),
        )

    if not matched_ids:
        raise RuntimeError(
            "No matching studies found between CSV and data directory. "
            "Check your --data-dir and --csv-path."
        )

    # Progress bar (optional dependency)
    try:
        from tqdm import tqdm
        progress = tqdm(matched_ids, desc="Running inference", unit="study")
    except ImportError:
        logger.info("Tip: install tqdm for a progress bar (`pip install tqdm`).")
        progress = matched_ids

    study_ids: List[str] = []
    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    intermediates_list: List[Dict[str, float]] = []
    errors: List[str] = []

    start_time = time.time()

    for study_id in progress:
        study_dir = study_dirs[study_id]
        gt = ground_truth[study_id]

        try:
            intermediates = model_predict(str(study_dir), models=models)
            pred_class = triage_from_intermediates(intermediates)

            study_ids.append(study_id)
            y_true_list.append(int(gt["triage_class"]))
            y_pred_list.append(pred_class)
            intermediates_list.append(intermediates)

        except Exception as exc:
            err_msg = f"{study_id}: {exc}"
            errors.append(err_msg)
            if isinstance(progress, type(matched_ids)):
                logger.warning("  ❌ Failed on study %s: %s", study_id, exc)
            continue

    elapsed = time.time() - start_time
    logger.info(
        "Inference complete: %d/%d studies processed in %.1fs (%.2f s/study).",
        len(study_ids), len(matched_ids), elapsed,
        elapsed / max(len(study_ids), 1),
    )

    if errors:
        logger.warning("%d study(s) failed during inference:", len(errors))
        for err in errors[:10]:
            logger.warning("  - %s", err)
        if len(errors) > 10:
            logger.warning("  ... and %d more.", len(errors) - 10)

    if not study_ids:
        raise RuntimeError("All studies failed during inference. Check logs above.")

    return (
        study_ids,
        np.array(y_true_list, dtype=int),
        np.array(y_pred_list, dtype=int),
        intermediates_list,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the leaderboard CLI."""
    project_root = _get_project_root()
    default_models = project_root / "submission" / "models"
    default_data = project_root / "data" / "raw" / "training"
    default_csv = project_root / "Data" / "metadata" / "training_df.csv"
    default_output_csv = project_root / "leaderboard" / "results.csv"
    default_output_json = project_root / "leaderboard" / "metrics.json"

    parser = argparse.ArgumentParser(
        description="🏆 Personal Leaderboard — IAAA 2026 Brain CT Triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leaderboard.evaluate
  python -m leaderboard.evaluate --device cpu
  python -m leaderboard.evaluate --models-dir submission/models --device cuda
  python -m leaderboard.evaluate --output-csv my_results.csv --output-json my_metrics.json
        """,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models,
        help=f"Directory containing trained model weights (default: {default_models})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help=f"Directory with DICOM study subdirectories (default: {default_data})",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv,
        help=f"Path to training_df.csv metadata (default: {default_csv})",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output_csv,
        help=f"Path for per-study results CSV (default: {default_output_csv})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_output_json,
        help=f"Path for metrics JSON (default: {default_output_json})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting CSV and JSON files.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the leaderboard evaluation.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Metrics dictionary.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = _get_project_root()

    print()
    print("=" * 60)
    print("  🏆  PERSONAL LEADERBOARD — IAAA 2026 Brain CT Triage")
    print("=" * 60)
    print(f"  Models dir : {args.models_dir}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  CSV path   : {args.csv_path}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    # ---- Auto-detect CUDA availability (before loading models) --------------
    if args.device == "cuda":
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except (ImportError, AttributeError):
            pass

        if not cuda_available:
            logger.warning(
                "CUDA requested but not available (PyTorch CPU-only or no GPU driver). "
                "Falling back to CPU. Install PyTorch with CUDA for GPU inference."
            )
            args.device = "cpu"

    # ---- Step 1: Load ground truth ------------------------------------------
    from leaderboard.ground_truth import load_study_labels

    logger.info("Step 1/4: Loading ground truth labels ...")
    ground_truth = load_study_labels(args.csv_path)
    logger.info("Loaded ground truth for %d studies.", len(ground_truth))

    # ---- Step 2: Discover study directories ---------------------------------
    logger.info("Step 2/4: Discovering DICOM study directories ...")
    study_dirs = discover_study_dirs(args.data_dir)

    # ---- Step 3: Load models ------------------------------------------------
    logger.info("Step 3/4: Loading models ...")

    try:
        from submission.model import load_models
    except ImportError as exc:
        logger.error(
            "Failed to import submission.model. "
            "Make sure the submission/ folder is intact. Error: %s", exc
        )
        raise

    # Check if model files actually exist (beyond .gitkeep)
    model_subdirs = ["nnunet", "yolo", "mls"]
    for sub in model_subdirs:
        sub_path = args.models_dir / sub
        real_files = [f for f in sub_path.iterdir()
                      if f.is_file() and f.name != ".gitkeep"]
        if not real_files:
            logger.warning(
                "  ⚠️  No model files found in %s/ (only .gitkeep). "
                "Place your trained weights there first.",
                sub_path,
            )

    try:
        models = load_models(str(args.models_dir), device=args.device)
    except Exception as exc:
        error_str = str(exc)

        # Detect CUDA-related errors specifically
        if "cuda" in error_str.lower() and "is_available" in error_str.lower():
            logger.error("CUDA error detected! Possible causes:")
            logger.error("  1. PyTorch is CPU-only → install PyTorch with CUDA:")
            logger.error("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            logger.error("  2. CUDA driver not installed → check with: nvidia-smi")
            logger.error("  3. Run with --device cpu instead: python -m leaderboard.evaluate --device cpu")
        else:
            logger.error("Failed to load models. Expected files:")
            logger.error(
                "  %s/nnunet/  → checkpoint_best.pth + dataset.json + plans.json + ...",
                args.models_dir,
            )
            logger.error("  %s/yolo/    → best.pt", args.models_dir)
            logger.error(
                "  %s/mls/     → slice_selector_best.ckpt + keypoint_best.ckpt",
                args.models_dir,
            )

        logger.error("Original error: %s", exc)
        raise RuntimeError(
            "Model loading failed. Check the error details above."
        ) from exc

    logger.info("Models loaded successfully.")

    # ---- Step 4: Run inference & scoring ------------------------------------
    logger.info("Step 4/4: Running inference on matched studies ...")

    study_ids, y_true, y_pred, intermediates = run_inference(
        study_dirs, models, ground_truth
    )

    # ---- Scoring ------------------------------------------------------------
    from leaderboard.scorer import (
        compute_metrics,
        print_report,
        export_results_csv,
        export_metrics_json,
    )

    patient_ids = [
        ground_truth[sid]["patient_id"]
        for sid in study_ids
    ]

    metrics = compute_metrics(study_ids, y_true, y_pred, intermediates)
    print_report(metrics)

    # ---- Export -------------------------------------------------------------
    if not args.no_export:
        export_results_csv(
            args.output_csv,
            study_ids,
            y_true,
            y_pred,
            patient_ids=patient_ids,
            ground_truth_info=ground_truth,
            intermediates_list=intermediates,
        )
        export_metrics_json(metrics, args.output_json)

    # ---- Final summary ------------------------------------------------------
    qwk = metrics.get("qwk", 0)
    print(f"🏆 Final QWK: {qwk:.4f}")
    print(f"📄 Results CSV: {args.output_csv.resolve()}")
    print(f"📄 Metrics JSON: {args.output_json.resolve()}")
    print("✅ Leaderboard evaluation complete!\n")

    return metrics


if __name__ == "__main__":
    main()
