"""
evaluate.py — Main entry point for the personal leaderboard.

Evaluates trained models (placed in submission/models/) against ground truth
from Data/metadata/training_df.csv using Quadratic Weighted Kappa (QWK).

Supports multiple ICH strategies via the --ich-strategy flag.

Usage:
    python -m leaderboard.evaluate                                              # default: nnunet
    python -m leaderboard.evaluate --ich-strategy smp                           # SMP
    python -m leaderboard.evaluate --ich-strategy monai --device cpu            # MONAI on CPU
    python -m leaderboard.evaluate --ich-strategy yolo_seg --models-dir my_models
    python -m leaderboard.evaluate --list-strategies                            # show all
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Pre-import pyarrow to prevent WinError 6714 crash during nnU-Net import chain.
# When pyarrow is cached in sys.modules, subsequent pandas→pyarrow imports
# triggered by nnU-Net will skip the faulty _fill_cache scan on Windows.
import pyarrow  # noqa: F401

import numpy as np
import torch

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
    """Add project root to sys.path so submission/ package is importable."""
    project_root = _get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------
def _load_available_strategies():
    """Import and return (list_strategies, STRATEGY_NAMES)."""
    try:
        from src.strategies import list_strategies, STRATEGY_NAMES
        return list_strategies, STRATEGY_NAMES
    except ImportError:
        return None, []


def _resolve_default_ich_strategy() -> str:
    """Get default strategy from config, falling back to 'nnunet'."""
    try:
        from src.config import ICH_DEFAULT_STRATEGY
        return ICH_DEFAULT_STRATEGY
    except (ImportError, AttributeError):
        return "nnunet"


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
# CUDA diagnostics
# ---------------------------------------------------------------------------
def _check_cuda(device_arg: str) -> str:
    """Validate CUDA availability and return the actual device to use."""
    if device_arg == "cpu":
        return "cpu"

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("✅ GPU detected: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
            return "cuda"
        else:
            logger.warning(
                "CUDA requested but PyTorch has no GPU driver / no CUDA build.\n"
                "  Possible fixes:\n"
                "    1. Install PyTorch WITH CUDA:  uv pip install torch torchvision --index https://download.pytorch.org/whl/cu118\n"
                "    2. GTX 1660 Ti supports CUDA 11.x (NOT 12.x). Check pyproject.toml:\n"
                "       Change 'pytorch-cu126' to 'pytorch-cu118' in [tool.uv.sources].\n"
                "    3. Or run with --device cpu (slower but works).\n"
                "  Falling back to CPU."
            )
            return "cpu"
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not check CUDA (%s). Falling back to CPU.", exc)
        return "cpu"


# ---------------------------------------------------------------------------
# Runtime estimation
# ---------------------------------------------------------------------------
def _estimate_runtime(ich_strategy: str, n_studies: int, device: str) -> None:
    """Log an estimated runtime based on strategy + device."""
    # Rough per-study estimates (seconds) — calibrated from typical runs
    if device == "cuda":
        estimates = {"nnunet": 3, "smp": 2, "monai": 8, "yolo_seg": 1}
    else:
        estimates = {"nnunet": 8, "smp": 10, "monai": 120, "yolo_seg": 5}

    sec_per = estimates.get(ich_strategy, 30)
    total_sec = sec_per * n_studies
    if total_sec < 60:
        logger.info("⏱️  Estimated runtime: ~%d seconds (%s, %d studies, %ds/study)",
                     total_sec, device, n_studies, sec_per)
    elif total_sec < 3600:
        logger.info("⏱️  Estimated runtime: ~%d minutes (%s, %d studies, %ds/study)",
                     total_sec // 60, device, n_studies, sec_per)
    else:
        logger.info("⏱️  Estimated runtime: ~%d hours (%s, %d studies, %ds/study). "
                     "Consider using --device cuda or a faster strategy.",
                     total_sec // 3600, device, n_studies, sec_per)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------
def _get_strategy_model_subdirs(strategy_name: str) -> list[str]:
    """Return the expected model subdirectories for a given ICH strategy."""
    ich_map = {
        "nnunet": "nnunet",
        "smp": "smp",
        "monai": "monai",
        "yolo_seg": "yolo_seg",
    }
    ich_dir = ich_map.get(strategy_name, "nnunet")
    # All strategies share the same yolo/, mls/ and mls_heatmap/ subdirs
    return [ich_dir, "yolo", "mls", "mls_heatmap"]


def validate_model_dirs(models_dir: Path, ich_strategy: str) -> None:
    """Check that required model subdirectories exist and contain weights.

    Logs warnings for missing subdirs — does NOT raise (models may still load).
    """
    subdirs = _get_strategy_model_subdirs(ich_strategy)
    logger.info("Checking model directories for strategy '%s' ...", ich_strategy)

    for sub in subdirs:
        sub_path = models_dir / sub
        if not sub_path.is_dir():
            logger.warning("  ⚠️  Missing directory: %s/", sub_path)
            continue

        real_files = [
            f for f in sub_path.iterdir()
            if f.is_file() and f.name != ".gitkeep"
        ]
        if not real_files:
            logger.warning(
                "  ⚠️  No model files in %s/ (only .gitkeep). "
                "Place your trained weights there first.",
                sub_path,
            )
        else:
            logger.info("  ✅ %s/ — %d file(s)", sub, len(real_files))


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

        finally:
            # ── Clean GPU memory after each study ──────────────────────────
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
# Compare-all: run every ICH strategy and produce a comparison table
# ---------------------------------------------------------------------------

ALL_STRATEGIES = ["nnunet", "smp", "monai", "yolo_seg"]


def _run_single_strategy(
    strategy_name: str,
    models_dir: Path,
    device: str,
    study_dirs: Dict[str, Path],
    ground_truth: Dict[str, Dict[str, Any]],
) -> dict:
    """Run the full leaderboard evaluation for a single ICH strategy.

    Returns a metrics dict plus the elapsed time.
    """
    from submission.model import load_models as _load
    from submission.triage import triage_from_intermediates
    from leaderboard.scorer import compute_metrics

    logger.info("─" * 50)
    logger.info("🔄  Evaluating strategy: %s", strategy_name)
    logger.info("─" * 50)

    start_t = time.time()

    models = _load(str(models_dir), device=device, ich_strategy=strategy_name)
    logger.info("MLS model: %s (auto-detected)",
                models.get("mls_mode", "legacy"))

    # Inference
    csv_ids = set(ground_truth.keys())
    dir_ids = set(study_dirs.keys())
    matched_ids = sorted(csv_ids & dir_ids)

    y_true_list, y_pred_list, inter_list = [], [], []
    errors = []

    try:
        from tqdm import tqdm
        progress = tqdm(matched_ids, desc=f"  {strategy_name}", unit="study")
    except ImportError:
        progress = matched_ids

    for study_id in progress:
        try:
            inter = _predict_from_submission(str(study_dirs[study_id]), models)
            y_true_list.append(int(ground_truth[study_id]["triage_class"]))
            y_pred_list.append(triage_from_intermediates(inter))
            inter_list.append(inter)
        except Exception as exc:
            errors.append(f"{study_id}: {exc}")
        finally:
            # Clear GPU memory after each study to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - start_t

    if not y_true_list:
        logger.error("  ❌ %s: all %d studies failed!", strategy_name, len(matched_ids))
        return {"strategy": strategy_name, "error": "All studies failed", "time_s": elapsed}

    y_true = np.array(y_true_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)

    metrics = compute_metrics(matched_ids[:len(y_true)], y_true, y_pred, inter_list)
    metrics["time_s"] = round(elapsed, 1)
    metrics["n_failed"] = len(errors)

    if errors:
        logger.warning("  ⚠️  %s: %d/%d studies failed", strategy_name, len(errors), len(matched_ids))

    logger.info("  %s → QWK = %.4f (%.1fs)", strategy_name, metrics.get("qwk", 0), elapsed)
    return metrics


def _predict_from_submission(study_dir: str, models: dict) -> dict:
    """Thin wrapper around submission.model.predict."""
    from submission.model import predict as model_predict
    return model_predict(study_dir, models=models)


def _print_comparison_table(results: list[dict]) -> None:
    """Pretty-print a side-by-side comparison of all strategies."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "  📊  STRATEGY COMPARISON — IAAA 2026 Brain CT Triage".ljust(68) + "║")
    print("╠" + "═" * 70 + "╣")
    print("║  {:<12s}  {:>8s}  {:>8s}  {:>6s}  {:>8s}  {:>6s}  {:>10s}║".format(
        "Strategy", "QWK", "Accuracy", "F1-N", "F1-E", "F1-C", "Time (s)",
    ))
    print("║" + "  " + "─" * 66 + "  ║")

    best_qwk = -1.0
    best_name = ""
    for r in results:
        if "error" in r:
            continue
        qwk = r.get("qwk", 0)
        if qwk > best_qwk:
            best_qwk = qwk
            best_name = r["strategy"]
        acc = r.get("accuracy", 0)
        pc = r.get("per_class", {})
        f1_n = pc.get("Normal", {}).get("f1_score", 0)
        f1_e = pc.get("Emergency", {}).get("f1_score", 0)
        f1_c = pc.get("Critical", {}).get("f1_score", 0)
        t = r.get("time_s", 0)
        marker = "👑" if r["strategy"] == best_name else "  "
        print("║ {}{:<10s}  {:>8.4f}  {:>8.4f}  {:>6.3f}  {:>6.3f}  {:>6.3f}  {:>7.1f}s  ║".format(
            marker, r["strategy"], qwk, acc, f1_n, f1_e, f1_c, t,
        ))

    print("╠" + "═" * 70 + "╣")
    print("║" + f"  👑 Best QWK: {best_name} = {best_qwk:.4f}".ljust(68) + "║")
    print("╚" + "═" * 70 + "╝")
    print()


def _run_compare_all(
    models_dir: Path,
    device: str,
    study_dirs: Dict[str, Path],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all ICH strategies and generate a comparison report."""
    print()
    print("=" * 70)
    print("  🧪 COMPARE-ALL MODE: Evaluating every ICH strategy")
    print("=" * 70)
    print(f"  Models dir : {models_dir}")
    print(f"  Data dir   : {study_dirs}")
    print(f"  Device     : {device}")
    print("=" * 70)
    print()

    results = []
    for strategy in ALL_STRATEGIES:
        # Check if model directory exists for this strategy
        strategy_dir = models_dir / strategy
        if not strategy_dir.is_dir():
            logger.warning("Skipping '%s' — directory not found: %s", strategy, strategy_dir)
            continue
        real_files = [f for f in strategy_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
        if not real_files:
            logger.warning("Skipping '%s' — no model files in %s", strategy, strategy_dir)
            continue

        try:
            metrics = _run_single_strategy(strategy, models_dir, device, study_dirs, ground_truth)
            results.append(metrics)
        except Exception as exc:
            logger.error("  ❌ %s failed completely: %s", strategy, exc)
            results.append({"strategy": strategy, "error": str(exc)})

    _print_comparison_table(results)

    # Find best QWK
    best_qwk_val = max(
        (r.get("qwk", -1) for r in results if "error" not in r),
        default=-1.0,
    )
    return {"comparison": results, "best_qwk": best_qwk_val}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the leaderboard CLI."""
    project_root = _get_project_root()
    default_models = project_root / "submission" / "models"
    default_data = project_root / "Data" / "raw" / "training"
    default_csv = project_root / "Data" / "metadata" / "training_df.csv"
    default_output_csv = project_root / "leaderboard" / "results.csv"
    default_output_json = project_root / "leaderboard" / "metrics.json"
    default_strategy = _resolve_default_ich_strategy()

    parser = argparse.ArgumentParser(
        description="🏆 Personal Leaderboard — IAAA 2026 Brain CT Triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
	Examples:
	  python -m leaderboard.evaluate                                              # nnU-Net default
	  python -m leaderboard.evaluate --ich-strategy smp                           # SMP
	  python -m leaderboard.evaluate --ich-strategy monai --device cpu            # MONAI on CPU
	  python -m leaderboard.evaluate --compare-all                                # ALL strategies side-by-side
	  python -m leaderboard.evaluate --list-strategies                            # list all strategies
	  python -m leaderboard.evaluate --ich-strategy yolo_seg --models-dir my_experiments/models
	        """,
    )
    parser.add_argument(
        "--ich-strategy",
        type=str,
        default=default_strategy,
        help=f"ICH segmentation strategy (default: {default_strategy}). "
             f"Use --list-strategies to see available options.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available ICH strategies and exit.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run ALL available ICH strategies and compare side-by-side.",
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
    _add_submission_to_path()
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = _get_project_root()

    # ── Handle --list-strategies ──────────────────────────────────────────
    if args.list_strategies:
        list_fn, names = _load_available_strategies()
        if list_fn is None or not names:
            print("\n⚠️   No strategies found. Make sure 'src/strategies/' is intact.")
            print("    Try importing: from src.strategies import list_strategies\n")
        else:
            print("\n📋 Available ICH strategies:")
            for s in list_fn():
                print(f"   • {s['name']:12s} — {s['display_name']}")
                print(f"     {s['description'][:100]}...")
                print()
        return {"strategy_list": list_fn() if list_fn else []}

    # ── Handle --compare-all ──────────────────────────────────────────────
    if args.compare_all:
        logger.info("🧪 COMPARE-ALL mode — will run every available ICH strategy.")

        # Load ground truth + discover studies
        from leaderboard.ground_truth import load_study_labels
        ground_truth = load_study_labels(args.csv_path)
        study_dirs = discover_study_dirs(args.data_dir)

        args.device = _check_cuda(args.device)
        _estimate_runtime("smp", len(study_dirs), args.device)  # ballpark

        return _run_compare_all(args.models_dir, args.device, study_dirs, ground_truth)

    # ── Validate ich_strategy ─────────────────────────────────────────────
    valid_strategies = ["nnunet", "smp", "monai", "yolo_seg"]
    if args.ich_strategy not in valid_strategies:
        logger.error(
            "Unknown ICH strategy '%s'. Choose from: %s. "
            "Use --list-strategies for details.",
            args.ich_strategy, ", ".join(valid_strategies),
        )
        sys.exit(1)

    print()
    print("=" * 60)
    print("  🏆  PERSONAL LEADERBOARD — IAAA 2026 Brain CT Triage")
    print("=" * 60)
    print(f"  ICH strategy : {args.ich_strategy}")
    print(f"  Models dir   : {args.models_dir}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  CSV path     : {args.csv_path}")
    print(f"  Device       : {args.device}")
    print("=" * 60)

    # ---- Auto-detect CUDA --------------------------------------------------
    args.device = _check_cuda(args.device)

    # ---- Step 1: Load ground truth ------------------------------------------
    from leaderboard.ground_truth import load_study_labels

    logger.info("Step 1/4: Loading ground truth labels ...")
    ground_truth = load_study_labels(args.csv_path)
    logger.info("Loaded ground truth for %d studies.", len(ground_truth))

    # ---- Step 2: Discover study directories ---------------------------------
    logger.info("Step 2/4: Discovering DICOM study directories ...")
    study_dirs = discover_study_dirs(args.data_dir)

    # ---- Runtime estimate ---------------------------------------------------
    _estimate_runtime(args.ich_strategy, len(study_dirs), args.device)

    # ---- Step 3: Load models ------------------------------------------------
    logger.info("Step 3/4: Loading models ...")

    # Validate model directories
    validate_model_dirs(args.models_dir, args.ich_strategy)

    try:
        from submission.model import load_models
    except ImportError as exc:
        logger.error(
            "Failed to import submission.model. "
            "Make sure the submission/ folder is intact. Error: %s", exc
        )
        raise

    try:
        models = load_models(
            str(args.models_dir),
            device=args.device,
            ich_strategy=args.ich_strategy,
        )
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
            subdirs = _get_strategy_model_subdirs(args.ich_strategy)
            for sub in subdirs:
                logger.error("  %s/%s/ → weights file(s)", args.models_dir, sub)

        logger.error("Original error: %s", exc)
        raise RuntimeError(
            "Model loading failed. Check the error details above."
        ) from exc

    logger.info(
        "Models loaded successfully (ICH strategy: %s).",
        models.get("ich_strategy", args.ich_strategy),
    )
    logger.info(
        "MLS model: %s (auto-detected — 'heatmap' if mls_heatmap_best.pth "
        "exists in %s/mls_heatmap/, else 'legacy').",
        models.get("mls_mode", "legacy"), args.models_dir,
    )

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
    print(f"🏆 Final QWK: {qwk:.4f}  (ICH strategy: {args.ich_strategy})")
    print(f"📄 Results CSV: {args.output_csv.resolve()}")
    print(f"📄 Metrics JSON: {args.output_json.resolve()}")
    print("✅ Leaderboard evaluation complete!\n")

    return metrics


if __name__ == "__main__":
    main()
