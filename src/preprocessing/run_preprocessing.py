"""
run_preprocessing.py — Integrated preprocessing runner.

Orchestrates all dataset builders (generic NIfTI, nnU-Net, YOLO, MLS)
with validation, progress reporting, and optional selective execution.

Usage:
    python -m src.preprocessing.run_preprocessing --builders nifti,nnunet,yolo,mls
    python -m src.preprocessing.run_preprocessing --builders nifti
    python -m src.preprocessing.run_preprocessing --builders nnunet
    python -m src.preprocessing.run_preprocessing --builders yolo,mls
    python -m src.preprocessing.run_preprocessing --all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import (
    RAW_TRAINING_DIR, RAW_ANNOTATIONS_DIR,
    ICH_NIFTI_DIR, NNUNET_RAW_DIR, YOLO_DIR, MLS_DIR,
)
from src.preprocessing.builders.nifti_builder import NiftiDatasetBuilder
from src.preprocessing.builders.nnunet_builder import NNUnetDatasetBuilder
from src.preprocessing.builders.yolo_builder import YoloDatasetBuilder
from src.preprocessing.builders.mls_builder import MlsDatasetBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocessing")


def validate_directories():
    """Check that required input directories exist."""
    issues = []
    if not RAW_TRAINING_DIR.exists():
        issues.append(f"❌ DICOM training dir not found: {RAW_TRAINING_DIR}")
    if not RAW_ANNOTATIONS_DIR.exists():
        issues.append(f"❌ Annotations dir not found: {RAW_ANNOTATIONS_DIR}")

    if issues:
        for msg in issues:
            logger.error(msg)
        return False
    return True


def run_nifti_builder():
    """Build generic NIfTI dataset (strategy-agnostic)."""
    logger.info("=" * 50)
    logger.info("  Building Generic NIfTI Dataset (ICH_NIFTI_DIR)")
    logger.info("=" * 50)
    t0 = time.time()

    builder = NiftiDatasetBuilder()
    builder.build()

    elapsed = time.time() - t0
    logger.info(f"  ⏱️  NIfTI build completed in {elapsed:.1f}s")

    # Check output
    nii_files = list(ICH_NIFTI_DIR.rglob("images/*.nii.gz"))
    logger.info(f"  📦 Output: {len(nii_files)} NIfTI files in {ICH_NIFTI_DIR}")
    return len(nii_files) > 0


def run_nnunet_builder():
    """Build nnU-Net dataset (uses NNUNET_RAW_DIR)."""
    logger.info("=" * 50)
    logger.info("  Building nnU-Net Dataset")
    logger.info("=" * 50)
    t0 = time.time()

    builder = NNUnetDatasetBuilder()
    builder.build()

    elapsed = time.time() - t0
    logger.info(f"  ⏱️  nnU-Net build completed in {elapsed:.1f}s")

    # Check output
    nii_files = list(NNUNET_RAW_DIR.rglob("*_0000.nii.gz"))
    logger.info(f"  📦 Output: {len(nii_files)} NIfTI files in {NNUNET_RAW_DIR}")
    return len(nii_files) > 0


def run_yolo_builder():
    """Build YOLO fracture dataset."""
    logger.info("=" * 50)
    logger.info("  Building YOLO Fracture Dataset")
    logger.info("=" * 50)
    t0 = time.time()

    builder = YoloDatasetBuilder()
    builder.build()

    elapsed = time.time() - t0
    logger.info(f"  ⏱️  YOLO build completed in {elapsed:.1f}s")

    # Check output
    jpg_files = list(YOLO_DIR.rglob("images/**/*.jpg"))
    logger.info(f"  📦 Output: {len(jpg_files)} images in {YOLO_DIR}")
    return len(jpg_files) > 0


def run_mls_builder():
    """Build MLS dataset."""
    logger.info("=" * 50)
    logger.info("  Building MLS Dataset")
    logger.info("=" * 50)
    t0 = time.time()

    builder = MlsDatasetBuilder()
    builder.build()

    elapsed = time.time() - t0
    logger.info(f"  ⏱️  MLS build completed in {elapsed:.1f}s")

    # Check output
    png_files = list(MLS_DIR.rglob("images/*.png"))
    csv_path = MLS_DIR / "mls_labels.csv"
    logger.info(f"  📦 Output: {len(png_files)} images + {csv_path.exists()}")
    return len(png_files) > 0


def main():
    parser = argparse.ArgumentParser(
        description="IAAA Brain CT Triage — Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--builders",
        type=str,
        default="all",
        help="Comma-separated list: nifti,nnunet,yolo,mls (default: all)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip input directory validation",
    )
    args = parser.parse_args()

    # Determine which builders to run
    if args.builders == "all":
        builders_to_run = ["nifti", "nnunet", "yolo", "mls"]
    else:
        builders_to_run = [b.strip().lower() for b in args.builders.split(",")]

    logger.info(f"🧠 IAAA Brain CT Triage — Preprocessing Pipeline")
    logger.info(f"  Builders: {', '.join(builders_to_run)}")
    logger.info(f"  DICOM:    {RAW_TRAINING_DIR}")
    logger.info(f"  JSON:     {RAW_ANNOTATIONS_DIR}")

    # Validate input directories
    if not args.no_validate and not validate_directories():
        logger.error("Input validation failed. Aborting.")
        sys.exit(1)

    # Run selected builders
    results = {}
    for builder_name in builders_to_run:
        if builder_name == "nifti":
            results["nifti"] = run_nifti_builder()
        elif builder_name == "nnunet":
            results["nnunet"] = run_nnunet_builder()
        elif builder_name == "yolo":
            results["yolo"] = run_yolo_builder()
        elif builder_name == "mls":
            results["mls"] = run_mls_builder()
        else:
            logger.warning(f"Unknown builder: {builder_name}. Skipping.")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("  BUILD SUMMARY")
    logger.info("=" * 50)
    all_ok = True
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        logger.info(f"  {status} {name}: {'Success' if ok else 'Failed'}")
        all_ok = all_ok and ok

    if all_ok and results:
        logger.info("\n🎉 All preprocessing completed successfully!")
    elif results:
        logger.warning("\n⚠️  Some preprocessing steps did not complete.")


if __name__ == "__main__":
    main()
