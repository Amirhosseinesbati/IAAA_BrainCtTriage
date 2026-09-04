"""Audit MLS update budgets from frozen manifests without model or image execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--fixed-epoch", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing budget evidence")
    if args.fixed_epoch < 1:
        raise ValueError("Fixed epoch must be positive")

    dataset_root = ROOT / "Data" / "processed" / "mls_multitask_v2"
    labels = dataset_root / "mls_labels_multitask.csv"
    runs = []
    expected_population = None
    for manifest in args.manifest:
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        config = MLSHeatmapConfig.model_validate(payload["training_config"])
        if not config.use_competition_folds or config.dataset_variant != "multitask_v2":
            raise ValueError("Expected immutable competition split and multitask_v2")
        if config.epochs < args.fixed_epoch:
            raise ValueError("Requested audit epoch is outside the run schedule")
        # Constructing these loaders reads only metadata. Never iterate them,
        # instantiate a model, load a checkpoint, or open image/prediction files.
        train_loader, val_loader = create_mls_dataloaders(
            csv_path=str(labels), img_dir=str(dataset_root / "images"),
            img_size=config.image_size, heatmap_size=config.image_size // 4,
            heatmap_sigma=config.heatmap_sigma, batch_size=config.batch_size,
            fold=config.fold, seed=config.seed, use_competition_folds=True,
            augment=False, num_workers=0, include_negatives=True,
            return_selector=True, balanced_sampling=True,
            sampling_mode=config.sampling_mode,
        )
        population = (config.fold, len(train_loader.dataset), len(val_loader.dataset))
        if expected_population is None:
            expected_population = population
        if population != expected_population:
            raise ValueError("Run populations differ; refusing a pooled budget comparison")
        primary_batches = len(train_loader)
        if not train_loader.drop_last or primary_batches != population[1] // config.batch_size:
            raise ValueError("Current loader no longer matches the audited drop-last contract")
        accumulation = config.gradient_accumulation_steps
        updates = math.ceil(primary_batches / accumulation)
        runs.append({
            "run_name": payload["run_name"],
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "fold": config.fold, "seed": config.seed,
            "train_rows": population[1], "validation_rows": population[2],
            "batch_size": config.batch_size, "recorded_num_workers": config.num_workers,
            "gradient_accumulation_steps": accumulation,
            "epochs": config.epochs, "learning_rate": config.learning_rate,
            "primary_batches_per_epoch": primary_batches,
            "optimizer_updates_per_epoch": updates,
            "optimizer_updates_at_fixed_epoch": updates * args.fixed_epoch,
            "processed_primary_draws_at_fixed_epoch": primary_batches * config.batch_size * args.fixed_epoch,
            "rank_weight": config.within_study_rank_loss_weight,
            "rank_detach_backbone": config.within_study_rank_detach_backbone,
            "scheduled_rank_batches_per_epoch": (
                math.ceil(primary_batches / config.within_study_rank_every_n_steps)
                if config.within_study_rank_loss_weight > 0 else 0
            ),
            "rank_creates_separate_optimizer_steps": False,
        })
    result = {
        "schema_version": 1,
        "status": "metadata_audit_completed",
        "compute_policy": "metadata_only_no_model_or_image_execution",
        "scope": "update_counts_derived_from_current_checksum_bound_data_and_source_not_historical_step_telemetry",
        "fixed_epoch": args.fixed_epoch,
        "labels_sha256": sha256(labels),
        "fold_manifest_sha256": sha256(ROOT / "config" / "folds.csv"),
        "dataset_source_sha256": sha256(ROOT / "src/strategies/mls_heatmap/dataset.py"),
        "training_source_sha256": sha256(ROOT / "src/strategies/mls_heatmap/train_multitask.py"),
        "runs": runs,
        "causal_attribution_established": False,
        "promotion_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
