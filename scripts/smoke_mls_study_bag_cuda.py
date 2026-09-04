"""CUDA-only worst-positive-bag forward/backward preflight for MLS A3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import (
    collate_mls_study_bag,
    create_mls_dataloaders,
    MLSPositiveStudyBagDataset,
)
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import (
    configure_training_determinism,
    study_bag_selection_loss,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Study-bag preflight is CUDA-only; CPU fallback is forbidden")
    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    config = MLSHeatmapConfig.model_validate(payload["training_config"])
    if config.study_bag_loss_weight <= 0.0:
        raise ValueError("Study-bag preflight requires study_bag_loss_weight>0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    determinism = configure_training_determinism(config.training_determinism)
    dataset_root = PROJECT_ROOT / "Data" / "processed" / "mls_multitask_v2"
    train_loader, _val_loader = create_mls_dataloaders(
        csv_path=str(dataset_root / "mls_labels_multitask.csv"),
        img_dir=str(dataset_root / "images"),
        img_size=config.image_size,
        heatmap_size=config.image_size // 4,
        heatmap_sigma=config.heatmap_sigma,
        batch_size=config.batch_size,
        val_split=config.val_split,
        augment=True,
        rotation_deg=config.rotation_deg,
        translation=config.translation,
        intensity_jitter_scale=config.intensity_jitter,
        augment_prob=config.augment_prob,
        num_workers=0,
        seed=config.seed,
        fold=config.fold,
        use_competition_folds=config.use_competition_folds,
        include_negatives=True,
        return_selector=True,
        balanced_sampling=True,
        sampling_mode=config.sampling_mode,
        deterministic_workers=True,
    )
    bag_dataset = MLSPositiveStudyBagDataset(train_loader.dataset)
    largest_index = max(range(len(bag_dataset)), key=lambda index: len(bag_dataset._bags[index]))
    batch = collate_mls_study_bag([bag_dataset[largest_index]])
    images, _targets, masks, keypoints, spacing, is_target, study_mls, _study_ids = batch
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    keypoints = keypoints.to(device, non_blocking=True)
    spacing = spacing.to(device, non_blocking=True)
    is_target = is_target.to(device, non_blocking=True)
    study_mls = study_mls.to(device, non_blocking=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = HRNetHeatmapModel(
        backbone_name=config.backbone,
        in_channels=config.input_channels,
        num_keypoints=3,
        pretrained=False,
        head_dropout=config.head_dropout,
        use_selector=True,
        selector_head_mode=config.selector_head_mode,
        use_ordinal_aux_head=False,
    ).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    with torch.amp.autocast("cuda", enabled=config.use_amp):
        heatmaps, selector = model.forward_multitask(images)
        loss, parts = study_bag_selection_loss(
            heatmaps, selector, masks, keypoints, spacing, is_target, study_mls, config,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite CUDA study-bag preflight loss")
    loss.backward()
    optimizer.step()
    result = {
        "compute_policy": "cuda_only_no_cpu_fallback",
        "cuda_device": torch.cuda.get_device_name(0),
        "largest_positive_bag_slices": int(images.shape[0]),
        "positive_study_bags": int(len(bag_dataset)),
        "loss": float(loss.detach()),
        "study_bag_regression": float(parts["study_bag_regression"]),
        "study_bag_threshold": float(parts["study_bag_threshold"]),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "training_determinism": determinism,
        "status": "ok",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
