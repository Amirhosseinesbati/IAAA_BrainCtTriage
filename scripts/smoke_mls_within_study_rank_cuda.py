"""CUDA-only A4 forward/backward VRAM preflight at its exact primary batch size."""

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
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import (
    configure_training_determinism,
    within_study_pair_rank_loss,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4 preflight is CUDA-only; CPU fallback is forbidden")
    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    config = MLSHeatmapConfig.model_validate(payload["training_config"])
    if config.within_study_rank_loss_weight <= 0.0:
        raise ValueError("A4 preflight requires within_study_rank_loss_weight>0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    determinism = configure_training_determinism(config.training_determinism)
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
    primary_images = torch.randn(
        config.batch_size,
        config.input_channels,
        config.image_size,
        config.image_size,
        device=device,
    )
    primary_heatmaps, primary_selector = model.forward_multitask(primary_images)
    primary_loss = primary_heatmaps.square().mean() + primary_selector.square().mean()
    if not torch.isfinite(primary_loss):
        raise FloatingPointError("Non-finite primary CUDA preflight loss")
    primary_loss.backward()

    pair_images = torch.randn(
        2, config.input_channels, config.image_size, config.image_size, device=device,
    )
    pair_masks = torch.ones(2, 3, device=device)
    pair_keypoints = torch.tensor(
        [[[128.0, 64.0], [128.0, 448.0], [188.0, 256.0]],
         [[128.0, 64.0], [128.0, 448.0], [138.0, 256.0]]],
        device=device,
    )
    pair_spacing = torch.full((2,), 0.5, device=device)
    pair_is_target = torch.ones(2, device=device)
    _pair_heatmaps, pair_selector = model.forward_multitask(pair_images)
    pair_loss, pair_parts = within_study_pair_rank_loss(
        pair_selector,
        pair_masks,
        pair_keypoints,
        pair_spacing,
        pair_is_target,
        config,
    )
    if not torch.isfinite(pair_loss) or float(pair_parts["within_study_rank_qualified_pairs"]) != 1.0:
        raise FloatingPointError("Invalid same-study pair CUDA preflight loss")
    (config.within_study_rank_loss_weight * pair_loss).backward()
    optimizer.step()
    result = {
        "status": "ok",
        "compute_policy": "cuda_only_no_cpu_fallback",
        "cuda_device": torch.cuda.get_device_name(0),
        "primary_batch_size": config.batch_size,
        "pair_size": 2,
        "primary_loss": float(primary_loss.detach()),
        "pair_rank_loss": float(pair_loss.detach()),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "training_determinism": determinism,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
