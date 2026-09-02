"""Strict CUDA smoke test for the dual-head MLS selector implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import multitask_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Dual-selector smoke is CUDA-only; CPU fallback is forbidden")

    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    config = MLSHeatmapConfig.model_validate(payload["training_config"])
    if config.selector_head_mode != "dual":
        raise ValueError("Smoke manifest must request selector_head_mode=dual")

    device = torch.device("cuda:0")
    model = HRNetHeatmapModel(
        backbone_name=config.backbone,
        in_channels=config.input_channels,
        num_keypoints=3,
        pretrained=False,
        head_dropout=config.head_dropout,
        use_selector=True,
        selector_head_mode=config.selector_head_mode,
    ).to(device).train()
    torch.cuda.reset_peak_memory_stats(device)

    images = torch.zeros(
        1, config.input_channels, config.image_size, config.image_size, device=device,
    )
    heatmaps, selector_logits = model.forward_multitask(images)
    targets = torch.zeros_like(heatmaps)
    targets[:, :, heatmaps.shape[-2] // 2, heatmaps.shape[-1] // 2] = 1.0
    masks = torch.ones(1, 3, device=device)
    keypoints = torch.tensor(
        [[[200.0, 240.0], [256.0, 240.0], [256.0, 300.0]]], device=device,
    )
    spacing = torch.tensor([0.5], device=device)
    is_target = torch.tensor([1.0], device=device)
    study_mls = torch.tensor([28.0], device=device)
    loss, parts = multitask_loss(
        heatmaps, selector_logits, targets, masks, keypoints, spacing,
        is_target, study_mls, config,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite CUDA loss in dual-selector smoke")
    loss.backward()
    selector_gradient = model.selector_head[-1].weight.grad
    if selector_gradient is None or not torch.isfinite(selector_gradient).all():
        raise FloatingPointError("Missing/non-finite dual-selector CUDA gradient")
    if tuple(selector_logits.shape) != (1, 2):
        raise AssertionError(f"Unexpected dual-selector shape: {tuple(selector_logits.shape)}")
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Model parameters left CUDA during smoke")

    result = {
        "compute_policy": "cuda_only_no_cpu_fallback",
        "device": torch.cuda.get_device_name(0),
        "selector_logits_shape": list(selector_logits.shape),
        "loss": float(loss.detach()),
        "selector_loss": float(parts["selector"]),
        "presence_loss": float(parts["selector_presence"]),
        "peak_loss": float(parts["selector_peak"]),
        "both_selector_rows_have_gradient": bool(
            (selector_gradient.abs().sum(dim=1) > 0).all().item()
        ),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 2**30),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
