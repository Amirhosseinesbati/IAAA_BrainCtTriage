"""CUDA-only forward/backward and historical-loader smoke for MLS A1."""

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
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model
from src.strategies.mls_heatmap.train_multitask import multitask_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda")
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = MLSHeatmapConfig.model_validate(payload["training_config"])
    if not config.use_ordinal_aux_head:
        raise ValueError("Smoke config must enable use_ordinal_aux_head")

    model = HRNetHeatmapModel(
        backbone_name=config.backbone,
        in_channels=config.input_channels,
        num_keypoints=3,
        pretrained=False,
        head_dropout=config.head_dropout,
        use_selector=True,
        selector_head_mode=config.selector_head_mode,
        use_ordinal_aux_head=True,
    ).to(device).train()
    images = torch.randn(1, config.input_channels, config.image_size, config.image_size, device=device)
    heatmaps, selector, ordinal = model.forward_multitask_extended(images)
    if ordinal is None or tuple(ordinal.shape) != (1, 3):
        raise AssertionError(f"Unexpected ordinal shape: {None if ordinal is None else tuple(ordinal.shape)}")
    targets = torch.zeros_like(heatmaps)
    masks = torch.ones(1, 3, device=device)
    keypoints = torch.tensor(
        [[[80.0, 180.0], [430.0, 180.0], [255.0, 190.0]]], device=device,
    )
    total, parts = multitask_loss(
        heatmaps, selector, targets, masks, keypoints,
        torch.tensor([0.5], device=device),
        torch.tensor([1.0], device=device),
        torch.tensor([5.0], device=device),
        config,
        ordinal_logits=ordinal,
    )
    total.backward()
    ordinal_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.ordinal_aux_head.parameters()
        if parameter.grad is not None
    )
    if not torch.isfinite(total) or ordinal_grad <= 0.0:
        raise AssertionError("Ordinal CUDA loss/gradient smoke failed")

    historical_model, historical_config = load_multitask_model(
        args.historical_checkpoint, device,
    )
    historical_outputs = historical_model.forward_multitask(images)
    if len(historical_outputs) != 2 or historical_config.use_ordinal_aux_head:
        raise AssertionError("Historical two-output checkpoint contract changed")
    print(json.dumps({
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "loss": float(total.detach()),
        "ordinal_loss": float(parts["ordinal_head"]),
        "ordinal_gradient_l1": ordinal_grad,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "historical_checkpoint_compatible": True,
    }))


if __name__ == "__main__":
    main()
