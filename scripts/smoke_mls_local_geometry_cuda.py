"""A6 synthetic full-loss optimizer step; CUDA only, no patient data."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_a2_fold0_resource_screen import _atomic_json
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train import differentiable_mls_mm
from src.strategies.mls_heatmap.train_multitask import configure_training_determinism, multitask_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite preflight evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    config = MLSHeatmapConfig.model_validate(yaml.safe_load(args.manifest.read_text())["training_config"])
    if config.training_geometry_decoder != "local_softargmax" or config.batch_size != 5:
        raise ValueError("Expected preregistered A6 configuration")
    determinism = configure_training_determinism(config.training_determinism)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()
    model = HRNetHeatmapModel(backbone_name=config.backbone, in_channels=config.input_channels, pretrained=False, head_dropout=config.head_dropout, use_selector=True, selector_head_mode=config.selector_head_mode).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    images = torch.randn(5, 3, config.image_size, config.image_size, device="cuda")
    heatmaps, selector = model.forward_multitask(images)
    keypoints = heatmaps.new_tensor([[[256., 64.], [256., 448.], [268., 256.]]]).repeat(5, 1, 1)
    spacing = heatmaps.new_full((5,), .5)
    is_target = heatmaps.new_tensor([1., 1., 1., 0., 0.])
    masks = is_target[:, None].expand(-1, 3)
    height, width = heatmaps.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing="ij")
    centers = keypoints / keypoints.new_tensor([config.image_size / width, config.image_size / height])
    targets = torch.exp(-((x - centers[:, :, 0, None, None])**2 + (y - centers[:, :, 1, None, None])**2) / (2 * config.heatmap_sigma**2)) * masks[:, :, None, None]
    loss, parts = multitask_loss(heatmaps, selector, targets, masks, keypoints, spacing, is_target, differentiable_mls_mm(keypoints, spacing), config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite full multitask loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise FloatingPointError("Missing/nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite updated parameters")
    torch.cuda.synchronize()
    result = {"status": "ok", "compute_policy": "cuda_only_no_cpu_fallback", "cuda_device": torch.cuda.get_device_name(), "batch_size": 5, "training_geometry_decoder": config.training_geometry_decoder, "local_softargmax_radius": config.local_softargmax_radius, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "loss": float(loss.detach()), "loss_parts": {k: float(v) for k,v in parts.items()}, "gradient_norm": float(norm), "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, "training_determinism": determinism, "synthetic_only": True, "optimizer_steps": 1}
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
