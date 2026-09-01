"""CUDA-only forward/backward VRAM smoke test for MLS HRNet variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.mls_heatmap.model import HRNetHeatmapModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="hrnet_w32")
    parser.add_argument("--batches", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    torch.empty(1, device=device)
    results: list[dict] = []
    for batch_size in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            model = HRNetHeatmapModel(
                backbone_name=args.backbone, in_channels=3, num_keypoints=3,
                pretrained=False, head_dropout=0.1, use_selector=True,
            ).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
            inputs = torch.randn(
                batch_size, 3, args.image_size, args.image_size, device=device
            )
            heatmaps, selector = model.forward_multitask(inputs)
            loss = heatmaps.square().mean() + selector.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            results.append({
                "batch_size": batch_size,
                "finite": bool(torch.isfinite(loss)),
                "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "status": "ok",
            })
            del loss, heatmaps, selector, inputs, optimizer, model
        except torch.cuda.OutOfMemoryError as exc:
            results.append({
                "batch_size": batch_size,
                "status": "oom",
                "error": str(exc).splitlines()[0],
            })
        finally:
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
