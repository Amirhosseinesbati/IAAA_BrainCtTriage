"""Average compatible MLS checkpoint tensors strictly on CUDA.

Checkpoint deserialization and file writing are I/O, but every floating-point
tensor arithmetic operation is guarded to run on a CUDA device. The script is
intended for neighbouring snapshots from one training trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _weighted_path(value: str) -> tuple[Path, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be PATH=WEIGHT")
    raw_path, raw_weight = value.rsplit("=", 1)
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be numeric") from exc
    if weight <= 0:
        raise argparse.ArgumentTypeError("weight must be positive")
    return Path(raw_path), weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", type=_weighted_path, required=True,
        help="Repeat PATH=WEIGHT for compatible snapshots.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if len(args.checkpoint) < 2:
        raise ValueError("At least two checkpoints are required")
    paths = [path.resolve() for path, _ in args.checkpoint]
    if len(paths) != len(set(paths)):
        raise ValueError("Checkpoint paths must be unique")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU tensor averaging is forbidden")

    device = torch.device("cuda:0")
    weights = torch.tensor(
        [weight for _, weight in args.checkpoint], device=device, dtype=torch.float64
    )
    weights = weights / weights.sum()
    if weights.device.type != "cuda":
        raise RuntimeError("Weight normalization did not run on CUDA")

    torch.cuda.reset_peak_memory_stats(device)
    averaged_state: dict[str, torch.Tensor] = {}
    reference_config = None
    reference_keys = None
    reference_metadata = None
    source_metadata = []

    for source_index, path in enumerate(paths):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"]
        if not state:
            raise ValueError(f"Empty model_state_dict: {path}")
        if any(tensor.device.type != "cuda" for tensor in state.values()):
            raise RuntimeError(f"Checkpoint tensor was not loaded on CUDA: {path}")

        config = checkpoint.get("config")
        if source_index == 0:
            reference_config = config
            reference_keys = list(state)
            reference_metadata = checkpoint
        else:
            if config != reference_config:
                raise ValueError(f"Checkpoint configs differ: {path}")
            if list(state) != reference_keys:
                raise ValueError(f"State-dict keys/order differ: {path}")

        scalar_weight = weights[source_index].to(dtype=torch.float32)
        for key, tensor in state.items():
            if source_index == 0:
                if tensor.is_floating_point() or tensor.is_complex():
                    averaged_state[key] = tensor.detach().clone().mul_(
                        scalar_weight.to(dtype=tensor.dtype)
                    )
                else:
                    averaged_state[key] = tensor.detach().clone()
            else:
                target = averaged_state[key]
                if tensor.shape != target.shape or tensor.dtype != target.dtype:
                    raise ValueError(f"Tensor mismatch for {key} in {path}")
                if tensor.is_floating_point() or tensor.is_complex():
                    target.add_(tensor, alpha=float(weights[source_index].item()))
                elif key.endswith("num_batches_tracked"):
                    # This counter naturally increases between neighbouring
                    # snapshots and has no averaging meaning during eval.
                    target.copy_(torch.maximum(target, tensor))
                elif not torch.equal(tensor, target):
                    raise ValueError(
                        f"Non-floating state differs for {key}; refusing implicit choice"
                    )

        source_metadata.append({
            "path": str(path),
            "epoch": checkpoint.get("epoch"),
            "weight": float(weights[source_index].item()),
            "size_bytes": path.stat().st_size,
            "mlflow_run_id": checkpoint.get("mlflow_run_id"),
        })
        del state
        del checkpoint
        torch.cuda.empty_cache()

    if any(tensor.device.type != "cuda" for tensor in averaged_state.values()):
        raise RuntimeError("Averaged state escaped CUDA")

    output_payload = {
        "schema_version": int(reference_metadata.get("schema_version", 4)),
        "epoch": "cuda_weight_average_013_015_017",
        "model_state_dict": averaged_state,
        "config": reference_config,
        "val_metrics": {},
        "selection_objective": None,
        "checkpoint_selection": "cuda_weight_average_neighbouring_snapshots",
        "source_checkpoints": source_metadata,
        "compute_policy": "cuda_only_tensor_arithmetic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(args.output)

    report_payload = {
        "output": str(args.output.resolve()),
        "output_size_bytes": args.output.stat().st_size,
        "sources": source_metadata,
        "n_state_tensors": len(averaged_state),
        "all_output_tensors_cuda_before_save": True,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "compute_policy": "cuda_only_tensor_arithmetic",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
