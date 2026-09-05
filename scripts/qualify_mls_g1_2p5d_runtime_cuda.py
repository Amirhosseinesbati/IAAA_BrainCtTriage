"""Qualify the G1 cache-to-deployment MLS input contract on CUDA.

The cache is a training artifact only.  This script proves, on deterministic
raw-DICOM anchors, that its central/adjacent inputs equal the deployed raw
runtime inputs and that the selected G1 checkpoint accepts them on CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_mls_g1_three_seed_fold_cuda import (
    ARM_CHANNELS,
    _canonical_sha256,
    _load_passing_cache_receipt,
    _model_config_signature,
    _require_g1_config,
)
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.context_cache import (
    load_mls_2p5d_cache_manifest,
    sha256_file,
)
from src.strategies.mls_heatmap.input_contract import create_study_windowed_input
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model


SOURCE_FILES = {
    "qualification": Path(__file__),
    "g1_evaluator": PROJECT_ROOT / "scripts" / "evaluate_mls_g1_three_seed_fold_cuda.py",
    "input_contract": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py",
    "context_cache": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "context_cache.py",
    "model": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "model.py",
    "predict_multitask": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "predict_multitask.py",
    "mls_utils": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "utils.py",
    "dicom_reader": PROJECT_ROOT / "src" / "preprocessing" / "core" / "dicom_reader.py",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _input_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _cache_input(cache: np.ndarray, index: int, input_channels: int) -> np.ndarray:
    positions = [index] if input_channels == 3 else [
        max(0, index - 1), index, min(cache.shape[0] - 1, index + 1),
    ]
    return np.concatenate([np.asarray(cache[position]) for position in positions], axis=0)


def _anchor_studies(labels: pd.DataFrame, maximum: int) -> list[str]:
    if maximum < 2:
        raise ValueError("--anchor-studies must be at least two to cover positive and clean-negative studies")
    labels = labels.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels["is_target"] = pd.to_numeric(labels["is_target"], errors="raise")
    study_has_target = labels.groupby("patient_id", sort=True)["is_target"].max()
    positive = sorted(study_has_target.index[study_has_target > 0.5].astype(str))
    clean_negative = sorted(study_has_target.index[study_has_target <= 0.5].astype(str))
    if not positive or not clean_negative:
        raise ValueError("G1 qualification requires at least one positive and one clean-negative study")
    chosen = [positive[0], clean_negative[0]]
    for study_id in sorted(study_has_target.index.astype(str)):
        if study_id not in chosen:
            chosen.append(study_id)
        if len(chosen) >= maximum:
            return chosen
    return chosen


def qualify(
    *,
    checkpoint: Path,
    arm: str,
    cache_root: Path,
    expected_cache_sha256: str,
    cache_validation_receipt: Path,
    raw_root: Path,
    anchor_studies: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only G1 runtime qualification requested; CPU fallback is forbidden")
    expected_channels = ARM_CHANNELS[arm]
    manifest, cache_sha256 = load_mls_2p5d_cache_manifest(
        cache_root, expected_sha256=expected_cache_sha256,
    )
    receipt = _load_passing_cache_receipt(cache_validation_receipt, cache_sha256)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("epoch", -1)) != 15:
        raise ValueError("G1 runtime qualification requires the preregistered epoch-15 checkpoint")
    device = torch.device("cuda:0")
    model, config = load_multitask_model(checkpoint, device)
    _require_g1_config(
        config,
        arm,
        cache_sha256,
        sha256_file(cache_validation_receipt),
    )
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("G1 qualification model is not resident on CUDA")

    labels = pd.read_csv(cache_root / str(manifest["labels_csv"]), dtype={"patient_id": str})
    studies_dir = cache_root / str(manifest["study_cache_dir"])
    anchors: list[dict[str, Any]] = []
    cuda_input: np.ndarray | None = None
    for study_id in _anchor_studies(labels, anchor_studies):
        reader = BrainDicomReader(str(raw_root / study_id)).load_and_sort()
        raw_volume = reader.get_3d_volume_hu()
        cache = np.load(studies_dir / f"{study_id}.npy", mmap_mode="r", allow_pickle=False)
        if raw_volume.shape[2] != cache.shape[0]:
            raise ValueError(f"G1 qualification cache/raw depth mismatch for study {study_id}")
        indices = sorted({0, raw_volume.shape[2] // 2, raw_volume.shape[2] - 1})
        for index in indices:
            raw_input = create_study_windowed_input(raw_volume, index, expected_channels)
            cached_input = _cache_input(cache, index, expected_channels)
            if not np.array_equal(raw_input, cached_input):
                raise ValueError(
                    f"G1 cache/deployment input mismatch for study={study_id}, z={index}, "
                    f"channels={expected_channels}"
                )
            anchors.append({
                "study_id": study_id,
                "slice_index": index,
                "input_channels": expected_channels,
                "input_sha256": _input_digest(raw_input),
                "edge_replicated": bool(index in {0, raw_volume.shape[2] - 1}),
            })
            if cuda_input is None:
                cuda_input = raw_input
    if cuda_input is None:
        raise RuntimeError("G1 qualification selected no raw-DICOM anchors")

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(cuda_input)).unsqueeze(0).to(device)
        heatmaps, selector = model.forward_multitask(tensor)
    if (
        heatmaps.device.type != "cuda"
        or selector.device.type != "cuda"
        or not torch.isfinite(heatmaps).all().item()
        or not torch.isfinite(selector).all().item()
    ):
        raise RuntimeError("G1 CUDA runtime forward did not yield finite CUDA tensors")
    return {
        "schema_version": 1,
        "status": "passed",
        "campaign": "g1_2p5d_deploy_aligned",
        "arm": arm,
        "input_channels": expected_channels,
        "cache_manifest_sha256": cache_sha256,
        "cache_validation_receipt_sha256": sha256_file(cache_validation_receipt),
        "cache_validation_at_utc": receipt.get("validated_at_utc"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "config_fold": int(config.fold),
        "model_config_signature_sha256": _canonical_sha256(
            _model_config_signature(config.model_dump(mode="json"))
        ),
        "anchors": anchors,
        "cuda_forward": {
            "heatmap_shape": [int(value) for value in heatmaps.shape],
            "selector_shape": [int(value) for value in selector.shape],
            "device": torch.cuda.get_device_name(0),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "runtime_source_sha256": {name: sha256_file(path) for name, path in SOURCE_FILES.items()},
        "qualified_at_utc": datetime.now(timezone.utc).isoformat(),
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "raw_predictions_uploaded_to_mlflow": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_CHANNELS), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--cache-validation-receipt", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "Data" / "raw" / "training")
    parser.add_argument("--anchor-studies", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(
        checkpoint=args.checkpoint.resolve(), arm=args.arm,
        cache_root=args.cache_root.resolve(),
        expected_cache_sha256=args.expected_cache_manifest_sha256.strip().lower(),
        cache_validation_receipt=args.cache_validation_receipt.resolve(),
        raw_root=args.raw_root.resolve(), anchor_studies=args.anchor_studies,
    )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"], "arm": result["arm"],
        "anchors": len(result["anchors"]), "input_channels": result["input_channels"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
