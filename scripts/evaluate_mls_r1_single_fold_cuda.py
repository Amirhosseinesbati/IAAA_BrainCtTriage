"""CUDA-only raw-DICOM evaluator for one locked R1 reflection arm.

This screen intentionally reports MLS metrics only.  It is a necessary
precondition, not evidence of final triage promotion; a candidate that passes
must still enter the frozen ICH/fracture deploy-aligned triage gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_mls_g1_three_seed_fold_cuda import _metrics
from src.evaluation.splits import load_fold_manifest
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.context_cache import sha256_file
from src.strategies.mls_heatmap.predict_multitask import (
    aggregate_study_mls,
    load_multitask_model,
    predict_reader_slices,
)


SOURCE_FILES = {
    "evaluator": Path(__file__),
    "config_models": ROOT / "src" / "strategies" / "config_models.py",
    "dataset": ROOT / "src" / "strategies" / "mls_heatmap" / "dataset.py",
    "train_multitask": ROOT / "src" / "strategies" / "mls_heatmap" / "train_multitask.py",
    "predict_multitask": ROOT / "src" / "strategies" / "mls_heatmap" / "predict_multitask.py",
    "model": ROOT / "src" / "strategies" / "mls_heatmap" / "model.py",
    "mls_utils": ROOT / "src" / "strategies" / "mls_heatmap" / "utils.py",
    "input_contract": ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py",
    "fold_manifest": ROOT / "config" / "folds.csv",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _load_contract(preregistration: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(preregistration.read_text(encoding="utf-8"))
    if document.get("status") != "locked_before_any_r1_cuda_outcome":
        raise ValueError("R1 preregistration was not locked before CUDA")
    if arm not in {"control", "candidate"}:
        raise ValueError("R1 arm must be control or candidate")
    config_entry = next((item for item in document["configs"] if item["arm"] == arm), None)
    if config_entry is None:
        raise ValueError("requested R1 arm is absent from preregistration")
    expected = document["arms"][arm]
    return document, {**config_entry, **expected}


def _validate_checkpoint(
    checkpoint: Path, contract: dict[str, Any], preregistration: dict[str, Any],
) -> tuple[MLSHeatmapConfig, dict[str, Any], str]:
    expected_path = checkpoint.parent.parent / "matrix" / str(contract["file"])
    if expected_path.is_file() and _sha256(expected_path) != contract["sha256"]:
        raise ValueError("matrix config file changed after R1 preregistration")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(payload.get("epoch", -1)) != int(preregistration["fixed_audit_epoch"]):
        raise ValueError("checkpoint epoch is not the locked R1 audit epoch")
    config = MLSHeatmapConfig.model_validate(payload["config"])
    if (
        int(config.fold) != int(preregistration["fold"])
        or int(config.seed) != int(preregistration["seed"])
        or int(config.input_channels) != 3
        or config.dataset_variant != "multitask_2p5d_v1"
        or float(config.horizontal_flip_prob) != float(contract["horizontal_flip_prob"])
    ):
        raise ValueError("checkpoint config does not match the locked R1 arm")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_sha256"), dict):
        raise ValueError("checkpoint lacks source provenance")
    expected_sources = preregistration["source_sha256"]
    observed_sources = provenance["source_sha256"]
    for name in ("config_models", "dataset", "train_multitask", "predict_multitask", "model", "mls_utils", "input_contract"):
        if observed_sources.get(name) != expected_sources.get(name):
            raise ValueError(f"checkpoint source provenance differs for {name}")
    checkpoint_sha = _sha256(checkpoint)
    return config, dict(provenance), checkpoint_sha


def _aggregate(slices: list[Any], config: MLSHeatmapConfig) -> float:
    if not slices or [item.index for item in slices] != list(range(len(slices))):
        raise ValueError("incomplete or unordered CUDA slice output")
    value = aggregate_study_mls(
        slices, selector_threshold=config.selector_threshold, top_k=config.top_k_slices,
        aggregation=config.aggregation, relative_ratio=config.selector_relative_ratio,
        aggregation_quantile=config.aggregation_quantile,
        probability_weighted=config.aggregation_probability_weighted,
        anchor_window_radius=config.anchor_window_radius,
        min_active_slices=config.min_active_slices,
        heatmap_guard_ratio=config.heatmap_guard_ratio,
        negative_value=config.negative_value_mm,
    )
    if not math.isfinite(value):
        raise ValueError("non-finite study MLS")
    return float(np.clip(value, 0.0, 30.0))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only evaluator found no GPU")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite R1 evaluator output")
    preregistration, contract = _load_contract(args.preregistration.resolve(), args.arm)
    if _sha256(args.preregistration.resolve()) != args.preregistration_sha256:
        raise ValueError("R1 preregistration checksum differs")
    checkpoint = args.checkpoint.resolve()
    config, provenance, checkpoint_sha = _validate_checkpoint(checkpoint, contract, preregistration)
    receipt = json.loads(args.cache_validation_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("cache_manifest_sha256") != preregistration["cache_manifest_sha256"]
        or _sha256(args.cache_validation_receipt) != preregistration["cache_validation_receipt_sha256"]
    ):
        raise ValueError("R1 cache receipt differs from preregistration")
    folds = load_fold_manifest()
    heldout = folds.loc[folds["fold"] == int(config.fold), ["study_id", "patient_id", "triage_class"]].copy()
    if len(heldout) != int(preregistration["studies"]):
        raise ValueError("held-out fold coverage differs from preregistration")
    truth = pd.read_csv(args.truth_table, dtype={"dicom_series.id": str})[
        ["dicom_series.id", "MLS_mm"]
    ].rename(columns={"dicom_series.id": "study_id", "MLS_mm": "gt_MLS_mm"})
    frame = heldout.merge(truth, on="study_id", how="left", validate="one_to_one").reset_index(drop=True)
    if frame["gt_MLS_mm"].isna().any():
        raise ValueError("missing MLS truth in held-out fold")

    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"
    private_path = args.output_dir / "study_predictions_private.csv"
    frame["MLS_mm"] = np.nan
    started = time.perf_counter()
    device = torch.device("cuda:0")
    model, loaded_config = load_multitask_model(checkpoint, device)
    torch.cuda.reset_peak_memory_stats(device)
    _atomic_json(status_path, {"status": "running", "expected_studies": len(frame), "completed_studies": 0})
    for index, row in frame.iterrows():
        reader = BrainDicomReader(str(args.data_root / str(row["study_id"]))).load_and_sort()
        slices = predict_reader_slices(reader, model, loaded_config, device, batch_size=args.batch_size)
        frame.at[index, "MLS_mm"] = _aggregate(slices, loaded_config)
        _atomic_csv(private_path, frame)
        _atomic_json(status_path, {"status": "running", "expected_studies": len(frame), "completed_studies": index + 1})
        torch.cuda.empty_cache()
    metrics = _metrics(frame["gt_MLS_mm"].to_numpy(float), frame["MLS_mm"].to_numpy(float))
    summary = {
        "schema_version": 1, "status": "completed", "campaign": "mls_reflection_paired",
        "scope": "raw_dicom_single_fold_mls_screen_only", "arm": args.arm,
        "fold": int(config.fold), "seed": int(config.seed), "studies": len(frame),
        "fixed_epoch": int(preregistration["fixed_audit_epoch"]),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "preregistration_sha256": _sha256(args.preregistration.resolve()),
        "cache_validation_receipt_sha256": _sha256(args.cache_validation_receipt),
        "source_sha256": {name: _sha256(path) for name, path in SOURCE_FILES.items()},
        "checkpoint_provenance": provenance,
        "metrics": metrics, "private_predictions_sha256": _sha256(private_path),
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "runtime_total_s": time.perf_counter() - started,
        "promotion_eligible": False, "submission_zip_allowed": False,
    }
    _atomic_json(args.output_dir / "aggregate_summary.json", summary)
    _atomic_json(status_path, {"status": "completed", "completed_studies": len(frame)})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--cache-validation-receipt", type=Path, required=True)
    parser.add_argument("--truth-table", type=Path, default=ROOT / "reports" / "eda" / "deep" / "deep_series_table.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Data" / "raw" / "training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "arm": result["arm"], "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
