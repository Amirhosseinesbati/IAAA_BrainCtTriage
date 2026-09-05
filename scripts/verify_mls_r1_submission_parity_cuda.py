"""Fail-closed CUDA differential test for the locked R1 selector runtime.

It compares source-runtime and self-contained submission-runtime predictions on
the same raw DICOM studies.  It records only equality diagnostics, never the
private per-study/per-slice predictions themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submission.model as submission_model
from scripts.evaluate_mls_r1_single_fold_cuda import (
    _load_contract,
    _sha256,
    _validate_checkpoint,
)
from src.evaluation.splits import load_fold_manifest
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.predict_multitask import (
    aggregate_study_mls,
    load_multitask_model,
    predict_reader_slices,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_study_mls(slices: list[Any], config: Any) -> float:
    value = aggregate_study_mls(
        slices,
        selector_threshold=config.selector_threshold,
        top_k=config.top_k_slices,
        aggregation=config.aggregation,
        relative_ratio=config.selector_relative_ratio,
        aggregation_quantile=config.aggregation_quantile,
        probability_weighted=config.aggregation_probability_weighted,
        anchor_window_radius=config.anchor_window_radius,
        min_active_slices=config.min_active_slices,
        heatmap_guard_ratio=config.heatmap_guard_ratio,
        negative_value=config.negative_value_mm,
    )
    return float(np.clip(value, 0.0, 30.0))


def _slice_deltas(source: list[Any], package: tuple[Any, ...]) -> dict[str, float]:
    if len(source) != len(package):
        raise ValueError(f"slice count differs: source={len(source)} package={len(package)}")
    maxima = {"selector_probability": 0.0, "peak_probability": 0.0, "heatmap_peak": 0.0, "mls_mm": 0.0}
    for expected, actual in zip(source, package, strict=True):
        if int(expected.index) != int(actual.index):
            raise ValueError("slice order/index differs between source and submission runtime")
        for key in maxima:
            left = getattr(expected, key)
            right = getattr(actual, key)
            if left is None or right is None:
                if left != right:
                    raise ValueError(f"slice {expected.index} has incompatible {key} schema")
                continue
            maxima[key] = max(maxima[key], abs(float(left) - float(right)))
    return maxima


def _study_ids(args: argparse.Namespace, fold: int) -> list[str]:
    if args.study_dir:
        return [Path(item).name for item in args.study_dir]
    if args.data_root is None:
        raise ValueError("provide --study-dir or --data-root")
    folds = load_fold_manifest()
    return [str(value) for value in folds.loc[folds["fold"] == fold, "study_id"].tolist()]


def _study_path(args: argparse.Namespace, study_id: str) -> Path:
    if args.study_dir:
        return next(path for path in (Path(value) for value in args.study_dir) if path.name == study_id)
    assert args.data_root is not None
    return args.data_root / study_id


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R1 submission parity verification is CUDA-only")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity receipt: {args.output}")
    device = torch.device("cuda:0")
    preregistration, contract = _load_contract(args.preregistration.resolve(), args.arm)
    if _sha256(args.preregistration.resolve()) != args.preregistration_sha256:
        raise ValueError("R1 preregistration checksum differs")
    matrix_dir = args.matrix_dir.resolve() if args.matrix_dir else args.preregistration.resolve().parent
    locked_config, _, checkpoint_sha = _validate_checkpoint(
        args.checkpoint.resolve(), contract, preregistration, matrix_dir,
    )
    source_model, config = load_multitask_model(args.checkpoint, device)
    if config.model_dump(mode="json") != locked_config.model_dump(mode="json"):
        raise ValueError("source runtime config differs from locked R1 checkpoint config")
    if config.input_channels != 3 or not config.use_selector or config.selector_head_mode != "single":
        raise ValueError("checkpoint is not the locked central-3ch single-selector R1 contract")
    package_models = submission_model.load_mls_models(str(args.models_dir), device="cuda")
    if package_models.get("mls_locked_runtime") is not True:
        raise ValueError("submission loader did not activate a locked selector runtime")
    if int(package_models["mls_batch_size"]) != int(args.batch_size):
        raise ValueError("submission batch size is not the requested locked parity batch")

    study_ids = _study_ids(args, int(config.fold))
    if not study_ids:
        raise ValueError("no parity studies selected")
    overall = {"selector_probability": 0.0, "peak_probability": 0.0, "heatmap_peak": 0.0, "mls_mm": 0.0, "study_mls_mm": 0.0}
    errors: list[str] = []
    completed = 0
    torch.cuda.reset_peak_memory_stats(device)
    for study_id in study_ids:
        path = _study_path(args, study_id)
        try:
            source_reader = BrainDicomReader(str(path)).load_and_sort()
            package_reader = submission_model.DicomReader(str(path)).load_and_sort()
            source_volume = source_reader.get_3d_volume_hu()
            package_volume = package_reader.get_3d_volume_hu()
            if source_volume.shape != package_volume.shape or not np.array_equal(source_volume, package_volume):
                raise ValueError("source/submission raw HU volume differs")
            if float(source_reader.metadata["spacing_x"]) != float(package_reader.metadata["spacing_x"]):
                raise ValueError("source/submission spacing_x differs")
            source_slices = predict_reader_slices(
                source_reader, source_model, config, device, batch_size=args.batch_size,
            )
            source_value = _source_study_mls(source_slices, config)
            package_value, package_slices = submission_model._predict_mls_heatmap(
                vol_hu=package_volume,
                heatmap_model=package_models["mls_model"],
                spacing_x=package_reader.metadata["spacing_x"],
                device=device,
                batch_size=package_models["mls_batch_size"],
                min_peak=package_models["mls_min_peak"],
                top_k=package_models["mls_top_k"],
                aggregation=package_models["mls_aggregation"],
                use_selector=package_models["mls_use_selector"],
                selector_head_mode=package_models["mls_selector_head_mode"],
                selector_threshold=package_models["mls_selector_threshold"],
                selector_relative_ratio=package_models["mls_selector_relative_ratio"],
                aggregation_quantile=package_models["mls_aggregation_quantile"],
                aggregation_probability_weighted=package_models["mls_aggregation_probability_weighted"],
                anchor_window_radius=package_models["mls_anchor_window_radius"],
                min_active_slices=package_models["mls_min_active_slices"],
                heatmap_guard_ratio=package_models["mls_heatmap_guard_ratio"],
                negative_value_mm=package_models["mls_negative_value_mm"],
                return_trace=True,
            )
            for key, value in _slice_deltas(source_slices, package_slices).items():
                overall[key] = max(overall[key], value)
            overall["study_mls_mm"] = max(overall["study_mls_mm"], abs(float(source_value) - float(package_value)))
            if abs(float(source_value) - float(package_value)) > args.atol:
                raise ValueError("study MLS differs beyond tolerance")
            completed += 1
        except Exception as exc:
            errors.append(f"{study_id}: {type(exc).__name__}: {exc}")
            break
        finally:
            torch.cuda.empty_cache()

    passed = not errors and all(value <= args.atol for value in overall.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "campaign": "mls_reflection_r1_submission_runtime_parity",
        "scope": "raw_dicom_source_vs_submission_cuda_differential",
        "fold": int(config.fold),
        "arm": args.arm,
        "checkpoint_sha256": checkpoint_sha,
        "preregistration_sha256": args.preregistration_sha256,
        "studies_requested": len(study_ids),
        "studies_completed": completed,
        "batch_size": int(args.batch_size),
        "atol": float(args.atol),
        "max_absolute_deltas": overall,
        "errors": errors,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "private_predictions_persisted": False,
    }
    _atomic_json(args.output, result)
    if not passed:
        raise RuntimeError("R1 source/submission parity did not pass")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--study-dir", type=Path, action="append")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
