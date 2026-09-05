"""Leak-free CUDA audit of a fixed three-seed MLS ensemble on one held-out fold.

This evaluator deliberately has no pooling grid or checkpoint search. All three
members must be the preregistered epoch-15 checkpoint from the same fold and
must share every model-affecting config field except ``seed``. Study inference
uses each checkpoint's frozen pooling config and deployment uses their median.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import load_fold_manifest
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.predict_multitask import (
    aggregate_study_mls,
    load_multitask_model,
    predict_reader_slices,
)


SAFE_LABEL = re.compile(r"^[a-zA-Z0-9_.-]+$")
IMMUTABLE_FOLDS = (0, 1, 2, 3, 4)
ALLOWED_NON_MODEL_CONFIG_DIFFERENCES = {
    "seed",
    "snapshot_start_epoch",
    "snapshot_every_n_epochs",
    "resume_checkpoint",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _persist_final_predictions(
    frame: pd.DataFrame,
    value_columns: list[str],
    private_path: Path,
) -> str:
    """Persist the canonical ensemble column and return the exact file hash."""
    frame["median_MLS_mm"] = frame[value_columns].median(axis=1)
    # The canonical triage reducer consumes the same checksum-bound private
    # audit CSV and verifies that this stored median equals the three members.
    # Persist it *before* taking the hash; otherwise a completed audit cannot
    # be handed off to the deploy-aligned triage gate.
    _atomic_csv(frame, private_path)
    return _sha256(private_path)


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not SAFE_LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError(f"unsafe checkpoint label: {label!r}")
    return label, Path(raw_path).expanduser()


def _config_difference(configs: dict[str, dict[str, Any]]) -> set[str]:
    labels = sorted(configs)
    reference = configs[labels[0]]
    different: set[str] = set()
    for label in labels[1:]:
        candidate = configs[label]
        for key in reference.keys() | candidate.keys():
            if reference.get(key) != candidate.get(key):
                different.add(key)
    return different


def _resume_contract(
    *,
    fold: int,
    expected_studies: int,
    fixed_epoch: int,
    checkpoint_manifest: dict[str, dict[str, Any]],
    study_ids: list[str],
    data_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the immutable identity for safely resuming private predictions."""
    studies_digest = hashlib.sha256(
        ("\n".join(study_ids) + "\n").encode("utf-8")
    ).hexdigest()
    contract = {
        "schema_version": 1,
        "protocol": "heldout_fold_fixed_epoch15_three_distinct_seed_median",
        "fold": fold,
        "expected_studies": expected_studies,
        "fixed_epoch": fixed_epoch,
        "study_ids_sha256": studies_digest,
        "checkpoints": {
            label: {
                "bytes": int(metadata["bytes"]),
                "sha256": str(metadata["sha256"]),
                "epoch": int(metadata["epoch"]),
                "seed": int(metadata["seed"]),
            }
            for label, metadata in sorted(checkpoint_manifest.items())
        },
    }
    # Old private-audit files did not record data sources.  New callers bind
    # the exact fold/truth/evaluator inputs so a resumed raw-DICOM audit cannot
    # silently cross a mutable configuration boundary.
    if data_sources is not None:
        contract["data_sources"] = dict(sorted(data_sources.items()))
    return contract


def _require_matching_resume_contract(
    contract_path: Path,
    expected: dict[str, Any],
) -> None:
    if not contract_path.is_file():
        raise RuntimeError(
            "Private predictions exist without a resume contract; refusing unsafe reuse"
        )
    try:
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Resume contract is unreadable; refusing unsafe reuse") from exc
    if observed != expected:
        raise RuntimeError(
            "Resume contract does not match this fold/checkpoint ensemble; "
            "refusing stale private predictions"
        )


def _aggregate(slices: list[Any], config: Any) -> float:
    return float(np.clip(aggregate_study_mls(
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
    ), 0.0, 30.0))


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    threshold_f1 = {
        str(threshold): float(f1_score(
            truth >= threshold, prediction >= threshold, zero_division=0,
        ))
        for threshold in (1.0, 3.0, 5.0)
    }
    boundary = (threshold_f1["3.0"] + threshold_f1["5.0"]) / 2.0
    mae = float(np.mean(np.abs(prediction - truth)))
    return {
        "mae_mm": mae,
        "rmse_mm": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "bias_mm": float(np.mean(prediction - truth)),
        "f1_1mm": threshold_f1["1.0"],
        "f1_3mm": threshold_f1["3.0"],
        "f1_5mm": threshold_f1["5.0"],
        "boundary_f1": boundary,
        "selection_objective": mae + 2.0 * (1.0 - boundary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--fold", type=int, choices=IMMUTABLE_FOLDS, required=True)
    parser.add_argument("--fixed-epoch", type=int, default=15)
    parser.add_argument("--expected-studies", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Data/raw/training")
    parser.add_argument(
        "--fold-manifest", type=Path,
        default=PROJECT_ROOT / "config/folds.csv",
    )
    parser.add_argument(
        "--truth-table", type=Path,
        default=PROJECT_ROOT / "reports/eda/deep/deep_series_table.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    fold_manifest_path = args.fold_manifest.resolve()
    truth_table_path = args.truth_table.resolve()
    for label, path in {
        "raw DICOM root": data_root,
        "fold manifest": fold_manifest_path,
        "truth table": truth_table_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"raw DICOM root is not a directory: {data_root}")
    if not fold_manifest_path.is_file() or not truth_table_path.is_file():
        raise ValueError("fold manifest and truth table must be files")
    data_sources = {
        "data_root": str(data_root),
        "fold_manifest_path": str(fold_manifest_path),
        "fold_manifest_sha256": _sha256(fold_manifest_path),
        "truth_table_path": str(truth_table_path),
        "truth_table_sha256": _sha256(truth_table_path),
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
    }

    if len(args.checkpoint) != 3 or len({label for label, _ in args.checkpoint}) != 3:
        raise ValueError("Exactly three uniquely labelled seed checkpoints are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only MLS audit requested; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    # PyTorch 2.10+cu128 on the Vast RTX 3060 accepts the current device (or
    # its integer index) here but rejects a ``torch.device`` instance with
    # ``RuntimeError: Invalid device argument``.  The audit is deliberately
    # single-GPU, so using the already-current CUDA device is unambiguous.
    torch.cuda.reset_peak_memory_stats()

    models: dict[str, torch.nn.Module] = {}
    configs: dict[str, Any] = {}
    config_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_manifest: dict[str, dict[str, Any]] = {}
    for label, raw_path in args.checkpoint:
        checkpoint = raw_path.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        epoch = int(payload.get("epoch", -1))
        if epoch != args.fixed_epoch:
            raise ValueError(
                f"{label} is epoch {epoch}; fixed preregistered epoch is {args.fixed_epoch}"
            )
        model, config = load_multitask_model(checkpoint, device)
        if int(config.fold) != args.fold:
            raise ValueError(f"{label} config fold={config.fold}; audit fold={args.fold}")
        models[label] = model
        configs[label] = config
        config_payloads[label] = config.model_dump(mode="json")
        checkpoint_manifest[label] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "epoch": epoch,
            "seed": int(config.seed),
            "use_ordinal_aux_head": bool(config.use_ordinal_aux_head),
        }
        del payload
    seeds = {int(config.seed) for config in configs.values()}
    if len(seeds) != 3:
        raise ValueError(f"Expected three distinct seeds, got {sorted(seeds)}")
    differences = _config_difference(config_payloads)
    if "seed" not in differences or not differences.issubset(
        ALLOWED_NON_MODEL_CONFIG_DIFFERENCES
    ):
        raise ValueError(
            "Seed ensemble configs differ in a model-affecting field: "
            f"{sorted(differences)}"
        )

    folds = load_fold_manifest(fold_manifest_path)
    manifest = folds.loc[
        folds["fold"] == args.fold, ["study_id", "patient_id", "triage_class"],
    ].copy()
    truth = pd.read_csv(
        truth_table_path, dtype={"dicom_series.id": str},
    )[["dicom_series.id", "MLS_mm"]].rename(
        columns={"dicom_series.id": "study_id", "MLS_mm": "gt_MLS_mm"},
    )
    frame = manifest.merge(truth, on="study_id", how="left", validate="one_to_one")
    if len(frame) != args.expected_studies or frame["gt_MLS_mm"].isna().any():
        raise ValueError(
            f"Fold contract mismatch: studies={len(frame)}, expected={args.expected_studies}, "
            f"missing_truth={int(frame['gt_MLS_mm'].isna().sum())}"
        )

    output_dir = args.output_dir.resolve()
    private_path = output_dir / "study_member_predictions_private.csv"
    resume_contract_path = output_dir / "resume_contract.json"
    status_path = output_dir / "status.json"
    for label in sorted(models):
        frame[f"{label}_MLS_mm"] = np.nan
        frame[f"{label}_runtime_s"] = np.nan
    frame["error"] = ""
    resume_contract = _resume_contract(
        fold=args.fold,
        expected_studies=args.expected_studies,
        fixed_epoch=args.fixed_epoch,
        checkpoint_manifest=checkpoint_manifest,
        study_ids=frame["study_id"].astype(str).tolist(),
        data_sources=data_sources,
    )
    if private_path.is_file():
        _require_matching_resume_contract(resume_contract_path, resume_contract)
        previous = pd.read_csv(private_path, dtype={"study_id": str})
        reusable = [
            "study_id", "error",
            *(f"{label}_MLS_mm" for label in sorted(models)),
            *(f"{label}_runtime_s" for label in sorted(models)),
        ]
        missing_columns = sorted(set(reusable) - set(previous.columns))
        if missing_columns:
            raise RuntimeError(
                f"Private resume file is missing required columns: {missing_columns}"
            )
        frame = frame.drop(columns=[column for column in reusable if column != "study_id"])
        frame = frame.merge(previous[reusable], on="study_id", how="left", validate="one_to_one")
    else:
        _atomic_json(resume_contract_path, resume_contract)

    started = time.perf_counter()
    _atomic_json(status_path, {
        "schema_version": 1,
        "status": "running",
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "fold": args.fold,
        "expected_studies": args.expected_studies,
        "fixed_epoch": args.fixed_epoch,
        "checkpoints": checkpoint_manifest,
        "data_sources": data_sources,
    })
    labels = sorted(models)
    for index, row in frame.iterrows():
        if all(np.isfinite(float(row.get(f"{label}_MLS_mm", np.nan))) for label in labels):
            continue
        study_id = str(row["study_id"])
        try:
            reader = BrainDicomReader(str(data_root / study_id)).load_and_sort()
            for label in labels:
                member_started = time.perf_counter()
                slices = predict_reader_slices(
                    reader, models[label], configs[label], device,
                    batch_size=args.batch_size,
                )
                frame.at[index, f"{label}_MLS_mm"] = _aggregate(slices, configs[label])
                frame.at[index, f"{label}_runtime_s"] = time.perf_counter() - member_started
            frame.at[index, "error"] = ""
        except Exception as exc:
            frame.at[index, "error"] = f"{type(exc).__name__}: {exc}"
        _atomic_csv(frame, private_path)
        complete = int(frame[[f"{label}_MLS_mm" for label in labels]].notna().all(axis=1).sum())
        print(f"three-seed fold{args.fold} {study_id}: {complete}/{len(frame)}", flush=True)
        torch.cuda.empty_cache()

    value_columns = [f"{label}_MLS_mm" for label in labels]
    failures = frame["error"].fillna("").ne("")
    incomplete = frame[value_columns].isna().any(axis=1)
    if failures.any() or incomplete.any():
        _atomic_json(status_path, {
            "schema_version": 1,
            "status": "failed",
            "fold": args.fold,
            "failures": int(failures.sum()),
            "incomplete": int(incomplete.sum()),
            "checkpoints": checkpoint_manifest,
        })
        raise RuntimeError(
            f"Strict three-seed audit incomplete: failures={int(failures.sum())}, "
            f"incomplete={int(incomplete.sum())}"
        )

    private_predictions_sha256 = _persist_final_predictions(
        frame,
        value_columns,
        private_path,
    )
    truth_values = frame["gt_MLS_mm"].to_numpy(float)
    member_metrics = {
        label: _metrics(truth_values, frame[f"{label}_MLS_mm"].to_numpy(float))
        for label in labels
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "protocol": "heldout_fold_fixed_epoch15_three_distinct_seed_median",
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "finished_utc": _utc_now(),
        "fold": args.fold,
        "studies": len(frame),
        "fixed_epoch": args.fixed_epoch,
        "seeds": sorted(seeds),
        "config_differences": sorted(differences),
        "checkpoint_manifest": checkpoint_manifest,
        "data_sources": data_sources,
        "member_metrics": member_metrics,
        "median_metrics": _metrics(truth_values, frame["median_MLS_mm"].to_numpy(float)),
        "runtime_total_s": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "private_predictions_sha256": private_predictions_sha256,
        "raw_predictions_uploaded_to_mlflow": False,
    }
    _atomic_json(output_dir / "aggregate_summary.json", result)
    _atomic_json(status_path, result)
    print(json.dumps({
        "status": result["status"],
        "fold": result["fold"],
        "studies": result["studies"],
        "median_metrics": result["median_metrics"],
        "peak_vram_gib": result["peak_vram_gib"],
    }))


if __name__ == "__main__":
    main()
