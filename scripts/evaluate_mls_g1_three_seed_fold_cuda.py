"""Independent CUDA evaluator for the MLS G1 float32-cache 2.5D campaign.

Training consumes the immutable cache, while this evaluator intentionally
reconstructs every input from raw DICOM through the deployed runtime.  It is
therefore deploy-aligned and must not be replaced with a cache-based shortcut.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_mls_three_seed_fold_cuda import _aggregate, _metrics
from src.evaluation.splits import load_fold_manifest
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.context_cache import (
    load_mls_2p5d_cache_manifest,
    sha256_file,
)
from src.strategies.mls_heatmap.predict_multitask import (
    load_multitask_model,
    predict_reader_slices,
)


SAFE_LABEL = re.compile(r"^[a-zA-Z0-9_.-]+$")
# G1 is preregistered only as a fold-3 screen and conditional fold-4
# confirmation.  Other folds require a separately locked campaign.
IMMUTABLE_FOLDS = (3, 4)
FIXED_PROTOCOL = "heldout_fold_fixed_epoch15_three_distinct_seed_median"
ARM_CHANNELS = {"g1_c0_3ch": 3, "g1_a_9ch": 9}
FIXED_SEEDS = (42, 2026, 3407)
RUNTIME_SOURCE_FILES = {
    "g1_evaluator": Path(__file__),
    "base_three_seed_evaluator": PROJECT_ROOT / "scripts" / "evaluate_mls_three_seed_fold_cuda.py",
    "input_contract": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py",
    "context_cache": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "context_cache.py",
    "model": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "model.py",
    "predict_multitask": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "predict_multitask.py",
    "mls_utils": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "utils.py",
    "config_models": PROJECT_ROOT / "src" / "strategies" / "config_models.py",
    "dicom_reader": PROJECT_ROOT / "src" / "preprocessing" / "core" / "dicom_reader.py",
    "canonical_triage_reducer": PROJECT_ROOT / "scripts" / "evaluate_mls_deploy_aligned_seed_medians.py",
    "triage_rules": PROJECT_ROOT / "src" / "evaluation" / "triage.py",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return sha256_file(path)


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
    differences: set[str] = set()
    for label in labels[1:]:
        candidate = configs[label]
        for key in reference.keys() | candidate.keys():
            if reference.get(key) != candidate.get(key):
                differences.add(key)
    return differences


def _model_config_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return the fold/seed-free config that defines one G1 arm's recipe.

    Fold membership is bound independently in the held-out audit contract;
    removing it here lets fold-3 and preregistered fold-4 prove they use the
    identical optimization/pooling recipe.
    """
    return {key: value for key, value in config.items() if key not in {"seed", "fold"}}


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_passing_cache_receipt(path: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"G1 cache validation receipt is missing: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("G1 cache validation receipt must be a JSON object")
    if (
        receipt.get("status") != "passed"
        or receipt.get("cache_manifest_sha256") != expected_manifest_sha256
        or receipt.get("raw_fingerprints_verified") is not True
        or int(receipt.get("studies", -1)) != 338
        or int(receipt.get("rows", -1)) != 3484
        or receipt.get("model_compute") != "none"
    ):
        raise ValueError("G1 cache validation receipt does not prove the required immutable cache")
    return receipt


def _load_runtime_qualification(
    path: Path,
    *,
    arm: str,
    cache_manifest_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"G1 runtime qualification receipt is missing: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("G1 runtime qualification receipt must be a JSON object")
    if (
        receipt.get("status") != "passed"
        or receipt.get("campaign") != "g1_2p5d_deploy_aligned"
        or receipt.get("arm") != arm
        or receipt.get("cache_manifest_sha256") != cache_manifest_sha256
        or int(receipt.get("input_channels", -1)) != ARM_CHANNELS[arm]
        or receipt.get("compute_policy") != "cuda_only_no_cpu_model_fallback"
    ):
        raise ValueError("G1 runtime qualification receipt does not match this arm/cache")
    return receipt


def _resume_contract(
    *,
    fold: int,
    expected_studies: int,
    fixed_epoch: int,
    arm: str,
    input_channels: int,
    cache_manifest_sha256: str,
    cache_receipt_sha256: str,
    runtime_qualification_sha256: str,
    runtime_source_sha256: dict[str, str],
    checkpoint_manifest: dict[str, dict[str, Any]],
    study_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": FIXED_PROTOCOL,
        "campaign": "g1_2p5d_deploy_aligned",
        "arm": arm,
        "dataset_variant": "multitask_2p5d_v1",
        "input_channels": input_channels,
        "fold": fold,
        "expected_studies": expected_studies,
        "fixed_epoch": fixed_epoch,
        "study_ids_sha256": hashlib.sha256(
            ("\n".join(study_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_validation_receipt_sha256": cache_receipt_sha256,
        "runtime_qualification_receipt_sha256": runtime_qualification_sha256,
        "runtime_source_sha256": runtime_source_sha256,
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


def _require_matching_resume_contract(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError("Private predictions exist without a G1 resume contract")
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("G1 resume contract differs; stale private predictions are forbidden")


def _require_g1_config(
    config: Any,
    arm: str,
    expected_cache_sha256: str,
    expected_cache_receipt_sha256: str,
) -> None:
    expected_channels = ARM_CHANNELS[arm]
    if (
        config.dataset_variant != "multitask_2p5d_v1"
        or int(config.input_channels) != expected_channels
        or str(config.context_cache_manifest_sha256) != expected_cache_sha256
        or str(config.context_cache_validation_receipt_sha256) != expected_cache_receipt_sha256
        or int(config.image_size) != 512
        or not bool(config.use_selector)
    ):
        raise ValueError(
            "Checkpoint is not the requested immutable G1 arm/cache/runtime contract"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_CHANNELS), required=True)
    parser.add_argument("--fold", type=int, choices=IMMUTABLE_FOLDS, required=True)
    parser.add_argument("--fixed-epoch", type=int, default=15)
    parser.add_argument("--expected-studies", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--cache-validation-receipt", type=Path, required=True)
    parser.add_argument("--runtime-qualification-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Data" / "raw" / "training")
    parser.add_argument(
        "--truth-table", type=Path,
        default=PROJECT_ROOT / "reports" / "eda" / "deep" / "deep_series_table.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if len(args.checkpoint) != 3 or len({label for label, _ in args.checkpoint}) != 3:
        raise ValueError("Exactly three uniquely labelled seed checkpoints are required")
    if args.fixed_epoch != 15:
        raise ValueError("G1 preregistration fixes evaluation to epoch 15")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only G1 MLS audit requested; CPU fallback is forbidden")
    cache_manifest_sha256 = args.expected_cache_manifest_sha256.strip().lower()
    if len(cache_manifest_sha256) != 64 or any(char not in "0123456789abcdef" for char in cache_manifest_sha256):
        raise ValueError("--expected-cache-manifest-sha256 must be a SHA-256 digest")
    cache_manifest, observed_cache_sha256 = load_mls_2p5d_cache_manifest(
        args.cache_root.resolve(), expected_sha256=cache_manifest_sha256,
    )
    receipt = _load_passing_cache_receipt(args.cache_validation_receipt.resolve(), observed_cache_sha256)
    cache_receipt_sha256 = _sha256(args.cache_validation_receipt.resolve())
    qualification = _load_runtime_qualification(
        args.runtime_qualification_receipt.resolve(),
        arm=args.arm, cache_manifest_sha256=observed_cache_sha256,
    )
    qualification_sha256 = _sha256(args.runtime_qualification_receipt.resolve())
    runtime_source_sha256 = {name: _sha256(path) for name, path in RUNTIME_SOURCE_FILES.items()}

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    models: dict[str, torch.nn.Module] = {}
    configs: dict[str, Any] = {}
    config_payloads: dict[str, dict[str, Any]] = {}
    checkpoint_manifest: dict[str, dict[str, Any]] = {}
    for label, raw_path in args.checkpoint:
        checkpoint = raw_path.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        # Checkpoint deserialization is metadata/state loading only.  All
        # inference below is explicitly on CUDA and no CPU forward exists.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        epoch = int(payload.get("epoch", -1))
        if epoch != args.fixed_epoch:
            raise ValueError(f"{label} is epoch {epoch}; G1 fixes epoch {args.fixed_epoch}")
        provenance = payload.get("provenance")
        training_source_sha256 = (
            provenance.get("source_sha256") if isinstance(provenance, dict) else None
        )
        if (
            not isinstance(training_source_sha256, dict)
            or not training_source_sha256
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                for name, digest in training_source_sha256.items()
            )
        ):
            raise ValueError(
                f"{label} checkpoint lacks the immutable G1 training-source provenance"
            )
        model, config = load_multitask_model(checkpoint, device)
        _require_g1_config(config, args.arm, observed_cache_sha256, cache_receipt_sha256)
        if int(config.fold) != args.fold:
            raise ValueError(f"{label} config fold={config.fold}; audit fold={args.fold}")
        models[label] = model
        configs[label] = config
        config_payloads[label] = config.model_dump(mode="json")
        checkpoint_manifest[label] = {
            "path": str(checkpoint),
            "bytes": int(checkpoint.stat().st_size),
            "sha256": _sha256(checkpoint),
            "epoch": epoch,
            "seed": int(config.seed),
            "training_source_sha256": dict(sorted(training_source_sha256.items())),
        }
        del payload
    seeds = {int(config.seed) for config in configs.values()}
    if seeds != set(FIXED_SEEDS):
        raise ValueError(
            f"G1 preregistration requires exact seeds {list(FIXED_SEEDS)}, got {sorted(seeds)}"
        )
    labels = sorted(models)
    differences = _config_difference(config_payloads)
    if differences != {"seed"}:
        raise ValueError(
            "G1 seed ensemble configs must differ only in seed, got "
            f"{sorted(differences)}"
        )
    model_config_signature = _model_config_signature(config_payloads[labels[0]])
    for label in labels[1:]:
        if _model_config_signature(config_payloads[label]) != model_config_signature:
            raise ValueError("G1 seed ensemble has inconsistent seed-free model configuration")
    model_config_signature_sha256 = _canonical_sha256(model_config_signature)
    if qualification.get("model_config_signature_sha256") != model_config_signature_sha256:
        raise ValueError("G1 runtime qualification does not bind this seed-free model recipe")

    pinned_fold_sha = str(cache_manifest.get("sources", {}).get("fold_manifest_sha256", ""))
    current_fold_path = PROJECT_ROOT / "config" / "folds.csv"
    if _sha256(current_fold_path) != pinned_fold_sha:
        raise ValueError("Current fold manifest differs from the immutable G1 cache contract")
    folds = load_fold_manifest()
    manifest = folds.loc[
        folds["fold"] == args.fold, ["study_id", "patient_id", "triage_class"],
    ].copy()
    truth_path = args.truth_table.resolve()
    truth = pd.read_csv(truth_path, dtype={"dicom_series.id": str})[
        ["dicom_series.id", "MLS_mm"]
    ].rename(columns={"dicom_series.id": "study_id", "MLS_mm": "gt_MLS_mm"})
    frame = manifest.merge(truth, on="study_id", how="left", validate="one_to_one")
    if len(frame) != args.expected_studies or frame["gt_MLS_mm"].isna().any():
        raise ValueError(
            f"Fold contract mismatch: studies={len(frame)}, expected={args.expected_studies}, "
            f"missing_truth={int(frame['gt_MLS_mm'].isna().sum())}"
        )

    output_dir = args.output_dir.resolve()
    private_path = output_dir / "study_member_predictions_private.csv"
    resume_path = output_dir / "resume_contract.json"
    status_path = output_dir / "status.json"
    for label in labels:
        frame[f"{label}_MLS_mm"] = np.nan
        frame[f"{label}_runtime_s"] = np.nan
    frame["error"] = ""
    resume = _resume_contract(
        fold=args.fold,
        expected_studies=args.expected_studies,
        fixed_epoch=args.fixed_epoch,
        arm=args.arm,
        input_channels=ARM_CHANNELS[args.arm],
        cache_manifest_sha256=observed_cache_sha256,
        cache_receipt_sha256=cache_receipt_sha256,
        runtime_qualification_sha256=qualification_sha256,
        runtime_source_sha256=runtime_source_sha256,
        checkpoint_manifest=checkpoint_manifest,
        study_ids=frame["study_id"].astype(str).tolist(),
    )
    if private_path.is_file():
        _require_matching_resume_contract(resume_path, resume)
        previous = pd.read_csv(private_path, dtype={"study_id": str})
        reusable = [
            "study_id", "error",
            *(f"{label}_MLS_mm" for label in labels),
            *(f"{label}_runtime_s" for label in labels),
        ]
        if set(reusable) - set(previous):
            raise RuntimeError("G1 private resume file lacks required columns")
        frame = frame.drop(columns=[column for column in reusable if column != "study_id"])
        frame = frame.merge(previous[reusable], on="study_id", how="left", validate="one_to_one")
    else:
        _atomic_json(resume_path, resume)

    started = time.perf_counter()
    _atomic_json(status_path, {
        "schema_version": 1,
        "status": "running",
        "campaign": "g1_2p5d_deploy_aligned",
        "arm": args.arm,
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "fold": args.fold,
        "expected_studies": args.expected_studies,
        "fixed_epoch": args.fixed_epoch,
        "cache_manifest_sha256": observed_cache_sha256,
        "cache_validation_receipt_sha256": cache_receipt_sha256,
        "runtime_qualification_receipt_sha256": qualification_sha256,
        "checkpoints": checkpoint_manifest,
    })
    for index, row in frame.iterrows():
        if all(np.isfinite(float(row.get(f"{label}_MLS_mm", np.nan))) for label in labels):
            continue
        study_id = str(row["study_id"])
        try:
            reader = BrainDicomReader(str(args.data_root.resolve() / study_id)).load_and_sort()
            for label in labels:
                member_started = time.perf_counter()
                slices = predict_reader_slices(
                    reader, models[label], configs[label], device, batch_size=args.batch_size,
                )
                frame.at[index, f"{label}_MLS_mm"] = _aggregate(slices, configs[label])
                frame.at[index, f"{label}_runtime_s"] = time.perf_counter() - member_started
            frame.at[index, "error"] = ""
        except Exception as exc:
            frame.at[index, "error"] = f"{type(exc).__name__}: {exc}"
        _atomic_csv(frame, private_path)
        complete = int(frame[[f"{label}_MLS_mm" for label in labels]].notna().all(axis=1).sum())
        print(f"G1 {args.arm} fold{args.fold} {study_id}: {complete}/{len(frame)}", flush=True)
        torch.cuda.empty_cache()

    value_columns = [f"{label}_MLS_mm" for label in labels]
    failures = frame["error"].fillna("").ne("")
    incomplete = frame[value_columns].isna().any(axis=1)
    if failures.any() or incomplete.any():
        _atomic_json(status_path, {
            "schema_version": 1,
            "status": "failed",
            "campaign": "g1_2p5d_deploy_aligned",
            "arm": args.arm,
            "fold": args.fold,
            "failures": int(failures.sum()),
            "incomplete": int(incomplete.sum()),
            "checkpoints": checkpoint_manifest,
        })
        raise RuntimeError(
            f"Strict G1 three-seed audit incomplete: failures={int(failures.sum())}, "
            f"incomplete={int(incomplete.sum())}"
        )

    frame["median_MLS_mm"] = frame[value_columns].median(axis=1)
    # The canonical triage reducer consumes the persisted private table, not
    # this in-memory frame.  Persist the final ensemble column before hashing
    # or handing the artifact to the staged gate.
    _atomic_csv(frame, private_path)
    truth_values = frame["gt_MLS_mm"].to_numpy(float)
    result = {
        "schema_version": 1,
        "status": "completed",
        "protocol": FIXED_PROTOCOL,
        "campaign": "g1_2p5d_deploy_aligned",
        "arm": args.arm,
        "dataset_variant": "multitask_2p5d_v1",
        "input_channels": ARM_CHANNELS[args.arm],
        "compute_policy": "cuda_only_no_cpu_model_fallback",
        "finished_utc": _utc_now(),
        "fold": args.fold,
        "studies": len(frame),
        "fixed_epoch": args.fixed_epoch,
        "seeds": sorted(seeds),
        "config_differences": sorted(differences),
        "model_config_signature": model_config_signature,
        "model_config_signature_sha256": model_config_signature_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "cache_manifest_sha256": observed_cache_sha256,
        "cache_validation_receipt": {
            "path": str(args.cache_validation_receipt.resolve()),
            "sha256": cache_receipt_sha256,
            "validated_at_utc": receipt.get("validated_at_utc"),
        },
        "runtime_qualification_receipt": {
            "path": str(args.runtime_qualification_receipt.resolve()),
            "sha256": qualification_sha256,
            "qualified_at_utc": qualification.get("qualified_at_utc"),
        },
        "runtime_source_sha256": runtime_source_sha256,
        "truth_table_sha256": _sha256(truth_path),
        "fold_manifest_sha256": _sha256(PROJECT_ROOT / "config" / "folds.csv"),
        "member_metrics": {
            label: _metrics(truth_values, frame[f"{label}_MLS_mm"].to_numpy(float))
            for label in labels
        },
        "median_metrics": _metrics(truth_values, frame["median_MLS_mm"].to_numpy(float)),
        "runtime_total_s": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "private_predictions_sha256": _sha256(private_path),
        "raw_predictions_uploaded_to_mlflow": False,
    }
    _atomic_json(output_dir / "aggregate_summary.json", result)
    _atomic_json(status_path, result)
    print(json.dumps({
        "status": result["status"], "arm": args.arm, "fold": args.fold,
        "studies": result["studies"], "median_metrics": result["median_metrics"],
        "peak_vram_gib": result["peak_vram_gib"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
