"""Materialize a locked, paired MLS horizontal-reflection screen.

This is deliberately separate from G1: both arms retain the central 3-channel
input and differ only in training-time left/right reflection probability.  The
script performs no model compute and refuses to overwrite a matrix directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.context_cache import sha256_file


TEMPLATE = PROJECT_ROOT / "config" / "experiments" / "mls-vast-deploy-aligned-baseline-template.yaml"
SOURCE_FILES = {
    "materializer": Path(__file__),
    "config_models": PROJECT_ROOT / "src" / "strategies" / "config_models.py",
    "dataset": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "dataset.py",
    "train_multitask": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "train_multitask.py",
    "predict_multitask": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "predict_multitask.py",
    "model": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "model.py",
    "mls_utils": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "utils.py",
    "input_contract": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py",
    "fold_manifest": PROJECT_ROOT / "config" / "folds.csv",
}
ARMS = {
    "control": {"slug": "control", "horizontal_flip_prob": 0.0},
    "candidate": {"slug": "reflect", "horizontal_flip_prob": 0.5},
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def _model_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match for the two causal arms except the intervention."""
    return {key: value for key, value in config.items() if key not in {"fold", "seed"}}


def _load_receipt(path: Path, expected_cache_sha256: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError("cache validation receipt is not passing")
    if payload.get("cache_manifest_sha256") != expected_cache_sha256:
        raise ValueError("cache receipt is not bound to requested manifest")
    return sha256_file(path)


def materialize(
    *, cache_manifest_sha256: str, cache_validation_receipt: Path,
    output_dir: Path, fold: int, seed: int,
) -> dict[str, Any]:
    digest = cache_manifest_sha256.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("cache manifest must be a lowercase SHA-256 digest")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite locked matrix: {output_dir}")
    receipt_sha = _load_receipt(cache_validation_receipt, digest)
    folds = pd.read_csv(PROJECT_ROOT / "config" / "folds.csv")
    studies = int((pd.to_numeric(folds["fold"], errors="raise") == fold).sum())
    if studies < 1:
        raise ValueError(f"fold {fold} is absent from immutable manifest")
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or not isinstance(template.get("training_config"), dict):
        raise ValueError("invalid MLS baseline template")

    output_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    signatures: dict[str, dict[str, Any]] = {}
    for arm_name, arm in ARMS.items():
        payload = json.loads(json.dumps(template))
        config = payload["training_config"]
        config.update({
            "fold": fold,
            "seed": seed,
            "dataset_variant": "multitask_2p5d_v1",
            "input_channels": 3,
            "context_cache_manifest_sha256": digest,
            "context_cache_validation_receipt_sha256": receipt_sha,
            "horizontal_flip_prob": arm["horizontal_flip_prob"],
            "resume_checkpoint": None,
        })
        normalized = MLSHeatmapConfig.model_validate(config).model_dump(mode="json")
        payload["training_config"] = normalized
        payload["run_name"] = f"mls-reflection-{arm['slug']}-fold{fold}-seed{seed}"
        payload["notes"] = (
            "Locked paired MLS reflection screen. Both arms use the same central "
            "3-channel cache, fold, seed and recipe; only horizontal_flip_prob differs. "
            "Epoch 15 is fixed for the CUDA audit."
        )
        payload["tags"] = {
            "campaign_id": "mls_reflection_paired_20260905",
            "experiment_key": "R1",
            "arm": arm_name,
            "changed_training_factor": "horizontal_flip_prob_only",
            "horizontal_flip_prob": arm["horizontal_flip_prob"],
            "cache_manifest_sha256": digest,
            "cache_validation_receipt_sha256": receipt_sha,
            "fixed_audit_epoch": 15,
            "compute_policy": "cuda_only_no_cpu_fallback",
        }
        payload["hardware"] = {"gpu_profile": "RTX_3090", "disk_gb": 100}
        payload["runtime"] = {
            "git_branch": "codex/mls-a2-geometry-20260904",
            "prepare_data": False, "auto_destroy": False,
        }
        path = output_dir / f"mls-reflection-{arm['slug']}-fold{fold}-seed{seed}.yaml"
        _atomic_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        signature = _model_signature(normalized)
        signatures[arm_name] = signature
        entries.append({
            "arm": arm_name, "file": path.name, "sha256": sha256_file(path),
            "fold": fold, "seed": seed, "studies": studies,
            "model_config_signature_sha256": _canonical_sha256(signature),
        })
    differences = sorted({
        key for key in signatures["control"] | signatures["candidate"]
        if signatures["control"].get(key) != signatures["candidate"].get(key)
    })
    if differences != ["horizontal_flip_prob"]:
        raise RuntimeError(f"reflection arms differ beyond intervention: {differences}")
    preregistration = {
        "schema_version": 1,
        "status": "locked_before_any_r1_cuda_outcome",
        "campaign": "mls_reflection_paired",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "fixed_audit_epoch": 15,
        "fold": fold, "seed": seed, "studies": studies,
        "cache_manifest_sha256": digest,
        "cache_validation_receipt": str(cache_validation_receipt.resolve()),
        "cache_validation_receipt_sha256": receipt_sha,
        "source_sha256": {name: sha256_file(path) for name, path in SOURCE_FILES.items()},
        "arms": {
            name: {**arm, "model_config_signature": signatures[name],
                   "model_config_signature_sha256": _canonical_sha256(signatures[name])}
            for name, arm in ARMS.items()
        },
        "training_factor_diff": differences,
        "screen_gates": [
            "study_f1_3mm_noninferior", "study_f1_5mm_noninferior",
            "study_boundary_f1_noninferior", "study_mae_noninferior",
            "then_cuda_raw_dicom_deploy_aligned_triage_gate",
        ],
        "configs": entries,
        "model_compute": "none",
    }
    prereg_path = output_dir / "r1_preregistration.json"
    _atomic_text(prereg_path, json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    return {"status": preregistration["status"], "configs": len(entries),
            "preregistration": str(prereg_path.resolve()),
            "preregistration_sha256": sha256_file(prereg_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--cache-validation-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(materialize(
        cache_manifest_sha256=args.cache_manifest_sha256,
        cache_validation_receipt=args.cache_validation_receipt,
        output_dir=args.output_dir.resolve(), fold=args.fold, seed=args.seed,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
