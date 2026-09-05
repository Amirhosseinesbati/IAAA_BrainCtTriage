"""Validate a materialized G1 C0-vs-9-channel experiment matrix before CUDA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_mls_g1_three_seed_fold_cuda import _canonical_sha256, _model_config_signature
from scripts.evaluate_mls_g1_three_seed_fold_cuda import _load_passing_cache_receipt
from scripts.materialize_mls_g1_context_matrix import ARMS, FOLDS, SEEDS, SOURCE_FILES
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.context_cache import sha256_file


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("G1 preregistration must be a JSON object")
    return payload


def validate(*, matrix_dir: Path, expected_cache_sha256: str | None) -> dict[str, Any]:
    prereg_path = matrix_dir / "g1_preregistration.json"
    prereg = _read_json(prereg_path)
    cache_sha = str(prereg.get("cache_manifest_sha256", ""))
    if expected_cache_sha256 is not None and cache_sha != expected_cache_sha256.lower():
        raise ValueError("G1 matrix cache hash differs from the requested immutable cache")
    cache_receipt_path = Path(prereg["cache_validation_receipt"])
    _load_passing_cache_receipt(cache_receipt_path, cache_sha)
    cache_receipt_sha = sha256_file(cache_receipt_path)
    if prereg.get("cache_validation_receipt_sha256") != cache_receipt_sha:
        raise ValueError("G1 preregistration cache validation receipt checksum differs")
    if prereg.get("status") != "locked_before_any_g1_cuda_outcome":
        raise ValueError("G1 matrix is not locked before CUDA outcome")
    if prereg.get("campaign") != "g1_2p5d_deploy_aligned":
        raise ValueError("Unexpected G1 matrix campaign")
    if prereg.get("seeds") != list(SEEDS):
        raise ValueError("G1 matrix seed contract differs")
    expected_entries = {
        (arm_name, fold, seed)
        for arm_name in ARMS for fold in FOLDS for seed in SEEDS
    }
    observed_entries: set[tuple[str, int, int]] = set()
    recipes: dict[str, dict[str, Any]] = {}
    for entry in prereg.get("configs", []):
        arm_name = str(entry.get("arm_name"))
        fold = int(entry.get("fold", -1))
        seed = int(entry.get("seed", -1))
        key = (arm_name, fold, seed)
        if key in observed_entries or key not in expected_entries:
            raise ValueError(f"G1 matrix has duplicate/unexpected config entry {key}")
        observed_entries.add(key)
        path = matrix_dir / str(entry.get("file", ""))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"G1 matrix config checksum mismatch: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = MLSHeatmapConfig.model_validate(payload["training_config"]).model_dump(mode="json")
        arm = ARMS[arm_name]
        if (
            config["dataset_variant"] != "multitask_2p5d_v1"
            or config["context_cache_manifest_sha256"] != cache_sha
            or config["context_cache_validation_receipt_sha256"] != cache_receipt_sha
            or int(config["input_channels"]) != arm["input_channels"]
            or int(config["fold"]) != fold
            or int(config["seed"]) != seed
            or int(config["epochs"]) != 23
            or int(config["snapshot_start_epoch"]) != 15
            or config["training_determinism"] != "strict"
            or config.get("resume_checkpoint") is not None
        ):
            raise ValueError(f"G1 config violates its immutable recipe: {path.name}")
        signature = _model_config_signature(config)
        if _canonical_sha256(signature) != entry.get("model_config_signature_sha256"):
            raise ValueError(f"G1 config signature mismatch: {path.name}")
        prior = recipes.get(arm_name)
        if prior is None:
            recipes[arm_name] = signature
        elif prior != signature:
            raise ValueError(f"G1 configs within arm {arm_name} differ beyond fold/seed")
    if observed_entries != expected_entries:
        raise ValueError("G1 matrix is not the complete 2 arms × 2 folds × 3 seeds product")
    control = recipes["control"]
    candidate = recipes["candidate"]
    differences = sorted({
        key for key in control.keys() | candidate.keys()
        if control.get(key) != candidate.get(key)
    })
    if differences != ["input_channels"] or prereg.get("training_factor_diff_vs_c0") != differences:
        raise ValueError("G1 C0/candidate are not a pure input-channel causal comparison")
    for name, arm in ARMS.items():
        declared = prereg.get("arms", {}).get(name, {})
        if (
            declared.get("arm") != arm["arm"]
            or int(declared.get("input_channels", -1)) != arm["input_channels"]
            or declared.get("model_config_signature") != recipes[name]
            or declared.get("model_config_signature_sha256") != _canonical_sha256(recipes[name])
        ):
            raise ValueError(f"G1 preregistration arm identity mismatch for {name}")
    source_hashes = prereg.get("source_sha256", {})
    for name, path in SOURCE_FILES.items():
        if source_hashes.get(name) != sha256_file(path):
            raise ValueError(f"G1 source hash changed after materialization: {name}")
    if prereg.get("stages") != {
        "fold3_screen": {"fold": 3, "studies": FOLDS[3]},
        "fold4_confirmation": {"fold": 4, "studies": FOLDS[4]},
    }:
        raise ValueError("G1 preregistration fold3/fold4 exposure plan changed")
    return {
        "schema_version": 1,
        "status": "passed",
        "campaign": "g1_2p5d_deploy_aligned",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_dir": str(matrix_dir.resolve()),
        "preregistration_sha256": sha256_file(prereg_path),
        "cache_manifest_sha256": cache_sha,
        "configs": len(observed_entries),
        "training_factor_diff_vs_c0": differences,
        "model_compute": "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    result = validate(
        matrix_dir=args.matrix_dir.resolve(),
        expected_cache_sha256=args.expected_cache_manifest_sha256,
    )
    receipt = args.receipt or (args.matrix_dir / "matrix_validation_receipt.json")
    _atomic_json(receipt.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
