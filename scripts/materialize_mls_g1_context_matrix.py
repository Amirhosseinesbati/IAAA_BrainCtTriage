"""Materialize the immutable G1 C0-vs-9-channel experiment matrix.

This is deliberately separate from historical deploy-aligned materializers so
their source hashes and evidence remain untouched.  It creates all fold-3 and
fold-4 configs before any G1 CUDA outcome is observed; the caller may execute
only the six fold-3 configs until the staged triage gate permits fold 4.
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

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_mls_g1_three_seed_fold_cuda import _canonical_sha256, _model_config_signature
from scripts.evaluate_mls_g1_three_seed_fold_cuda import _load_passing_cache_receipt
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.context_cache import sha256_file


TEMPLATE = PROJECT_ROOT / "config" / "experiments" / "mls-vast-deploy-aligned-baseline-template.yaml"
FOLDS = {3: 66, 4: 68}
SEEDS = (42, 2026, 3407)
ARMS = {
    "control": {"arm": "g1_c0_3ch", "input_channels": 3, "slug": "c0-central3"},
    "candidate": {"arm": "g1_a_9ch", "input_channels": 9, "slug": "a-context9"},
}
SOURCE_FILES = {
    "materializer": Path(__file__),
    "validator": PROJECT_ROOT / "scripts" / "validate_mls_g1_context_matrix.py",
    "g1_evaluator": PROJECT_ROOT / "scripts" / "evaluate_mls_g1_three_seed_fold_cuda.py",
    "g1_qualification": PROJECT_ROOT / "scripts" / "qualify_mls_g1_2p5d_runtime_cuda.py",
    "g1_staged_gate": PROJECT_ROOT / "scripts" / "evaluate_mls_g1_staged_triage_gate.py",
    "cache_builder": PROJECT_ROOT / "scripts" / "build_mls_2p5d_cache.py",
    "cache_validator": PROJECT_ROOT / "scripts" / "validate_mls_2p5d_cache.py",
    "input_contract": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "input_contract.py",
    "context_cache": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "context_cache.py",
    "dataset": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "dataset.py",
    "model": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "model.py",
    "predict_multitask": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "predict_multitask.py",
    "mls_utils": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "utils.py",
    "train_multitask": PROJECT_ROOT / "src" / "strategies" / "mls_heatmap" / "train_multitask.py",
    "config_models": PROJECT_ROOT / "src" / "strategies" / "config_models.py",
    "dicom_reader": PROJECT_ROOT / "src" / "preprocessing" / "core" / "dicom_reader.py",
    "canonical_triage_reducer": PROJECT_ROOT / "scripts" / "evaluate_mls_deploy_aligned_seed_medians.py",
    "triage_rules": PROJECT_ROOT / "src" / "evaluation" / "triage.py",
}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
        text=True, check=True,
    )
    return completed.stdout.strip()


def _normalise_config(config: dict[str, Any]) -> dict[str, Any]:
    """Use Pydantic's serialized values so evaluator and matrix hashes agree."""
    return MLSHeatmapConfig.model_validate(config).model_dump(mode="json")


def _config_payload(
    template: dict[str, Any],
    *,
    arm: dict[str, Any],
    fold: int,
    seed: int,
    cache_manifest_sha256: str,
    cache_validation_receipt_sha256: str,
) -> dict[str, Any]:
    output = json.loads(json.dumps(template))
    training = output["training_config"]
    training.update({
        "fold": fold,
        "seed": seed,
        "dataset_variant": "multitask_2p5d_v1",
        "context_cache_manifest_sha256": cache_manifest_sha256,
        "context_cache_validation_receipt_sha256": cache_validation_receipt_sha256,
        "input_channels": arm["input_channels"],
        "resume_checkpoint": None,
    })
    output["run_name"] = f"mls-vast-g1-{arm['slug']}-fold{fold}-seed{seed}"
    output["notes"] = (
        "Preregistered G1 deploy-aligned causal arm. The paired C0 and 9-channel "
        "arms share the same float32 cache, seed, fold and every training factor "
        "except input_channels. CUDA-only model compute; epoch 15 is fixed for audit."
    )
    output["tags"] = {
        "campaign_id": "g1_2p5d_20260905",
        "experiment_key": "G1",
        "stage": "fold3_execute" if fold == 3 else "fold4_conditional_manual",
        "arm": arm["arm"],
        "changed_training_factor": "input_channels_only",
        "input_channels": arm["input_channels"],
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_validation_receipt_sha256": cache_validation_receipt_sha256,
        "fixed_audit_epoch": 15,
        "compute_policy": "cuda_only_no_cpu_fallback",
    }
    # Machine metadata is non-causal and records the actual planned 3090 host.
    output["hardware"] = {"gpu_profile": "RTX_3090", "disk_gb": 100}
    output["runtime"] = {
        "git_branch": "codex/mls-a2-geometry-20260904",
        "prepare_data": False,
        "auto_destroy": False,
    }
    # Fail now rather than emitting a config that can only fail after GPU time.
    output["training_config"] = _normalise_config(training)
    return output


def materialize(
    *,
    cache_manifest_sha256: str,
    cache_validation_receipt: Path,
    output_dir: Path,
) -> dict[str, Any]:
    digest = cache_manifest_sha256.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("cache manifest must be a lowercase 64-character SHA-256 digest")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing immutable G1 matrix: {output_dir}")
    cache_receipt = _load_passing_cache_receipt(cache_validation_receipt, digest)
    cache_receipt_sha256 = sha256_file(cache_validation_receipt)
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or not isinstance(template.get("training_config"), dict):
        raise ValueError("G1 base template is invalid")
    template_sha256 = sha256_file(TEMPLATE)
    output_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, Any]] = []
    arm_signatures: dict[str, dict[str, Any]] = {}
    for arm_name, arm in ARMS.items():
        for fold in sorted(FOLDS):
            for seed in SEEDS:
                payload = _config_payload(
                    template, arm=arm, fold=fold, seed=seed,
                    cache_manifest_sha256=digest,
                    cache_validation_receipt_sha256=cache_receipt_sha256,
                )
                filename = f"mls-vast-g1-{arm['slug']}-fold{fold}-seed{seed}.yaml"
                path = output_dir / filename
                _atomic_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
                config = payload["training_config"]
                signature = _model_config_signature(config)
                if arm_name not in arm_signatures:
                    arm_signatures[arm_name] = signature
                elif arm_signatures[arm_name] != signature:
                    raise RuntimeError("G1 materializer emitted inconsistent seed/fold-free arm configs")
                entries.append({
                    "file": filename,
                    "sha256": sha256_file(path),
                    "arm": arm["arm"],
                    "arm_name": arm_name,
                    "fold": fold,
                    "studies": FOLDS[fold],
                    "seed": seed,
                    "model_config_signature_sha256": _canonical_sha256(signature),
                })
    control_signature = arm_signatures["control"]
    candidate_signature = arm_signatures["candidate"]
    differences = sorted({
        key for key in control_signature.keys() | candidate_signature.keys()
        if control_signature.get(key) != candidate_signature.get(key)
    })
    if differences != ["input_channels"]:
        raise RuntimeError(f"G1 matrix causal recipe differs in {differences}, not only input_channels")

    manifest = {
        "schema_version": 1,
        "status": "locked_before_any_g1_cuda_outcome",
        "campaign": "g1_2p5d_deploy_aligned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "template": str(TEMPLATE.relative_to(PROJECT_ROOT)),
        "template_sha256": template_sha256,
        "cache_manifest_sha256": digest,
        "cache_validation_receipt": str(cache_validation_receipt.resolve()),
        "cache_validation_receipt_sha256": cache_receipt_sha256,
        "cache_validation_at_utc": cache_receipt.get("validated_at_utc"),
        "source_sha256": {name: sha256_file(path) for name, path in SOURCE_FILES.items()},
        "fixed_audit_epoch": 15,
        "seeds": list(SEEDS),
        "stages": {
            "fold3_screen": {"fold": 3, "studies": FOLDS[3]},
            "fold4_confirmation": {"fold": 4, "studies": FOLDS[4]},
        },
        "execution_plan": {
            "fold3": "executable_now",
            "fold4": "conditional_manual_after_passed_fold3_gate",
        },
        "arms": {
            name: {
                "arm": arm["arm"],
                "input_channels": arm["input_channels"],
                "dataset_variant": "multitask_2p5d_v1",
                "cache_manifest_sha256": digest,
                "fixed_epoch": 15,
                "seeds": list(SEEDS),
                "model_config_signature": arm_signatures[name],
                "model_config_signature_sha256": _canonical_sha256(arm_signatures[name]),
            }
            for name, arm in ARMS.items()
        },
        "training_factor_diff_vs_c0": differences,
        "fixed_screen_gates": [
            "frozen_macro_f1_strictly_improved",
            "frozen_urgent_f1_strictly_improved",
            "frozen_accuracy_noninferior",
            "normal_f1_not_below_minus_0p01",
            "critical_f1_not_below_minus_0p01",
            "f1_3mm_noninferior",
            "f1_5mm_noninferior",
            "catastrophic_errors_not_worse",
            "oracle_macro_and_urgent_directions_nonnegative",
        ],
        "configs": entries,
        "model_compute": "none",
    }
    manifest_path = output_dir / "g1_preregistration.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "status": manifest["status"],
        "output_dir": str(output_dir.resolve()),
        "preregistration": str(manifest_path.resolve()),
        "preregistration_sha256": sha256_file(manifest_path),
        "configs": len(entries),
        "fold3_configs": sum(entry["fold"] == 3 for entry in entries),
        "fold4_configs": sum(entry["fold"] == 4 for entry in entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--cache-validation-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        cache_manifest_sha256=args.cache_manifest_sha256,
        cache_validation_receipt=args.cache_validation_receipt.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
