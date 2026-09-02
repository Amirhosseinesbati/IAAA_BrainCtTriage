"""Strictly validate a materialized MLS fold/seed matrix before launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.config_models import MLSHeatmapConfig


EXPECTED_FOLDS = (0, 1, 2)
EXPECTED_SEEDS = (42, 2026, 3407)
ALLOWED_WITHIN_STAGE_CONFIG_DIFFERENCES = {"fold", "seed"}
EXPECTED_A1_DIFF = {"use_ordinal_aux_head", "ordinal_head_loss_weight"}


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _differences(payloads: list[dict[str, Any]]) -> set[str]:
    reference = payloads[0]
    output: set[str] = set()
    for candidate in payloads[1:]:
        for key in reference.keys() | candidate.keys():
            if reference.get(key) != candidate.get(key):
                output.add(key)
    return output


def validate_matrix(matrix_path: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    matrix_path = matrix_path.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    stage = str(matrix.get("stage"))
    if stage not in {"baseline", "a1_ordinal"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if tuple(matrix.get("folds", ())) != EXPECTED_FOLDS:
        raise ValueError("Fold contract must be exactly [0, 1, 2]")
    if tuple(matrix.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Seed contract must be exactly [42, 2026, 3407]")
    if int(matrix.get("fixed_audit_epoch", -1)) != 15:
        raise ValueError("Fixed audit epoch must remain 15")
    if matrix.get("adaptive_checkpoint_selection_allowed") is not False:
        raise ValueError("Adaptive checkpoint selection must be disabled")

    rows = matrix.get("runs")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("Matrix must contain exactly nine runs")
    expected_pairs = set(product(EXPECTED_FOLDS, EXPECTED_SEEDS))
    observed_pairs = {(int(row["fold"]), int(row["seed"])) for row in rows}
    if observed_pairs != expected_pairs or len(observed_pairs) != len(rows):
        raise ValueError("Matrix is not the unique 3x3 fold/seed Cartesian product")

    configs: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    execution_required = 0
    for row in rows:
        fold, seed = int(row["fold"]), int(row["seed"])
        expected_prefix = "baseline" if stage == "baseline" else "a1-ordinal"
        expected_run = f"mls-vast-da-{expected_prefix}-fold{fold}-seed{seed}"
        if row.get("stage") != stage or row.get("run_name") != expected_run:
            raise ValueError(f"Run identity mismatch for fold={fold}, seed={seed}")
        config_path = (project_root / str(row["config_path"])).resolve()
        allowed_root = (
            project_root / "config/experiments/generated/mls-deploy-aligned-20260902"
        ).resolve()
        if not config_path.is_relative_to(allowed_root) or not config_path.is_file():
            raise ValueError(f"Unsafe or missing config path: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if payload.get("task") != "mls" or payload.get("strategy") != "mls_heatmap":
            raise ValueError(f"Task/strategy mismatch: {config_path}")
        if payload.get("run_name") != expected_run:
            raise ValueError(f"Config run_name mismatch: {config_path}")
        tags = payload.get("tags", {})
        if (
            tags.get("matrix_stage") != stage
            or int(tags.get("fold", -1)) != fold
            or int(tags.get("seed", -1)) != seed
            or int(tags.get("fixed_audit_epoch", -1)) != 15
        ):
            raise ValueError(f"Config tags violate locked matrix: {config_path}")
        config = MLSHeatmapConfig.model_validate(payload["training_config"]).model_dump(
            mode="json"
        )
        if int(config["fold"]) != fold or int(config["seed"]) != seed:
            raise ValueError(f"Training fold/seed mismatch: {config_path}")
        if (
            int(config["snapshot_start_epoch"]) != 15
            or int(config["snapshot_every_n_epochs"]) != 100
            or int(config["epochs"]) != 23
            or config["training_determinism"] != "strict"
        ):
            raise ValueError(f"Fixed checkpoint or determinism contract changed: {config_path}")
        canonical_hash = _canonical_sha256(config)
        if canonical_hash != row.get("training_config_sha256"):
            raise ValueError(f"Stale canonical training hash: {config_path}")
        raw_hashes[expected_run] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        configs.append(config)

        should_execute = stage != "baseline" or seed != 42
        if bool(row.get("execution_required")) != should_execute:
            raise ValueError(f"Execution/reuse contract mismatch: {expected_run}")
        if should_execute:
            execution_required += 1
            if row.get("reuse_source") is not None:
                raise ValueError(f"Executable run unexpectedly has reuse source: {expected_run}")
        elif not str(row.get("reuse_source", "")).endswith("mls_multitask_epoch_015.pth"):
            raise ValueError(f"Seed42 reuse is not fixed epoch15: {expected_run}")

    observed_differences = _differences(configs)
    if observed_differences != ALLOWED_WITHIN_STAGE_CONFIG_DIFFERENCES:
        raise ValueError(
            "Unexpected within-stage model factor differences: "
            f"{sorted(observed_differences)}"
        )
    if stage == "a1_ordinal" and set(matrix.get("training_factor_diff_vs_baseline", ())) != EXPECTED_A1_DIFF:
        raise ValueError("A1 manifest does not declare the exact ordinal ablation")

    return {
        "schema_version": 1,
        "status": "validated",
        "stage": stage,
        "runs": len(rows),
        "execution_required": execution_required,
        "reused_fixed_epoch15": len(rows) - execution_required,
        "fixed_audit_epoch": 15,
        "within_stage_config_differences": sorted(observed_differences),
        "raw_config_sha256": raw_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_matrix(args.matrix), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
