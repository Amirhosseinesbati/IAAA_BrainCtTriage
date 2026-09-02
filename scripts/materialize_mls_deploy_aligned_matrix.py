"""Materialize the preregistered fold/seed matrix without changing model factors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.config_models import MLSHeatmapConfig


SEEDS = (42, 2026, 3407)
FOLDS = (0, 1, 2)
TEMPLATES = {
    "baseline": PROJECT_ROOT / "config/experiments/mls-vast-deploy-aligned-baseline-template.yaml",
    "a1_ordinal": PROJECT_ROOT / "config/experiments/mls-vast-deploy-aligned-a1-ordinal-template.yaml",
}
BASELINE_SEED42_SOURCES = {
    0: "mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth",
    1: "mls-vast-exp17-w32-fold1-strict-ensemble-refresh/mls_multitask_epoch_015.pth",
    2: "mls-vast-exp15r-w32-fold2-strict-repro-control/mls_multitask_epoch_015.pth",
}
EXPECTED_A1_TRAINING_DIFF = {
    "use_ordinal_aux_head",
    "ordinal_head_loss_weight",
}


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialize(stage: str, output_dir: Path) -> dict[str, object]:
    template_path = TEMPLATES[stage]
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    prefix = "baseline" if stage == "baseline" else "a1-ordinal"
    for fold in FOLDS:
        for seed in SEEDS:
            payload = copy.deepcopy(template)
            run_name = f"mls-vast-da-{prefix}-fold{fold}-seed{seed}"
            payload["run_name"] = run_name
            payload["tags"]["fold"] = fold
            payload["tags"]["seed"] = seed
            payload["tags"]["matrix_stage"] = stage
            payload["training_config"]["fold"] = fold
            payload["training_config"]["seed"] = seed
            resolved = MLSHeatmapConfig.model_validate(payload["training_config"])
            payload["training_config"] = resolved.model_dump(mode="json")
            output = output_dir / f"{run_name}.yaml"
            output.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            reuse_source = None
            execution_required = True
            if stage == "baseline" and seed == 42:
                reuse_source = BASELINE_SEED42_SOURCES[fold]
                execution_required = False
            rows.append({
                "stage": stage,
                "fold": fold,
                "seed": seed,
                "run_name": run_name,
                "config_path": output.relative_to(PROJECT_ROOT).as_posix(),
                "training_config_sha256": canonical_sha256(payload["training_config"]),
                "execution_required": execution_required,
                "reuse_source": reuse_source,
            })
    manifest = {
        "schema_version": 1,
        "status": "materialized_not_launched",
        "stage": stage,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "fixed_audit_epoch": 15,
        "adaptive_checkpoint_selection_allowed": False,
        "runs": rows,
    }
    if stage == "a1_ordinal":
        baseline = yaml.safe_load(TEMPLATES["baseline"].read_text(encoding="utf-8"))[
            "training_config"
        ]
        candidate = yaml.safe_load(TEMPLATES["a1_ordinal"].read_text(encoding="utf-8"))[
            "training_config"
        ]
        baseline = MLSHeatmapConfig.model_validate(baseline).model_dump(mode="json")
        candidate = MLSHeatmapConfig.model_validate(candidate).model_dump(mode="json")
        observed_diff = {
            key for key in baseline.keys() | candidate.keys()
            if baseline.get(key) != candidate.get(key)
        }
        if observed_diff != EXPECTED_A1_TRAINING_DIFF:
            raise ValueError(
                "A1 template is not a one-factor ablation: "
                f"expected {sorted(EXPECTED_A1_TRAINING_DIFF)}, got {sorted(observed_diff)}"
            )
        manifest["training_factor_diff_vs_baseline"] = sorted(observed_diff)
    manifest_path = output_dir / f"{stage}_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(TEMPLATES), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "config/experiments/generated/mls-deploy-aligned-20260902",
    )
    args = parser.parse_args()
    manifest = materialize(args.stage, args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "stage": manifest["stage"],
        "runs": len(manifest["runs"]),
    }))


if __name__ == "__main__":
    main()
