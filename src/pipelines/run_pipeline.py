"""Validated CLI entry point for local, UI and Vast experiment manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.deploy.experiment import ExperimentManifest
from src.pipelines.pipelines import (
    ich_pipeline, mls_strategy_pipeline, mls_pipeline, nnunet_pipeline, yolo_pipeline,
)
from src.strategies import get_mls_strategy, get_strategy, list_mls_strategies, list_strategies


def _print_strategies(items: list[dict], label: str) -> None:
    print(f"\nAvailable {label} strategies:")
    for item in items:
        print(f"  {item['name']:12s} - {item['display_name']}")


def _validated_json(payload: str) -> dict:
    value = json.loads(payload or "{}")
    if not isinstance(value, dict):
        raise ValueError("Training config must be a JSON object")
    return value


def _set_run_environment(manifest: ExperimentManifest) -> None:
    os.environ["IAAA_RUN_NAME"] = manifest.run_name
    os.environ["IAAA_RUN_NOTES"] = manifest.notes
    os.environ["IAAA_RUN_TAGS_JSON"] = json.dumps(manifest.tags)
    os.environ.setdefault("IAAA_RUN_SOURCE", "manifest")


def run_manifest(manifest: ExperimentManifest) -> None:
    _set_run_environment(manifest)
    config_json = json.dumps(manifest.training_config)
    if manifest.task == "ich":
        strategy = get_strategy(manifest.strategy)
        strategy.validate_config(manifest.training_config)
        ich_pipeline(manifest.strategy, config_json, manifest.runtime.prepare_data)
    elif manifest.task == "mls":
        strategy = get_mls_strategy(manifest.strategy)
        strategy.validate_config(manifest.training_config)
        mls_strategy_pipeline(manifest.strategy, config_json, manifest.runtime.prepare_data)
    elif manifest.task == "fracture":
        yolo_pipeline(config_json, manifest.runtime.prepare_data)
    elif manifest.task == "triage_calibration":
        from src.evaluation.calibration import (
            TriageCalibrator, assess_calibration_candidate, cross_validate_calibration,
        )
        from src.mlops import context_from_environment, experiment_run, log_run_summary
        import pandas as pd

        source = Path(manifest.training_config["oof_predictions"])
        output = Path(manifest.training_config.get("output", "models/calibration/triage_calibration.json"))
        context = context_from_environment(
            "triage_calibration", manifest.run_name, manifest.training_config,
            strategy="nested_isotonic",
        )
        with experiment_run(context):
            frame = pd.read_csv(source)
            calibrated, _ = cross_validate_calibration(frame)
            assessment = assess_calibration_candidate(frame, calibrated)
            candidate = TriageCalibrator.fit(frame)
            candidate_path = Path("reports/calibration/triage_calibration.candidate.json")
            candidate.save(candidate_path)
            if assessment["accepted"]:
                candidate.save(output)
            log_run_summary({
                "task": "triage_calibration",
                "source": str(source),
                "output": str(output) if assessment["accepted"] else None,
                "candidate_path": str(candidate_path),
                "assessment": assessment,
            })
            print(json.dumps(assessment, indent=2))
            if not assessment["accepted"]:
                raise RuntimeError(
                    "Nested-OOF calibration rejected: "
                    + ", ".join(assessment["rejection_reasons"])
                )
    else:  # guarded by Pydantic; keeps future schema changes explicit
        raise ValueError(f"Unsupported task: {manifest.task}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validated Brain CT experiments")
    parser.add_argument("--manifest", type=Path, help="Versioned YAML experiment manifest")
    parser.add_argument("--run", choices=["nnunet", "yolo", "mls", "mls-strategy", "ich", "all"])
    parser.add_argument("--strategy", default="nnunet")
    parser.add_argument("--config", default="{}")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--list-mls-strategies", action="store_true")
    args = parser.parse_args()

    if args.list_strategies:
        _print_strategies(list_strategies(), "ICH")
        return
    if args.list_mls_strategies:
        _print_strategies(list_mls_strategies(), "MLS")
        return
    if args.manifest:
        run_manifest(ExperimentManifest.from_yaml(args.manifest.read_text(encoding="utf-8")))
        return
    if not args.run:
        parser.error("one of --manifest or --run is required")

    config = _validated_json(args.config)
    prepare = not args.skip_prepare_data
    config_json = json.dumps(config)
    if args.run == "nnunet":
        nnunet_pipeline()
    elif args.run == "yolo":
        yolo_pipeline(config_json, prepare)
    elif args.run == "mls":
        mls_pipeline()
    elif args.run == "mls-strategy":
        get_mls_strategy(args.strategy).validate_config(config)
        mls_strategy_pipeline(args.strategy, config_json, prepare)
    elif args.run == "ich":
        get_strategy(args.strategy).validate_config(config)
        ich_pipeline(args.strategy, config_json, prepare)
    else:
        nnunet_pipeline()
        yolo_pipeline(config_json, prepare)
        mls_strategy_pipeline("mls_heatmap", "{}", prepare)


if __name__ == "__main__":
    main()
