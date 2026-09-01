"""Explicit, CUDA-only MLS launcher for a persistent Vast workspace.

The launcher intentionally does not depend on the UI/deployment manifest code,
so concurrent edits in those shared modules cannot change a server run.  The
``--allow-training`` flag is a deliberate gate: bootstrap/readiness commands do
not start an experiment accidentally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies import get_mls_strategy
from src.strategies.config_models import MLSHeatmapConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Required acknowledgement that readiness passed and the training goal is active.",
    )
    args = parser.parse_args()
    if not args.allow_training:
        raise SystemExit(
            "Training gate is closed. Activate the agreed goal, then pass --allow-training."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only MLS launcher found no available GPU")

    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if payload.get("task") != "mls" or payload.get("strategy") != "mls_heatmap":
        raise ValueError("Vast MLS launcher accepts only task=mls, strategy=mls_heatmap")
    config = MLSHeatmapConfig.model_validate(payload.get("training_config", {}))
    run_name = str(payload.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Manifest run_name is required")
    os.environ["IAAA_RUN_NAME"] = run_name
    os.environ["IAAA_RUN_NOTES"] = str(payload.get("notes", ""))
    os.environ["IAAA_RUN_TAGS_JSON"] = json.dumps(payload.get("tags", {}))
    os.environ["IAAA_RUN_SOURCE"] = "vast_persistent_manifest"

    strategy = get_mls_strategy("mls_heatmap")
    validated = strategy.validate_config(config.model_dump())
    if bool(payload.get("runtime", {}).get("prepare_data", False)):
        if not strategy.prepare_data(validated):
            raise RuntimeError("MLS data preparation failed")
    if not strategy.train(validated):
        raise RuntimeError("MLS training strategy returned failure")


if __name__ == "__main__":
    main()
