"""Run a validated MLS manifest locally without the optional ZenML store.

The local ZenML database can be newer than the pinned project client. This
launcher keeps the same manifest validation, strategy dispatch, run metadata,
CUDA guards and MLflow lifecycle while avoiding any mutation/migration of the
user's global ZenML database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy.experiment import ExperimentManifest
from src.strategies import get_mls_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = ExperimentManifest.from_yaml(args.manifest.read_text(encoding="utf-8"))
    if manifest.task != "mls":
        raise ValueError("Local MLS launcher accepts only task=mls manifests")
    os.environ["IAAA_RUN_NAME"] = manifest.run_name
    os.environ["IAAA_RUN_NOTES"] = manifest.notes
    os.environ["IAAA_RUN_TAGS_JSON"] = json.dumps(manifest.tags)
    os.environ["IAAA_RUN_SOURCE"] = "local_manifest_direct"
    config = manifest.effective_training_config()
    strategy = get_mls_strategy(manifest.strategy)
    validated = strategy.validate_config(config)
    if manifest.runtime.prepare_data and not strategy.prepare_data(validated):
        raise RuntimeError("MLS data preparation failed")
    if not strategy.train(validated):
        raise RuntimeError("MLS training strategy returned failure")


if __name__ == "__main__":
    main()
