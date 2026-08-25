from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.config import config_section, get_experiment_name
from src.mlops.tracking import build_source_snapshot, context_from_environment, flatten_mapping


class TestExperimentIsolation(unittest.TestCase):
    def test_task_experiment_names_are_unique(self):
        keys = list(config_section("mlflow", "experiments"))
        names = [get_experiment_name(key) for key in keys]
        self.assertEqual(len(names), len(set(names)))

    def test_environment_metadata_controls_run_not_experiment(self):
        with patch.dict(os.environ, {
            "IAAA_RUN_NAME": "fold-2-ablation",
            "IAAA_RUN_NOTES": "No MLS max aggregation",
            "IAAA_RUN_TAGS_JSON": json.dumps({"fold": 2, "owner": "test"}),
        }, clear=False):
            context = context_from_environment("mls_heatmap", "default", {"epochs": 1}, strategy="mls_heatmap")
        self.assertEqual(context.run_name, "fold-2-ablation")
        self.assertEqual(context.experiment_name, get_experiment_name("mls_heatmap"))
        self.assertEqual(context.tags["fold"], 2)


class TestArtifacts(unittest.TestCase):
    def test_snapshot_contains_code_config_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "snapshot.zip"
            manifest = build_source_snapshot(target)
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
            self.assertIn("config/project.yaml", names)
            self.assertIn("src/config.py", names)
            self.assertIn("snapshot_manifest.json", names)
            self.assertGreater(manifest["included_files"], 10)

    def test_nested_params_are_flattened(self):
        self.assertEqual(flatten_mapping({"model": {"lr": 0.1}}), {"model.lr": 0.1})


if __name__ == "__main__":
    unittest.main()
