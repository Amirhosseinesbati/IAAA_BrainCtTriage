from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.deploy.experiment import default_hardware, default_runtime
from src.strategies.config_models import MLSHeatmapConfig
from src.deploy.ui_helpers import (
    build_manifest, expand_fold_suite, parse_tags, save_manifest, slugify,
)


class TestTags(unittest.TestCase):
    def test_parse_tags(self):
        self.assertEqual(parse_tags("fold=2\nstage=ablation"), {"fold": "2", "stage": "ablation"})

    def test_invalid_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_tags("missing separator")


class TestSmallFloatHyperparameters(unittest.TestCase):
    def test_mls_accepts_small_adamw_values(self):
        config = MLSHeatmapConfig(learning_rate=0.0001, weight_decay=0.0001)
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertEqual(config.weight_decay, 0.0001)


class TestManifestPersistence(unittest.TestCase):
    def test_manifest_is_saved_without_overwrite(self):
        hardware, runtime = default_hardware(), default_runtime()
        manifest = build_manifest(
            task="mls", strategy="mls_heatmap", run_name="Fold 0 baseline", notes="",
            tags={}, training_config={}, gpu_profile=hardware.gpu_profile,
            disk_gb=hardware.disk_gb, min_price_per_hour=hardware.min_price_per_hour,
            max_price_per_hour=hardware.max_price_per_hour,
            min_reliability=hardware.min_reliability,
            min_download_mbps=hardware.min_download_mbps,
            max_download_mbps=hardware.max_download_mbps,
            min_cpu_cores=hardware.min_cpu_cores, max_cpu_cores=hardware.max_cpu_cores,
            top_k_enabled=hardware.top_k_enabled, top_k=hardware.top_k,
            git_branch=runtime.git_branch,
            prepare_data=True, auto_destroy=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = save_manifest(manifest, Path(temp_dir))
            second = save_manifest(manifest, Path(temp_dir))
            self.assertNotEqual(first, second)
            self.assertEqual(manifest, type(manifest).from_yaml(first.read_text(encoding="utf-8")))

    def test_slug_is_safe(self):
        self.assertEqual(slugify(" Fold 0 / Baseline "), "fold-0-baseline")

    def test_expand_fold_suite_sets_config_name_and_tags(self):
        hardware, runtime = default_hardware(), default_runtime()
        manifest = build_manifest(
            task="mls", strategy="mls_heatmap", run_name="threshold-aware-fold-0", notes="",
            tags={"stage": "ablation"}, training_config={"epochs": 10},
            gpu_profile=hardware.gpu_profile, disk_gb=hardware.disk_gb,
            min_price_per_hour=hardware.min_price_per_hour,
            max_price_per_hour=hardware.max_price_per_hour,
            min_reliability=hardware.min_reliability, git_branch=runtime.git_branch,
            min_download_mbps=hardware.min_download_mbps,
            max_download_mbps=hardware.max_download_mbps,
            min_cpu_cores=hardware.min_cpu_cores, max_cpu_cores=hardware.max_cpu_cores,
            top_k_enabled=hardware.top_k_enabled, top_k=hardware.top_k,
            prepare_data=True, auto_destroy=True,
        )
        suite = expand_fold_suite(manifest)
        self.assertEqual(len(suite), 5)
        self.assertEqual([item.training_config["fold"] for item in suite], list(range(5)))
        self.assertEqual(suite[4].run_name, "threshold-aware-fold-4")
        self.assertEqual(suite[3].tags["fold"], 3)


if __name__ == "__main__":
    unittest.main()
