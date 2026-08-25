from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.deploy.experiment import default_hardware, default_runtime
from src.deploy.ui_helpers import build_manifest, parse_tags, save_manifest, slugify


class TestTags(unittest.TestCase):
    def test_parse_tags(self):
        self.assertEqual(parse_tags("fold=2\nstage=ablation"), {"fold": "2", "stage": "ablation"})

    def test_invalid_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_tags("missing separator")


class TestManifestPersistence(unittest.TestCase):
    def test_manifest_is_saved_without_overwrite(self):
        hardware, runtime = default_hardware(), default_runtime()
        manifest = build_manifest(
            task="mls", strategy="mls_heatmap", run_name="Fold 0 baseline", notes="",
            tags={}, training_config={}, gpu_profile=hardware.gpu_profile,
            disk_gb=hardware.disk_gb, max_price_per_hour=hardware.max_price_per_hour,
            min_reliability=hardware.min_reliability, git_branch=runtime.git_branch,
            prepare_data=True, auto_destroy=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = save_manifest(manifest, Path(temp_dir))
            second = save_manifest(manifest, Path(temp_dir))
            self.assertNotEqual(first, second)
            self.assertEqual(manifest, type(manifest).from_yaml(first.read_text(encoding="utf-8")))

    def test_slug_is_safe(self):
        self.assertEqual(slugify(" Fold 0 / Baseline "), "fold-0-baseline")


if __name__ == "__main__":
    unittest.main()
