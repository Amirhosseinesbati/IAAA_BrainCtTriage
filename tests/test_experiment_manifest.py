from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.deploy.deploy import build_offer_query, select_offer
from src.deploy.experiment import ExperimentManifest, default_hardware, default_runtime


def manifest(**overrides):
    values = {
        "task": "mls",
        "strategy": "mls_heatmap",
        "run_name": "fold-0 baseline",
        "training_config": {"epochs": 10},
        "hardware": default_hardware(),
        "runtime": default_runtime(),
    }
    values.update(overrides)
    return ExperimentManifest(**values)


class TestManifest(unittest.TestCase):
    def test_yaml_and_base64_round_trip(self):
        original = manifest(notes="threshold-aware", tags={"fold": 0})
        self.assertEqual(original, ExperimentManifest.from_yaml(original.to_yaml()))
        self.assertEqual(original, ExperimentManifest.from_base64(original.to_base64()))
        self.assertEqual(original.task_key, "mls_heatmap")

    def test_invalid_task_strategy_is_rejected(self):
        with self.assertRaises(ValidationError):
            manifest(task="fracture", strategy="mls_heatmap")

    def test_branch_injection_is_rejected(self):
        with self.assertRaises(ValidationError):
            manifest(runtime={"git_branch": "main; echo bad", "prepare_data": True, "auto_destroy": True})

    def test_legacy_hardware_manifest_gets_new_filter_defaults(self):
        legacy_hardware = {
            "gpu_profile": "RTX_3090",
            "disk_gb": 40,
            "max_price_per_hour": 1.5,
            "min_reliability": 0.95,
        }
        loaded = manifest(hardware=legacy_hardware)
        self.assertEqual(loaded.hardware.min_price_per_hour, 0.0)
        self.assertEqual(loaded.hardware.min_download_mbps, 0.0)
        self.assertEqual(loaded.hardware.max_download_mbps, 100_000.0)
        self.assertEqual(loaded.hardware.min_cpu_cores, 1.0)
        self.assertFalse(loaded.hardware.top_k_enabled)
        self.assertEqual(loaded.hardware.top_k, 10)


class TestOfferSelection(unittest.TestCase):
    def test_query_comes_from_manifest_and_yaml(self):
        query = build_offer_query(manifest())
        self.assertIn("gpu_name=RTX_3090", query)
        self.assertIn("reliability>=0.95", query)
        self.assertIn("inet_down>=0.0", query)
        self.assertIn("cpu_cores_effective>=1.0", query)
        self.assertIn("dph_total<=0.15", query)
        self.assertIn("cuda_vers>=12.8", query)

    def test_cheapest_eligible_offer_is_selected(self):
        offers = [
            {"id": 1, "dph_total": 1.2, "reliability": 0.99},
            {"id": 2, "dph_total": 0.8, "reliability": 0.96},
            {"id": 3, "dph_total": 2.0, "reliability": 0.99},
        ]
        self.assertEqual(select_offer(offers, 1.5)["id"], 2)

    def test_best_offer_is_selected_from_top_k_cheapest(self):
        offers = [
            {"id": 1, "dph_total": 0.5, "score": 2.0, "dlperf_usd": 10, "reliability": 0.99},
            {"id": 2, "dph_total": 0.6, "score": 8.0, "dlperf_usd": 20, "reliability": 0.98},
            {"id": 3, "dph_total": 0.7, "score": 4.0, "dlperf_usd": 30, "reliability": 0.97},
            {"id": 4, "dph_total": 0.8, "score": 100.0, "dlperf_usd": 100, "reliability": 0.99},
        ]
        selected = select_offer(offers, 1.5, top_k_enabled=True, top_k=3)
        self.assertEqual(selected["id"], 2)

    def test_invalid_hardware_range_is_rejected(self):
        hardware = default_hardware().model_copy(
            update={"min_download_mbps": 500.0, "max_download_mbps": 100.0},
        )
        with self.assertRaises(ValidationError):
            manifest(hardware=hardware.model_dump())


if __name__ == "__main__":
    unittest.main()
