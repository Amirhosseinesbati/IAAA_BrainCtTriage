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


class TestOfferSelection(unittest.TestCase):
    def test_query_comes_from_manifest_and_yaml(self):
        query = build_offer_query(manifest())
        self.assertIn("gpu_name=RTX_3090", query)
        self.assertIn("reliability>=0.95", query)

    def test_cheapest_eligible_offer_is_selected(self):
        offers = [
            {"id": 1, "dph_total": 1.2, "reliability": 0.99},
            {"id": 2, "dph_total": 0.8, "reliability": 0.96},
            {"id": 3, "dph_total": 2.0, "reliability": 0.99},
        ]
        self.assertEqual(select_offer(offers, 1.5)["id"], 2)


if __name__ == "__main__":
    unittest.main()
