from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.assert_mls_final_promotion_gate import EXPECTED_HARD_GATES, authorize


def _valid_summary() -> dict:
    fold_sizes = {0: 70, 1: 67, 2: 67, 3: 66, 4: 68}
    fold_sources = [
        {
            "fold": fold,
            "studies": fold_sizes[fold],
            "sha256": f"{fold + 1:x}" * 64,
            "audit_summary_sha256": f"{fold + 6:x}" * 64,
            "checkpoint_sha256": {
                "seed42": "a" * 64,
                "seed2026": "b" * 64,
                "seed3407": "c" * 64,
            },
        }
        for fold in range(5)
    ]
    return {
        "schema_version": 1,
        "protocol": "deploy_aligned_fixed_three_seed_median_canonical_triage",
        "promotion_eligible": True,
        "evaluation_scope": "full_oof",
        "full_fold_coverage": True,
        "selected_folds": [0, 1, 2, 3, 4],
        "available_folds": [0, 1, 2, 3, 4],
        "studies": 338,
        "promotion_gates": {name: True for name in EXPECTED_HARD_GATES},
        "failed_hard_gates": [],
        "sources": {
            "baseline_folds": fold_sources,
            "candidate_folds": [dict(row) for row in fold_sources],
            "frozen_champion_predictions": {
                "studies": 338,
                "sha256": "d" * 64,
                "expected_sha256": "d" * 64,
            },
        },
    }


class FinalPromotionGateTests(unittest.TestCase):
    def test_urgent_f1_requires_strict_improvement(self) -> None:
        self.assertIn("urgent_f1_improved", EXPECTED_HARD_GATES)
        self.assertNotIn("urgent_f1_noninferior", EXPECTED_HARD_GATES)

    def test_authorizes_only_checksum_bound_full_oof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(json.dumps(_valid_summary()), encoding="utf-8")
            output = root / "authorization.json"
            result = authorize(summary, output)
            self.assertEqual(
                result["status"], "authorized_for_clean_submission_packaging"
            )
            self.assertFalse(result["zip_created"])
            self.assertTrue(output.is_file())

    def test_refuses_three_fold_development_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _valid_summary()
            payload.update({
                "promotion_eligible": False,
                "evaluation_scope": "development_oof_subset",
                "full_fold_coverage": False,
                "selected_folds": [0, 1, 2],
                "studies": 204,
                "promotion_gates": {
                    **{name: True for name in EXPECTED_HARD_GATES},
                    "full_immutable_fold_coverage": False,
                },
                "failed_hard_gates": ["full_immutable_fold_coverage"],
            })
            summary = root / "summary.json"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refused"):
                authorize(summary, root / "authorization.json")

    def test_refuses_fabricated_minimal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "promotion_eligible": True,
                "evaluation_scope": "full_oof",
                "full_fold_coverage": True,
                "selected_folds": [0, 1, 2, 3, 4],
                "available_folds": [0, 1, 2, 3, 4],
                "studies": 338,
                "promotion_gates": {"full_immutable_fold_coverage": True},
                "failed_hard_gates": [],
            }
            summary = root / "summary.json"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "all_hard_gates_present"):
                authorize(summary, root / "authorization.json")

    def test_refuses_unpinned_frozen_champion_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _valid_summary()
            payload["sources"]["frozen_champion_predictions"]["expected_sha256"] = "e" * 64
            summary = root / "summary.json"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expected_hash_matches"):
                authorize(summary, root / "authorization.json")


if __name__ == "__main__":
    unittest.main()
