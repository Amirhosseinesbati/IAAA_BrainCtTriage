from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.assert_mls_final_promotion_gate import authorize


def _valid_summary() -> dict:
    return {
        "promotion_eligible": True,
        "evaluation_scope": "full_oof",
        "full_fold_coverage": True,
        "selected_folds": [0, 1, 2, 3, 4],
        "available_folds": [0, 1, 2, 3, 4],
        "studies": 338,
        "promotion_gates": {"full_immutable_fold_coverage": True},
        "failed_hard_gates": [],
    }


class FinalPromotionGateTests(unittest.TestCase):
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
                "promotion_gates": {"full_immutable_fold_coverage": False},
                "failed_hard_gates": ["full_immutable_fold_coverage"],
            })
            summary = root / "summary.json"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refused"):
                authorize(summary, root / "authorization.json")


if __name__ == "__main__":
    unittest.main()
