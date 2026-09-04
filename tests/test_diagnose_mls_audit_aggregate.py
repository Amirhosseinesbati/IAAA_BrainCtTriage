from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnose_mls_audit_aggregate import diagnose


def _slice(index: int, *, mls: float, selector: float = 0.9) -> dict[str, float | int]:
    return {
        "index": index,
        "selector_probability": selector,
        "peak_probability": selector,
        "mls_mm": mls,
        "heatmap_peak": 0.8,
    }


class AggregateAuditDiagnosticTests(unittest.TestCase):
    def _write_private_fixture(self, path: Path) -> None:
        rows = [
            {"study_id": "private-a", "gt_MLS_mm": "0.5", "slice_predictions_json": json.dumps([_slice(0, mls=0.5), _slice(1, mls=0.5), _slice(2, mls=0.5)]), "error": ""},
            {"study_id": "private-b", "gt_MLS_mm": "3.4", "slice_predictions_json": json.dumps([_slice(0, mls=4.0), _slice(1, mls=4.0), _slice(2, mls=4.0)]), "error": ""},
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_output_is_aggregate_only_and_uses_fixed_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private.csv"
            output = root / "aggregate.json"
            self._write_private_fixture(source)
            result = diagnose(source, output, expected_studies=2)
            rendered = output.read_text(encoding="utf-8")
            self.assertEqual(result["n_studies"], 2)
            self.assertEqual(result["fixed_pooling"]["aggregation"], "p90")
            self.assertAlmostEqual(result["residual_summary"]["mae_mm"], 0.3, places=6)
            self.assertEqual(len(result["thresholds"]), 2)
            self.assertNotIn("private-a", rendered)
            self.assertNotIn("private-b", rendered)
            self.assertNotIn("slice_predictions_json", rendered)
            self.assertFalse(result["privacy"]["contains_study_ids"])

    def test_rejects_incomplete_or_failed_private_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private.csv"
            output = root / "aggregate.json"
            self._write_private_fixture(source)
            with self.assertRaisesRegex(ValueError, "Expected exactly 3"):
                diagnose(source, output, expected_studies=3)


if __name__ == "__main__":
    unittest.main()
