from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_ich_2p5d_segmentation_oof import _load_run_artifacts


class ICHOOFArtifactResolutionTests(unittest.TestCase):
    def test_resolves_locked_outer_hybrid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "hybrid_summary.json").write_text(
                json.dumps({
                    "evaluation_split": "outer_fold",
                    "outer_fold": 3,
                    "reference_labels": ["IVH", "SAH"],
                    "candidate_labels": ["IPH", "SDH", "EDH"],
                    "any_ich_source": "reference",
                    "reference": {"checkpoint_sha256": "reference-sha"},
                    "candidate": {"checkpoint_sha256": "candidate-sha"},
                }),
                encoding="utf-8",
            )
            artifacts = _load_run_artifacts(root)
            self.assertEqual(artifacts.outer_fold, 3)
            self.assertEqual(
                artifacts.predictions_path, root / "hybrid_slice_predictions.csv"
            )
            self.assertEqual(
                artifacts.provenance["artifact_kind"], "locked_channel_hybrid"
            )

    def test_rejects_calibration_hybrid_as_oof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "hybrid_summary.json").write_text(
                json.dumps({
                    "evaluation_split": "calibration_only_no_outer",
                    "outer_fold": None,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not an outer-fold"):
                _load_run_artifacts(root)

    def test_rejects_outer_hybrid_without_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "hybrid_summary.json").write_text(
                json.dumps({"evaluation_split": "outer_fold", "outer_fold": None}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outer_fold provenance"):
                _load_run_artifacts(root)


if __name__ == "__main__":
    unittest.main()
