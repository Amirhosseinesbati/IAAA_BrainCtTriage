"""
test_leaderboard_helpers.py — Regression tests for the leaderboard + submission.

Guards two bugs fixed for the mls_heatmap evaluation flow:
1. ``normalize_study_id`` — CSV round-trip turns integer study ids into
   floats (``1011`` → ``1011.0``), breaking GT↔prediction matching.
2. Submission ``_MLSHeatmapModel`` state dict keys MUST match the
   training-side ``HRNetHeatmapModel`` exactly (a bare ``nn.Sequential``
   head produced ``head.0.*`` keys that never loaded — silent garbage).
"""

from __future__ import annotations

import unittest

from leaderboard import normalize_study_id


class TestNormalizeStudyId(unittest.TestCase):
    """Study-id canonicalization (float/int/str mismatch)."""

    def test_integer_float(self):
        """1011.0 (float from CSV) → '1011' matching the int-derived key."""
        self.assertEqual(normalize_study_id(1011.0), "1011")

    def test_integer(self):
        self.assertEqual(normalize_study_id(1011), "1011")

    def test_string(self):
        self.assertEqual(normalize_study_id("1011"), "1011")

    def test_string_already_suffixed(self):
        """A genuine string id with a decimal is left untouched."""
        self.assertEqual(normalize_study_id("1011.5"), "1011.5")

    def test_consistency(self):
        """int, float and str forms of the same id all canonicalize equal."""
        self.assertEqual(
            normalize_study_id(1011),
            normalize_study_id(1011.0),
        )
        self.assertEqual(
            normalize_study_id(1011.0),
            normalize_study_id("1011"),
        )


class TestSubmissionStateDictMatchesTraining(unittest.TestCase):
    """
    The submission-side heatmap model must accept the training-side state
    dict with strict=True. This is the regression guard for the head
    key-mismatch bug (head.0.* vs head.conv1.*).
    """

    def test_w18_3ch_strict_load(self):
        import torch

        import submission.model as sm
        from src.strategies.mls_heatmap.model import HRNetHeatmapModel

        train_model = HRNetHeatmapModel(
            backbone_name="hrnet_w18", in_channels=3, pretrained=False
        )
        sub_model = sm._MLSHeatmapModel(
            backbone_name="hrnet_w18", in_channels=3
        )

        train_keys = set(train_model.state_dict().keys())
        sub_keys = set(sub_model.state_dict().keys())
        self.assertEqual(train_keys, sub_keys, "state dict keys must match")

        # strict=True load proves every weight transfers (incl. the head).
        sub_model.load_state_dict(train_model.state_dict(), strict=True)

    def test_w18_1ch_strict_load(self):
        import torch

        import submission.model as sm
        from src.strategies.mls_heatmap.model import HRNetHeatmapModel

        train_model = HRNetHeatmapModel(
            backbone_name="hrnet_w18", in_channels=1, pretrained=False
        )
        sub_model = sm._MLSHeatmapModel(
            backbone_name="hrnet_w18", in_channels=1
        )

        train_keys = set(train_model.state_dict().keys())
        sub_keys = set(sub_model.state_dict().keys())
        self.assertEqual(train_keys, sub_keys)
        sub_model.load_state_dict(train_model.state_dict(), strict=True)

        # 1-channel forward pass works.
        x = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            out = sub_model(x)
        self.assertEqual(tuple(out.shape), (1, 3, 128, 128))

    def test_w32_3ch_strict_load(self):
        import submission.model as sm
        from src.strategies.mls_heatmap.model import HRNetHeatmapModel

        train_model = HRNetHeatmapModel(
            backbone_name="hrnet_w32", in_channels=3, pretrained=False
        )
        sub_model = sm._MLSHeatmapModel(
            backbone_name="hrnet_w32", in_channels=3
        )

        train_keys = set(train_model.state_dict().keys())
        sub_keys = set(sub_model.state_dict().keys())
        self.assertEqual(train_keys, sub_keys)
        sub_model.load_state_dict(train_model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
