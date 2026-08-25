from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.folds import create_fold_manifest, validate_fold_manifest
from src.evaluation.metrics import compute_competition_metrics
from src.evaluation.triage import decision_margins, triage_from_intermediates, triage_rule_trace
from src.inference.triage_rules import apply_triage_rules
from submission.triage import triage_from_intermediates as submission_triage


def primitives(**overrides):
    values = {
        "V_EDH": 0.0, "V_SDH": 0.0, "V_IPH": 0.0,
        "V_SAH": 0.0, "V_IVH": 0.0,
        "fracture_prob": 0.0, "MLS_mm": 0.0,
    }
    values.update(overrides)
    return values


class TestOfficialTriage(unittest.TestCase):
    def test_submission_parity_at_boundaries(self):
        cases = [
            primitives(),
            primitives(V_IPH=0.1),
            primitives(MLS_mm=3.0),
            primitives(MLS_mm=5.0),
            primitives(V_EDH=30.0),
            primitives(V_SDH=70.0),
            primitives(V_IPH=70.0),
            primitives(V_IPH=40.0, MLS_mm=3.0),
            primitives(V_IPH=15.0, fracture_prob=0.5),
        ]
        for values in cases:
            self.assertEqual(triage_from_intermediates(values), submission_triage(values))

    def test_trace_and_margin(self):
        values = primitives(V_IPH=0.11)
        self.assertEqual(triage_rule_trace(values), (1, "urgent_any_ich"))
        self.assertAlmostEqual(decision_margins(values)["total_to_any_ich"], 0.01)

    def test_demo_wrapper_uses_official_rule(self):
        self.assertEqual(apply_triage_rules({"IPH": 0.11}, False, 0.0), "Level 2")
        self.assertEqual(apply_triage_rules({"IPH": 80.0}, False, 0.0), "Level 1")


class TestFolds(unittest.TestCase):
    def test_patient_never_crosses_folds(self):
        rows = []
        for patient in range(30):
            label = patient % 3
            for series in range(2 if patient < 5 else 1):
                rows.append({"study_id": f"{patient}-{series}", "patient_id": str(patient), "triage_class": label})
        manifest = create_fold_manifest(pd.DataFrame(rows), n_folds=5, seed=7)
        validate_fold_manifest(manifest, 5)
        self.assertTrue((manifest.groupby("patient_id").fold.nunique() == 1).all())


class TestMetrics(unittest.TestCase):
    def test_macro_f1_is_primary(self):
        metrics = compute_competition_metrics([0, 1, 2], [0, 1, 2], bootstrap_samples=10)
        self.assertEqual(metrics["official_metric"], "macro_f1")
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["catastrophic_errors"]["normal_to_critical"], 0)


if __name__ == "__main__":
    unittest.main()
