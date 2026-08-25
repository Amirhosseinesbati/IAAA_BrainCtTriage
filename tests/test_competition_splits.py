from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.splits import (
    build_nnunet_splits,
    normalize_study_id,
    split_items_by_fold,
    split_study_ids,
    study_id_from_path,
)


class TestCompetitionSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest_path = Path(self.tmp.name) / "folds.csv"
        rows = []
        for index in range(10):
            rows.append({
                "study_id": str(1000 + index),
                "patient_id": str(2000 + index),
                "triage_class": index % 3,
                "fold": index % 5,
            })
        pd.DataFrame(rows).to_csv(self.manifest_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalizes_csv_numeric_ids_and_nifti_names(self):
        self.assertEqual(normalize_study_id("1011.0"), "1011")
        self.assertEqual(study_id_from_path("BRN_1011_0000.nii.gz"), "1011")
        self.assertEqual(study_id_from_path("1011_slice0001.png"), "1011")

    def test_shared_fold_partition_is_disjoint_and_complete(self):
        studies = [str(1000 + index) for index in range(10)]
        train, val = split_study_ids(studies, 2, manifest_path=self.manifest_path)
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(studies))
        self.assertEqual(val, {"1002", "1007"})

    def test_items_keep_input_order(self):
        items = [Path(f"BRN_{1000 + index}.nii.gz") for index in range(10)]
        train, val = split_items_by_fold(
            items, 1, id_getter=study_id_from_path, manifest_path=self.manifest_path,
        )
        self.assertEqual([study_id_from_path(item) for item in val], ["1001", "1006"])
        self.assertEqual(len(train), 8)

    def test_unknown_study_is_rejected_instead_of_silently_leaking(self):
        with self.assertRaisesRegex(ValueError, "missing from"):
            split_study_ids(["9999", "1000"], 0, manifest_path=self.manifest_path)

    def test_nnunet_split_uses_case_names_but_shared_membership(self):
        cases = [f"BRN_{1000 + index}" for index in range(10)]
        splits = build_nnunet_splits(cases, manifest_path=self.manifest_path)
        self.assertEqual(splits[3]["val"], ["BRN_1003", "BRN_1008"])
        self.assertFalse(set(splits[3]["train"]) & set(splits[3]["val"]))


if __name__ == "__main__":
    unittest.main()
