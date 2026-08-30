from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_fracture_smooth_attention_mil import (
    StudyBag,
    _fit_standardizer,
    _inner_splits,
    _public_mlflow_artifacts,
    _selected_final_epochs,
    _study_bags,
)


def test_study_bags_are_slice_ordered_and_grouped() -> None:
    frame = pd.DataFrame(
        {
            "study_id": ["a", "a", "b"],
            "patient_id": ["pa", "pa", "pb"],
            "study_fracture": [1, 1, 0],
            "outer_fold": [0, 0, 1],
            "slice_index": [2, 0, 0],
        }
    )
    bags = _study_bags(frame)
    assert [bag.study_id for bag in bags] == ["a", "b"]
    assert bags[0].indices.tolist() == [1, 0]


def test_standardizer_uses_only_selected_training_bags() -> None:
    embeddings = np.asarray([[1.0, 3.0], [3.0, 7.0], [100.0, 100.0]])
    scores = np.asarray([0.2, 0.8, 0.99])
    bag = StudyBag("a", "pa", 1, 0, np.asarray([0, 1]))
    stats = _fit_standardizer(embeddings, scores, [bag])
    np.testing.assert_allclose(stats.embedding_mean, [2.0, 5.0])
    np.testing.assert_allclose(stats.embedding_scale, [1.0, 2.0])


def test_inner_splits_keep_patients_disjoint() -> None:
    bags = [
        StudyBag(str(i), f"p{i}", i % 2, 0, np.asarray([i])) for i in range(12)
    ]
    for train_index, validation_index in _inner_splits(bags, n_splits=3, seed=7):
        train_patients = {bags[int(i)].patient_id for i in train_index}
        validation_patients = {bags[int(i)].patient_id for i in validation_index}
        assert train_patients.isdisjoint(validation_patients)


def test_final_epochs_preserve_early_nested_best() -> None:
    assert _selected_final_epochs({"median_best_epoch": 1}) == 1
    assert _selected_final_epochs({"median_best_epoch": 9}) == 9


def test_public_mlflow_artifacts_exclude_private_predictions(tmp_path) -> None:
    for name in (
        "metrics.json",
        "mlflow_run.json",
        "model_seed42.pt",
        "study_predictions.csv",
        "slice_attention.csv",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert [path.name for path in _public_mlflow_artifacts(tmp_path)] == [
        "metrics.json",
        "mlflow_run.json",
        "model_seed42.pt",
    ]
