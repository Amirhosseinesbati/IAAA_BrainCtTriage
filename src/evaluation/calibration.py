"""Cross-fitted monotonic calibration of the seven official intermediates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation.metrics import compute_competition_metrics
from src.evaluation.triage import VOLUME_KEYS, triage_from_intermediates
from src.inference.postprocessing import sanitize_intermediates

INTERMEDIATE_KEYS = (*VOLUME_KEYS, "fracture_prob", "MLS_mm")


@dataclass
class IsotonicMap:
    x: list[float]
    y: list[float]

    def transform(self, value: float) -> float:
        return float(np.interp(float(value), self.x, self.y, left=self.y[0], right=self.y[-1]))


@dataclass
class TriageCalibrator:
    mappings: dict[str, IsotonicMap] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        pred_prefix: str = "pred_",
        truth_prefix: str = "gt_",
    ) -> "TriageCalibrator":
        mappings: dict[str, IsotonicMap] = {}
        for key in INTERMEDIATE_KEYS:
            pred_col, truth_col = f"{pred_prefix}{key}", f"{truth_prefix}{key}"
            if pred_col not in frame or truth_col not in frame:
                raise ValueError(f"Calibration frame requires {pred_col!r} and {truth_col!r}")
            x = frame[pred_col].astype(float).to_numpy()
            y = frame[truth_col].astype(float).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                raise ValueError(f"Not enough finite values to calibrate {key}")
            if np.unique(x[valid]).size == 1:
                mappings[key] = IsotonicMap([float(x[valid][0])], [float(np.mean(y[valid]))])
                continue
            model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0)
            model.fit(x[valid], y[valid])
            mappings[key] = IsotonicMap(model.X_thresholds_.astype(float).tolist(), model.y_thresholds_.astype(float).tolist())
        return cls(mappings=mappings)

    def transform(self, values: Mapping[str, float]) -> dict[str, float]:
        missing = set(INTERMEDIATE_KEYS) - set(self.mappings)
        if missing:
            raise ValueError(f"Calibration bundle is incomplete: {sorted(missing)}")
        calibrated = {key: self.mappings[key].transform(float(values[key])) for key in INTERMEDIATE_KEYS}
        return sanitize_intermediates(calibrated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mappings": {key: {"x": mapping.x, "y": mapping.y} for key, mapping in self.mappings.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TriageCalibrator":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported calibration schema")
        return cls({
            key: IsotonicMap(list(value["x"]), list(value["y"]))
            for key, value in payload["mappings"].items()
        })

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TriageCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def transform_frame(
    frame: pd.DataFrame,
    calibrator: TriageCalibrator,
    *,
    pred_prefix: str = "pred_",
) -> pd.DataFrame:
    result = frame.copy()
    triage: list[int] = []
    for index, row in result.iterrows():
        raw = {key: row[f"{pred_prefix}{key}"] for key in INTERMEDIATE_KEYS}
        calibrated = calibrator.transform(raw)
        for key, value in calibrated.items():
            result.at[index, f"cal_{key}"] = value
        triage.append(triage_from_intermediates(calibrated))
    result["cal_triage"] = triage
    return result


def cross_validate_calibration(
    frame: pd.DataFrame,
    *,
    fold_column: str = "fold",
    target_column: str = "triage_class",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if fold_column not in frame or target_column not in frame:
        raise ValueError("Calibration frame requires fold and triage target columns")
    parts: list[pd.DataFrame] = []
    for fold in sorted(frame[fold_column].unique()):
        train = frame[frame[fold_column] != fold]
        validation = frame[frame[fold_column] == fold]
        calibrator = TriageCalibrator.fit(train)
        parts.append(transform_frame(validation, calibrator))
    calibrated = pd.concat(parts).sort_index()
    metrics = compute_competition_metrics(
        calibrated[target_column], calibrated["cal_triage"],
        patient_ids=calibrated["patient_id"] if "patient_id" in calibrated else None,
    )
    return calibrated, metrics
