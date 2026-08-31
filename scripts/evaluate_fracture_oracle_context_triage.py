"""Cross-fit fracture thresholds against the official triage rule in oracle context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.triage import triage_from_intermediates

VOLUME_AREAS = {
    "V_EDH": "EpiduralHemorrhage_Area",
    "V_SDH": "SubduralHemorrhage_Area",
    "V_IPH": "IntraparenchymalHemorrhage_Area",
    "V_SAH": "SubarachnoidHemorrhage_Area",
    "V_IVH": "IntraventricularHemorrhage_Area",
}


def _series_intermediates(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dicom_series.id",
        "dicom_series.PixelSpacing0",
        "dicom_series.PixelSpacing1",
        "dicom_series.SliceThickness",
        "SkullFracture",
        "MidlineShiftMM",
        "triage_class",
        *VOLUME_AREAS.values(),
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    frame = metadata.copy()
    factor = (
        frame["dicom_series.PixelSpacing0"].astype(float)
        * frame["dicom_series.PixelSpacing1"].astype(float)
        * frame["dicom_series.SliceThickness"].astype(float)
        / 1000.0
    )
    for volume, area in VOLUME_AREAS.items():
        frame[volume] = frame[area].fillna(0).astype(float) * factor
    aggregation: dict[str, str] = {volume: "sum" for volume in VOLUME_AREAS}
    aggregation.update(
        {
            "SkullFracture": "max",
            "MidlineShiftMM": "max",
            "triage_class": "first",
        }
    )
    series = frame.groupby("dicom_series.id", sort=False).agg(aggregation).reset_index()
    series = series.rename(
        columns={"dicom_series.id": "study_id", "MidlineShiftMM": "MLS_mm"}
    )
    series["study_id"] = series["study_id"].astype(str)
    series["truth"] = series["SkullFracture"].astype(int)
    series["metadata_triage_class"] = series["triage_class"].astype(int)
    true_triage: list[int] = []
    no_fracture_triage: list[int] = []
    for row in series.to_dict(orient="records"):
        base = {key: float(row[key]) for key in VOLUME_AREAS}
        base["MLS_mm"] = float(row["MLS_mm"])
        true_triage.append(
            triage_from_intermediates(
                {**base, "fracture_prob": float(row["truth"])}
            )
        )
        no_fracture_triage.append(
            triage_from_intermediates({**base, "fracture_prob": 0.0})
        )
    series["true_triage"] = true_triage
    series["no_fracture_triage"] = no_fracture_triage
    return series


def _candidate_thresholds(score: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(score, dtype=np.float64))
    return np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique[-1], np.inf)],
        )
    )


def _triage_predictions(frame: pd.DataFrame, binary: np.ndarray) -> np.ndarray:
    result: list[int] = []
    for row, fracture in zip(frame.to_dict(orient="records"), binary, strict=True):
        values = {key: float(row[key]) for key in VOLUME_AREAS}
        values.update(
            {"MLS_mm": float(row["MLS_mm"]), "fracture_prob": float(fracture)}
        )
        result.append(triage_from_intermediates(values))
    return np.asarray(result, dtype=np.int64)


def _metrics(frame: pd.DataFrame, score: np.ndarray, threshold: float) -> dict[str, float]:
    fracture = np.asarray(score, dtype=np.float64) >= threshold
    triage = _triage_predictions(frame, fracture)
    return {
        "fracture_f1": float(f1_score(frame["truth"], fracture, zero_division=0)),
        "triage_macro_f1": float(
            f1_score(
                frame["true_triage"],
                triage,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


def _select_threshold(
    frame: pd.DataFrame, score_column: str, objective: str
) -> tuple[float, dict[str, float]]:
    if objective not in {"fracture_f1", "triage_macro_f1"}:
        raise ValueError(f"Unsupported objective: {objective}")
    score = frame[score_column].to_numpy(dtype=np.float64)
    best: tuple[tuple[float, float, float], float, dict[str, float]] | None = None
    for threshold in _candidate_thresholds(score):
        metrics = _metrics(frame, score, float(threshold))
        secondary = (
            metrics["triage_macro_f1"]
            if objective == "fracture_f1"
            else metrics["fracture_f1"]
        )
        key = (metrics[objective], secondary, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise RuntimeError("No threshold candidate was evaluated")
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    series = _series_intermediates(metadata)
    predictions = pd.read_csv(args.predictions, dtype={"study_id": str})
    required = {"study_id", "truth", "outer_fold", args.score_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    frame = series.merge(
        predictions[list(required)], on=["study_id", "truth"], validate="one_to_one"
    )
    if len(frame) != len(predictions) or frame["study_id"].duplicated().any():
        raise RuntimeError("Metadata and prediction identities do not match")
    if not np.isfinite(frame[args.score_column].to_numpy(dtype=float)).all():
        raise ValueError("Prediction scores must be finite")

    rows: list[pd.DataFrame] = []
    selections: list[dict[str, float | int | dict[str, float]]] = []
    for fold in sorted(frame["outer_fold"].unique()):
        development = frame.loc[frame["outer_fold"].ne(fold)]
        heldout = frame.loc[frame["outer_fold"].eq(fold)].copy()
        fracture_threshold, fracture_development = _select_threshold(
            development, args.score_column, "fracture_f1"
        )
        triage_threshold, triage_development = _select_threshold(
            development, args.score_column, "triage_macro_f1"
        )
        for prefix, threshold in (
            ("fracture_objective", fracture_threshold),
            ("triage_objective", triage_threshold),
        ):
            heldout[f"{prefix}_threshold"] = threshold
            heldout[f"{prefix}_binary"] = (
                heldout[args.score_column].to_numpy(float) >= threshold
            ).astype(int)
            heldout[f"{prefix}_triage"] = _triage_predictions(
                heldout, heldout[f"{prefix}_binary"].to_numpy(dtype=int)
            )
        rows.append(heldout)
        selections.append(
            {
                "held_out_fold": int(fold),
                "n_development": len(development),
                "fracture_objective_threshold": fracture_threshold,
                "fracture_objective_development": fracture_development,
                "triage_objective_threshold": triage_threshold,
                "triage_objective_development": triage_development,
            }
        )

    result = pd.concat(rows, ignore_index=True).sort_values("study_id")
    truth_fracture = result["truth"].to_numpy(dtype=int)
    truth_triage = result["true_triage"].to_numpy(dtype=int)

    def summarize(prefix: str) -> dict[str, float]:
        return {
            "fracture_f1": float(
                f1_score(
                    truth_fracture,
                    result[f"{prefix}_binary"],
                    zero_division=0,
                )
            ),
            "triage_macro_f1": float(
                f1_score(
                    truth_triage,
                    result[f"{prefix}_triage"],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    final_fracture_threshold, final_fracture_metrics = _select_threshold(
        result, args.score_column, "fracture_f1"
    )
    final_triage_threshold, final_triage_metrics = _select_threshold(
        result, args.score_column, "triage_macro_f1"
    )
    metadata_triage_agreement = float(
        np.mean(result["metadata_triage_class"] == result["true_triage"])
    )
    no_fracture_macro_f1 = float(
        f1_score(
            truth_triage,
            result["no_fracture_triage"],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    payload = {
        "protocol": "leave_one_outer_fold_out_oracle_ich_mls_fracture_threshold",
        "score_column": args.score_column,
        "n_studies": len(result),
        "n_positive_fractures": int(truth_fracture.sum()),
        "official_triage_distribution": {
            str(label): int(np.sum(truth_triage == label)) for label in (0, 1, 2)
        },
        "metadata_triage_agreement_with_official_derived": metadata_triage_agreement,
        "no_fracture_oracle_context_triage_macro_f1": no_fracture_macro_f1,
        "selections": selections,
        "fracture_objective_crossfit": summarize("fracture_objective"),
        "triage_objective_crossfit": summarize("triage_objective"),
        "deployment": {
            "fracture_objective_threshold_all_oof": final_fracture_threshold,
            "fracture_objective_apparent": final_fracture_metrics,
            "triage_objective_threshold_all_oof": final_triage_threshold,
            "triage_objective_apparent": final_triage_metrics,
        },
        "interpretation": (
            "Oracle-context analysis isolates fracture decisions by using true ICH "
            "volumes and MLS; it is not an end-to-end leaderboard estimate."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "private_predictions.csv", index=False)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (args.output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
