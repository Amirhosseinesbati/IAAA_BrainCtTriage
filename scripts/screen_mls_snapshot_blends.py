"""Lightweight screening of MLS snapshot blends from saved CUDA predictions.

No model is loaded and no image inference is performed. The script blends a
small, predeclared set of aligned per-slice outputs and evaluates four locked
pooling profiles. It is intentionally a screening gate before any CUDA weight
averaging or additional training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from search_mls_crossfold_pooling import PoolingProfile, _decode, _metrics, _predict


EPOCHS = (13, 15, 17, 21)
BLENDS = {
    "single_e15": {"epochs": (15,), "weights": (1.0,), "mode": "mean"},
    "mean_e15_e17": {"epochs": (15, 17), "weights": (0.5, 0.5), "mode": "mean"},
    "w75_e15_w25_e17": {
        "epochs": (15, 17), "weights": (0.75, 0.25), "mode": "mean"
    },
    "mean_e13_e15_e17": {
        "epochs": (13, 15, 17), "weights": (1 / 3, 1 / 3, 1 / 3), "mode": "mean"
    },
    "w25_e13_w50_e15_w25_e17": {
        "epochs": (13, 15, 17), "weights": (0.25, 0.5, 0.25), "mode": "mean"
    },
    "median_e13_e15_e17": {
        "epochs": (13, 15, 17), "weights": (1 / 3, 1 / 3, 1 / 3), "mode": "median"
    },
    "mean_e15_e21": {"epochs": (15, 21), "weights": (0.5, 0.5), "mode": "mean"},
    "w75_e15_w25_e21": {
        "epochs": (15, 21), "weights": (0.75, 0.25), "mode": "mean"
    },
}
PROFILES = {
    "production": PoolingProfile(
        "severity_window", 3, 0.0, 0.5, 3, 0.75, True, 0.0
    ),
    "production_guard": PoolingProfile(
        "severity_window", 3, 0.0, 0.5, 3, 0.75, True, 0.5
    ),
    "frozen_fold01_boundary": PoolingProfile(
        "severity_window", 3, 0.0, 0.5, 3, 0.75, False, 0.5
    ),
    "boundary_diagnostic": PoolingProfile(
        "relative_component", 0, 0.3, 0.5, 3, 0.9, False, 0.0
    ),
}
PREDICTION_KEYS = ("selector_probability", "mls_mm", "heatmap_peak")


def _prediction_path(root: Path, epoch: int) -> Path:
    path = root / f"epoch{epoch:03d}" / "study_slice_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"study_id": str, "patient_id": str})
    required = {"study_id", "gt_MLS_mm", "slice_predictions_json", "error"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    errors = frame["error"].fillna("").astype(str).str.strip()
    if (errors != "").any():
        raise RuntimeError(f"Saved evaluation contains errors: {path}")
    if frame["study_id"].duplicated().any():
        raise ValueError(f"Duplicate study IDs: {path}")
    return frame.sort_values("study_id").reset_index(drop=True)


def _validate(frames: dict[int, pd.DataFrame], fold: str) -> None:
    reference = frames[15]
    for epoch, frame in frames.items():
        if frame["study_id"].tolist() != reference["study_id"].tolist():
            raise ValueError(f"Study IDs differ in {fold}/epoch{epoch}")
        if not np.allclose(
            frame["gt_MLS_mm"].to_numpy(float),
            reference["gt_MLS_mm"].to_numpy(float),
            atol=1e-8,
        ):
            raise ValueError(f"Ground truth differs in {fold}/epoch{epoch}")


def _blend_items(payloads: list[list[dict]], weights: np.ndarray, mode: str) -> list[dict]:
    indices = [int(item["index"]) for item in payloads[0]]
    for payload in payloads[1:]:
        if [int(item["index"]) for item in payload] != indices:
            raise ValueError("Slice indices differ between aligned snapshots")

    blended = []
    for item_index, slice_index in enumerate(indices):
        item: dict[str, float | int] = {"index": slice_index}
        for key in PREDICTION_KEYS:
            values = np.asarray([
                float(payload[item_index][key]) for payload in payloads
            ])
            item[key] = float(
                np.median(values) if mode == "median" else np.average(values, weights=weights)
            )
        blended.append(item)
    return blended


def _selection_payload(row: pd.Series, held_out: str) -> dict:
    return {
        "candidate": str(row["candidate"]),
        "profile": str(row["profile"]),
        "held_out_mae_mm": float(row[f"{held_out}_mae_mm"]),
        "held_out_boundary_f1": float(row[f"{held_out}_boundary_f1"]),
        "held_out_objective": float(row[f"{held_out}_selection_objective"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0-root", type=Path, required=True)
    parser.add_argument("--fold1-root", type=Path, required=True)
    parser.add_argument("--fold2-root", type=Path, required=True)
    parser.add_argument("--swa-fold0", type=Path)
    parser.add_argument("--swa-fold1", type=Path)
    parser.add_argument("--swa-fold2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roots = {
        "fold0": args.fold0_root,
        "fold1": args.fold1_root,
        "fold2": args.fold2_root,
    }
    frames = {
        fold: {epoch: _load(_prediction_path(root, epoch)) for epoch in EPOCHS}
        for fold, root in roots.items()
    }
    for fold, fold_frames in frames.items():
        _validate(fold_frames, fold)

    swa_paths = {
        "fold0": args.swa_fold0,
        "fold1": args.swa_fold1,
        "fold2": args.swa_fold2,
    }
    supplied_swa = [path is not None for path in swa_paths.values()]
    if any(supplied_swa) and not all(supplied_swa):
        raise ValueError("Supply all three SWA fold CSVs or none")
    swa_frames = None
    if all(supplied_swa):
        swa_frames = {fold: _load(path) for fold, path in swa_paths.items()}
        for fold, swa_frame in swa_frames.items():
            reference = frames[fold][15]
            if swa_frame["study_id"].tolist() != reference["study_id"].tolist():
                raise ValueError(f"SWA study IDs differ in {fold}")
            if not np.allclose(
                swa_frame["gt_MLS_mm"].to_numpy(float),
                reference["gt_MLS_mm"].to_numpy(float),
                atol=1e-8,
            ):
                raise ValueError(f"SWA ground truth differs in {fold}")

    decoded = {
        fold: {epoch: _decode(frame) for epoch, frame in fold_frames.items()}
        for fold, fold_frames in frames.items()
    }
    truths = {
        fold: fold_frames[15]["gt_MLS_mm"].to_numpy(float)
        for fold, fold_frames in frames.items()
    }

    blended: dict[str, dict[str, list[list[dict]]]] = {}
    for candidate, spec in BLENDS.items():
        epochs = spec["epochs"]
        weights = np.asarray(spec["weights"], dtype=float)
        blended[candidate] = {}
        for fold in roots:
            n_studies = len(decoded[fold][15])
            blended[candidate][fold] = [
                _blend_items(
                    [decoded[fold][epoch][index] for epoch in epochs],
                    weights,
                    str(spec["mode"]),
                )
                for index in range(n_studies)
            ]
    if swa_frames is not None:
        blended["weight_average_e13_e15_e17"] = {
            fold: _decode(frame) for fold, frame in swa_frames.items()
        }

    rows = []
    for candidate, fold_payloads in blended.items():
        for profile_name, profile in PROFILES.items():
            row = {
                "candidate": candidate,
                "profile": profile_name,
                **{f"profile_{key}": value for key, value in asdict(profile).items()},
            }
            for fold, items_by_study in fold_payloads.items():
                predictions = np.asarray([
                    _predict(items, profile) for items in items_by_study
                ])
                metrics = _metrics(truths[fold], predictions)
                boundary = float(np.mean([metrics["f1_3mm"], metrics["f1_5mm"]]))
                metrics["boundary_f1"] = boundary
                metrics["selection_objective"] = metrics["mae_mm"] + 2 * (1 - boundary)
                for key, value in metrics.items():
                    row[f"{fold}_{key}"] = value
            row["mean_mae_mm"] = float(np.mean([
                row[f"{fold}_mae_mm"] for fold in roots
            ]))
            row["worst_mae_mm"] = float(np.max([
                row[f"{fold}_mae_mm"] for fold in roots
            ]))
            row["mean_boundary_f1"] = float(np.mean([
                row[f"{fold}_boundary_f1"] for fold in roots
            ]))
            row["mean_selection_objective"] = float(np.mean([
                row[f"{fold}_selection_objective"] for fold in roots
            ]))
            rows.append(row)

    results = pd.DataFrame(rows)
    balanced = results.sort_values(
        ["mean_selection_objective", "mean_mae_mm", "worst_mae_mm"]
    )
    production = results.loc[results["profile"] == "production"].sort_values(
        ["mean_selection_objective", "mean_mae_mm", "worst_mae_mm"]
    )

    loo = {}
    for held_out in roots:
        selection_folds = [fold for fold in roots if fold != held_out]
        train_objective = results[
            [f"{fold}_selection_objective" for fold in selection_folds]
        ].mean(axis=1)
        train_mae = results[
            [f"{fold}_mae_mm" for fold in selection_folds]
        ].mean(axis=1)
        selected_index = results.assign(
            _train_objective=train_objective, _train_mae=train_mae
        ).sort_values(["_train_objective", "_train_mae"]).index[0]
        selected = results.loc[selected_index]
        loo[held_out] = {
            "selected_using": selection_folds,
            "selection_mean_objective": float(train_objective.loc[selected_index]),
            **_selection_payload(selected, held_out),
        }

    loo_entries = list(loo.values())
    payload = {
        "warning": (
            "Screening only: no model/image inference was run. Blend choices were "
            "defined after prior fold analysis and are not a fresh unbiased estimate."
        ),
        "blend_definitions": BLENDS,
        "profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
        "best_all_fold_balanced": balanced.iloc[0].to_dict(),
        "fixed_production_profile_ranking": production.to_dict(orient="records"),
        "limited_nested_loo": loo,
        "limited_nested_aggregate": {
            "mean_held_out_mae_mm": float(np.mean([
                entry["held_out_mae_mm"] for entry in loo_entries
            ])),
            "worst_held_out_mae_mm": float(np.max([
                entry["held_out_mae_mm"] for entry in loo_entries
            ])),
            "mean_held_out_boundary_f1": float(np.mean([
                entry["held_out_boundary_f1"] for entry in loo_entries
            ])),
            "mean_held_out_objective": float(np.mean([
                entry["held_out_objective"] for entry in loo_entries
            ])),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "snapshot_blend_screen_grid.csv", index=False)
    (args.output_dir / "snapshot_blend_screen_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["limited_nested_aggregate"], indent=2))
    print(json.dumps(payload["best_all_fold_balanced"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
