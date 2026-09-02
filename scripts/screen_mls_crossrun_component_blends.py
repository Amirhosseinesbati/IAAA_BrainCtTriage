"""Screen cross-run MLS component blends from saved CUDA predictions.

This script performs no model or image inference. It consumes aligned
``study_slice_predictions.csv`` files that were already produced by CUDA
full-study audits. The primary question is whether a challenger regression
head adds useful signal when the trusted baseline supplies progressively more
of the selector/ranking path.

The production pooling profile is frozen. A guarded profile is reported only
as a diagnostic, and same-fold results cannot authorize model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from search_mls_crossfold_pooling import PoolingProfile, _metrics, _predict


PREDICTION_KEYS = (
    "selector_probability",
    "peak_probability",
    "mls_mm",
    "heatmap_peak",
)
COMPONENT_MODES = {
    "full": frozenset(PREDICTION_KEYS),
    "baseline_selector": frozenset(("peak_probability", "mls_mm", "heatmap_peak")),
    "baseline_selector_peak": frozenset(("mls_mm", "heatmap_peak")),
    "regression_only": frozenset(("mls_mm",)),
}
MODE_COMPLEXITY = {
    "baseline": 0,
    **{mode: len(keys) for mode, keys in COMPONENT_MODES.items()},
}
PROFILES = {
    "locked_production": PoolingProfile(
        "severity_window", 3, 0.0, 0.5, 3, 0.75, True, 0.0
    ),
    "locked_guard_05_diagnostic": PoolingProfile(
        "severity_window", 3, 0.0, 0.5, 3, 0.75, True, 0.5
    ),
}
GATE_ABS_TOLERANCE = 1e-9


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("challenger must be LABEL=CSV_PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("challenger label cannot be empty")
    return label, Path(raw_path)


def _parse_alpha(value: str) -> float:
    try:
        alpha = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alpha must be numeric") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be finite and in (0, 1]")
    return alpha


def _clean_error(value: object) -> str:
    text = str(value).strip()
    return "" if text in {"", "nan", "None"} else text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, expected_studies: int) -> tuple[pd.DataFrame, list[list[dict]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"study_id": str, "patient_id": str})
    required = {
        "study_id",
        "patient_id",
        "triage_class",
        "gt_MLS_mm",
        "slice_predictions_json",
        "runtime_s",
        "error",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if len(frame) != expected_studies:
        raise ValueError(
            f"{path} has {len(frame)} studies; expected {expected_studies}"
        )
    errors = frame["error"].map(_clean_error)
    if (errors != "").any():
        bad = frame.loc[errors != "", ["study_id", "error"]].head(3)
        raise RuntimeError(f"Evaluation errors in {path}: {bad.to_dict('records')}")
    if frame["study_id"].duplicated().any():
        raise ValueError(f"Duplicate study_id values in {path}")
    frame = frame.sort_values("study_id").reset_index(drop=True)

    decoded: list[list[dict]] = []
    for study_id, raw_payload in zip(frame["study_id"], frame["slice_predictions_json"]):
        payload = json.loads(raw_payload)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Empty/non-list slice payload for study {study_id}")
        indices: list[int] = []
        for item in payload:
            missing_keys = {
                "index", "selector_probability", "mls_mm", "heatmap_peak"
            } - set(item)
            if missing_keys:
                raise ValueError(
                    f"Study {study_id} slice payload missing {sorted(missing_keys)}"
                )
            indices.append(int(item["index"]))
            for key in ("selector_probability", "mls_mm", "heatmap_peak"):
                if not np.isfinite(float(item[key])):
                    raise ValueError(f"Non-finite {key} in study {study_id}")
            if "peak_probability" in item and not np.isfinite(
                float(item["peak_probability"])
            ):
                raise ValueError(f"Non-finite peak_probability in study {study_id}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate slice indices in study {study_id}")
        decoded.append(payload)
    return frame, decoded


def _validate_alignment(
    baseline_frame: pd.DataFrame,
    baseline_decoded: list[list[dict]],
    challenger_frame: pd.DataFrame,
    challenger_decoded: list[list[dict]],
    label: str,
) -> None:
    for column in ("study_id", "patient_id", "triage_class"):
        if challenger_frame[column].tolist() != baseline_frame[column].tolist():
            raise ValueError(f"{column} differs for challenger {label}")
    if not np.allclose(
        challenger_frame["gt_MLS_mm"].to_numpy(float),
        baseline_frame["gt_MLS_mm"].to_numpy(float),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(f"Ground truth differs for challenger {label}")
    for row_index, (baseline_items, challenger_items) in enumerate(
        zip(baseline_decoded, challenger_decoded)
    ):
        baseline_indices = [int(item["index"]) for item in baseline_items]
        challenger_indices = [int(item["index"]) for item in challenger_items]
        if challenger_indices != baseline_indices:
            study_id = baseline_frame.iloc[row_index]["study_id"]
            raise ValueError(f"Slice indices differ for {label}, study {study_id}")


def _value(item: dict, key: str) -> float:
    if key == "peak_probability":
        return float(item.get(key, item["selector_probability"]))
    return float(item[key])


def _blend_study(
    baseline_items: list[dict],
    challenger_items: list[dict],
    alpha: float,
    blended_keys: frozenset[str],
) -> list[dict]:
    output: list[dict] = []
    for baseline_item, challenger_item in zip(baseline_items, challenger_items):
        result: dict[str, float | int] = {"index": int(baseline_item["index"])}
        for key in PREDICTION_KEYS:
            baseline_value = _value(baseline_item, key)
            if key in blended_keys:
                challenger_value = _value(challenger_item, key)
                result[key] = float(
                    (1.0 - alpha) * baseline_value + alpha * challenger_value
                )
            else:
                result[key] = baseline_value
        output.append(result)
    return output


def _candidate_payloads(
    baseline_decoded: list[list[dict]],
    challenger_decoded: list[list[dict]],
    alpha: float,
    mode: str,
) -> list[list[dict]]:
    blended_keys = COMPONENT_MODES[mode]
    return [
        _blend_study(baseline_items, challenger_items, alpha, blended_keys)
        for baseline_items, challenger_items in zip(
            baseline_decoded, challenger_decoded
        )
    ]


def _metrics_row(
    candidate: str,
    challenger: str,
    mode: str,
    alpha: float,
    profile_name: str,
    profile: PoolingProfile,
    truth: np.ndarray,
    payloads: list[list[dict]],
    mae_limit: float,
    boundary_floor: float,
    objective_limit: float,
) -> dict:
    prediction = np.asarray([_predict(items, profile) for items in payloads])
    metrics = _metrics(truth, prediction)
    boundary_f1 = float(np.mean([metrics["f1_3mm"], metrics["f1_5mm"]]))
    objective = float(metrics["mae_mm"] + 2.0 * (1.0 - boundary_f1))
    # The incumbent must not fail its own frozen gate because the same metric
    # was serialized and parsed through a different floating-point path.
    # 1e-9 mm/F1 is many orders below any clinically or competitively relevant
    # difference and only absorbs machine-rounding noise.
    mae_pass = metrics["mae_mm"] <= mae_limit + GATE_ABS_TOLERANCE
    boundary_pass = boundary_f1 + GATE_ABS_TOLERANCE >= boundary_floor
    objective_pass = objective <= objective_limit + GATE_ABS_TOLERANCE
    return {
        "candidate": candidate,
        "challenger": challenger,
        "mode": mode,
        "blended_component_count": MODE_COMPLEXITY[mode],
        "challenger_alpha": alpha,
        "profile": profile_name,
        **{f"profile_{key}": value for key, value in asdict(profile).items()},
        **metrics,
        "boundary_f1": boundary_f1,
        "selection_objective": objective,
        "mae_pass": bool(mae_pass),
        "boundary_f1_pass": bool(boundary_pass),
        "selection_objective_pass": bool(objective_pass),
        "all_release_gates_pass": bool(
            mae_pass and boundary_pass and objective_pass
        ),
    }


def _row_payload(row: pd.Series) -> dict:
    payload: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        payload[key] = value
    return payload


def _assert_close(
    name: str,
    actual: float,
    expected: float | None,
    tolerance: float,
) -> None:
    if expected is None:
        return
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise RuntimeError(
            f"Baseline parity failed for {name}: actual={actual}, "
            f"expected={expected}, tolerance={tolerance}"
        )


def _write_prediction_csv(
    path: Path,
    reference: pd.DataFrame,
    payloads: list[list[dict]],
    runtime_s: np.ndarray,
) -> None:
    rows = []
    for index, (_, row) in enumerate(reference.iterrows()):
        rows.append({
            "study_id": str(row["study_id"]),
            "patient_id": str(row["patient_id"]),
            "triage_class": int(row["triage_class"]),
            "gt_MLS_mm": float(row["gt_MLS_mm"]),
            "slice_predictions_json": json.dumps(payloads[index]),
            "runtime_s": float(runtime_s[index]),
            "error": "",
        })
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _format_candidate(row: pd.Series) -> str:
    return (
        f"{row['candidate']}: MAE={row['mae_mm']:.6f} mm, "
        f"Boundary-F1={row['boundary_f1']:.6f}, "
        f"objective={row['selection_objective']:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--challenger",
        action="append",
        type=_named_path,
        required=True,
        help="Repeat LABEL=study_slice_predictions.csv.",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        type=_parse_alpha,
        default=[],
        help="Challenger contribution; repeat as needed. Defaults to .1,.25,.5,.75,1.",
    )
    parser.add_argument("--expected-studies", type=int, default=67)
    parser.add_argument("--mae-limit", type=float, required=True)
    parser.add_argument("--boundary-floor", type=float, required=True)
    parser.add_argument("--objective-limit", type=float, required=True)
    parser.add_argument("--expected-baseline-mae", type=float)
    parser.add_argument("--expected-baseline-boundary-f1", type=float)
    parser.add_argument("--expected-baseline-objective", type=float)
    parser.add_argument(
        "--baseline-parity-tolerance",
        type=float,
        default=1e-9,
        help=(
            "Absolute tolerance only for cross-runtime baseline reproducibility. "
            "This never changes the frozen release gates."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if (
        not np.isfinite(args.baseline_parity_tolerance)
        or args.baseline_parity_tolerance < 0
    ):
        raise ValueError("baseline parity tolerance must be finite and nonnegative")

    challenger_labels = [label for label, _ in args.challenger]
    if len(challenger_labels) != len(set(challenger_labels)):
        raise ValueError("Challenger labels must be unique")
    alphas = sorted(set(args.alpha or (0.1, 0.25, 0.5, 0.75, 1.0)))

    baseline_frame, baseline_decoded = _load(args.baseline, args.expected_studies)
    challengers: dict[str, tuple[pd.DataFrame, list[list[dict]], Path]] = {}
    for label, path in args.challenger:
        frame, decoded = _load(path, args.expected_studies)
        _validate_alignment(
            baseline_frame, baseline_decoded, frame, decoded, label
        )
        challengers[label] = (frame, decoded, path)

    truth = baseline_frame["gt_MLS_mm"].to_numpy(float)
    rows: list[dict] = []
    specifications: dict[str, dict] = {
        "baseline_exp09": {
            "challenger": "none", "mode": "baseline", "alpha": 0.0
        }
    }
    for profile_name, profile in PROFILES.items():
        rows.append(_metrics_row(
            "baseline_exp09",
            "none",
            "baseline",
            0.0,
            profile_name,
            profile,
            truth,
            baseline_decoded,
            args.mae_limit,
            args.boundary_floor,
            args.objective_limit,
        ))

    for label, (_, challenger_decoded, _) in challengers.items():
        for mode in COMPONENT_MODES:
            for alpha in alphas:
                alpha_text = str(alpha).replace(".", "p")
                candidate = f"{label}__{mode}__a{alpha_text}"
                specifications[candidate] = {
                    "challenger": label, "mode": mode, "alpha": alpha
                }
                payloads = _candidate_payloads(
                    baseline_decoded, challenger_decoded, alpha, mode
                )
                for profile_name, profile in PROFILES.items():
                    rows.append(_metrics_row(
                        candidate,
                        label,
                        mode,
                        alpha,
                        profile_name,
                        profile,
                        truth,
                        payloads,
                        args.mae_limit,
                        args.boundary_floor,
                        args.objective_limit,
                    ))

    results = pd.DataFrame(rows)
    locked = results.loc[results["profile"] == "locked_production"].copy()
    baseline_locked = locked.loc[locked["candidate"] == "baseline_exp09"].iloc[0]
    _assert_close(
        "mae_mm",
        baseline_locked["mae_mm"],
        args.expected_baseline_mae,
        args.baseline_parity_tolerance,
    )
    _assert_close(
        "boundary_f1",
        baseline_locked["boundary_f1"],
        args.expected_baseline_boundary_f1,
        args.baseline_parity_tolerance,
    )
    _assert_close(
        "selection_objective",
        baseline_locked["selection_objective"],
        args.expected_baseline_objective,
        args.baseline_parity_tolerance,
    )

    nonbaseline = locked.loc[locked["candidate"] != "baseline_exp09"].copy()
    nonbaseline = nonbaseline.sort_values(
        [
            "selection_objective",
            "mae_mm",
            "rmse_mm",
            "blended_component_count",
            "candidate",
        ]
    )
    eligible = nonbaseline.loc[nonbaseline["all_release_gates_pass"]]
    best_nonbaseline = nonbaseline.iloc[0]
    screen_selection = eligible.iloc[0] if len(eligible) else best_nonbaseline
    selection_spec = specifications[str(screen_selection["candidate"])]
    selection_frame, selection_decoded, _ = challengers[
        str(selection_spec["challenger"])
    ]
    selection_payloads = _candidate_payloads(
        baseline_decoded,
        selection_decoded,
        float(selection_spec["alpha"]),
        str(selection_spec["mode"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / "crossrun_component_blend_grid.csv"
    results.to_csv(grid_path, index=False)
    prediction_path = args.output_dir / "screen_selected_predictions.csv"
    combined_runtime = (
        baseline_frame["runtime_s"].to_numpy(float)
        + selection_frame["runtime_s"].to_numpy(float)
    )
    _write_prediction_csv(
        prediction_path,
        baseline_frame,
        selection_payloads,
        combined_runtime,
    )

    deltas = {
        "mae_mm": float(screen_selection["mae_mm"] - baseline_locked["mae_mm"]),
        "boundary_f1": float(
            screen_selection["boundary_f1"] - baseline_locked["boundary_f1"]
        ),
        "selection_objective": float(
            screen_selection["selection_objective"]
            - baseline_locked["selection_objective"]
        ),
    }
    improves_baseline_objective = bool(
        deltas["selection_objective"] < -GATE_ABS_TOLERANCE
    )
    if len(eligible):
        decision = (
            "Complementarity signal found under all frozen numerical gates; "
            "cross-fold/leaderboard validation is still required."
        )
    elif improves_baseline_objective:
        decision = (
            "A same-runtime complementarity signal improved the baseline "
            "objective, but no blend passed all frozen release gates."
        )
    else:
        decision = "No cross-run component blend improved the baseline objective."
    payload = {
        "schema_version": 1,
        "compute_policy": "saved_cuda_predictions_cpu_postprocessing_only",
        "warning": (
            "This is a same-fold diagnostic screen, not an unbiased promotion "
            "estimate. No model or image inference was run."
        ),
        "expected_studies": args.expected_studies,
        "inputs": {
            "baseline": {
                "path": str(args.baseline),
                "sha256": _sha256(args.baseline),
            },
            "challengers": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, (_, _, path) in challengers.items()
            },
        },
        "component_modes": {
            mode: sorted(keys) for mode, keys in COMPONENT_MODES.items()
        },
        "tie_break_rule": (
            "objective, MAE, RMSE, then fewer blended components, then label"
        ),
        "alphas": alphas,
        "profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
        "release_gates": {
            "mae_mm_lte": args.mae_limit,
            "boundary_f1_gte": args.boundary_floor,
            "selection_objective_lte": args.objective_limit,
            "absolute_numeric_tolerance": GATE_ABS_TOLERANCE,
        },
        "baseline_reproducibility": {
            "absolute_tolerance": args.baseline_parity_tolerance,
            "expected_mae_mm": args.expected_baseline_mae,
            "expected_boundary_f1": args.expected_baseline_boundary_f1,
            "expected_selection_objective": args.expected_baseline_objective,
            "mae_delta": (
                None
                if args.expected_baseline_mae is None
                else float(baseline_locked["mae_mm"] - args.expected_baseline_mae)
            ),
            "boundary_f1_delta": (
                None
                if args.expected_baseline_boundary_f1 is None
                else float(
                    baseline_locked["boundary_f1"]
                    - args.expected_baseline_boundary_f1
                )
            ),
            "selection_objective_delta": (
                None
                if args.expected_baseline_objective is None
                else float(
                    baseline_locked["selection_objective"]
                    - args.expected_baseline_objective
                )
            ),
        },
        "baseline_locked": _row_payload(baseline_locked),
        "best_nonbaseline_locked": _row_payload(best_nonbaseline),
        "eligible_nonbaseline_candidates": eligible["candidate"].tolist(),
        "screen_selection": _row_payload(screen_selection),
        "screen_selection_delta_vs_baseline": deltas,
        "improves_same_runtime_baseline_objective": improves_baseline_objective,
        "screen_selection_predictions": {
            "path": str(prediction_path),
            "sha256": _sha256(prediction_path),
        },
        "decision": decision,
    }
    summary_path = args.output_dir / "crossrun_component_blend_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Exp09 + Exp18 cross-run component-blend screen",
        "",
        payload["warning"],
        "",
        f"- Baseline: `{_format_candidate(baseline_locked)}`",
        f"- Best nonbaseline: `{_format_candidate(best_nonbaseline)}`",
        f"- Selected diagnostic: `{_format_candidate(screen_selection)}`",
        f"- Eligible nonbaseline candidates: `{len(eligible)}`",
        f"- Improves same-runtime baseline objective: `{improves_baseline_objective}`",
        f"- Decision: {payload['decision']}",
        "",
        "## Selected delta versus Exp09",
        "",
        f"- MAE: `{deltas['mae_mm']:+.6f} mm`",
        f"- Boundary-F1: `{deltas['boundary_f1']:+.6f}`",
        f"- objective: `{deltas['selection_objective']:+.6f}`",
        "",
        "The locked production profile was not tuned in this screen. The guarded",
        "profile is diagnostic only. Promotion requires independent cross-fold and",
        "ultimately leaderboard evidence.",
    ]
    (args.output_dir / "CROSSRUN_COMPONENT_BLEND_SCREEN.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "baseline": _row_payload(baseline_locked),
        "best_nonbaseline": _row_payload(best_nonbaseline),
        "eligible_nonbaseline_count": len(eligible),
        "decision": payload["decision"],
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
