"""Select an MLS checkpoint under an already frozen pooling profile.

This script performs no model inference. It validates a completed CUDA audit,
extracts exactly one frozen-profile row per audited checkpoint, applies the
pre-registered promotion gates, and writes a durable decision artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_PROFILE = {
    "family": "severity_window",
    "size": 3,
    "component_ratio": 0.0,
    "selector_gate": 0.5,
    "min_active_slices": 3,
    "quantile": 0.75,
    "probability_weighted": True,
    "heatmap_guard_ratio": 0.0,
}

NAMED_CHECKPOINTS = {
    "best_objective": "mls_multitask_best.pth",
    "best_mae": "mls_multitask_best_mae.pth",
    "best_selector_auc": "mls_multitask_best_selector_auc.pth",
    "best_peak_auc": "mls_multitask_best_peak_auc.pth",
    "best_study_boundary": "mls_multitask_best_study_boundary.pth",
    "best_study": "mls_multitask_best_study.pth",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-summary", type=Path, required=True)
    parser.add_argument("--audit-status", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epoch-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-studies", type=int, required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--mae-limit", type=float, required=True)
    parser.add_argument("--boundary-floor", type=float, required=True)
    parser.add_argument("--objective-limit", type=float, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _candidate_epoch(label: str, history: pd.DataFrame) -> int:
    snapshot = re.fullmatch(r"epoch(\d{3})", label)
    if snapshot:
        return int(snapshot.group(1))
    selectors = {
        "best_objective": ("selection_objective", True),
        "best_mae": ("mls_mae_mm", True),
        "best_selector_auc": ("selector_auc", False),
        "best_peak_auc": ("selector_peak_auc", False),
        "best_study_boundary": ("study_boundary_f1", False),
        "best_study": ("study_mls_mae_mm", True),
    }
    metric, ascending = selectors[label]
    ordered = history.sort_values([metric, "epoch"], ascending=[ascending, True])
    return int(ordered.iloc[0]["epoch"])


def _checkpoint_path(label: str, checkpoint_dir: Path) -> Path:
    if label in NAMED_CHECKPOINTS:
        filename = NAMED_CHECKPOINTS[label]
    else:
        snapshot = re.fullmatch(r"epoch(\d{3})", label)
        if not snapshot:
            raise ValueError(f"Unknown candidate label: {label}")
        filename = f"mls_multitask_epoch_{snapshot.group(1)}.pth"
    path = checkpoint_dir / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def main() -> int:
    args = _parse_args()
    audit = _read_json(args.audit_status)
    if audit.get("state") != "completed":
        raise RuntimeError("CUDA audit is not completed")
    if audit.get("compute_policy") != "cuda_only_no_cpu_fallback":
        raise RuntimeError("Unexpected audit compute policy")
    if int(audit.get("fold", -1)) != args.fold:
        raise RuntimeError("Audit fold mismatch")
    if int(audit.get("expected_studies", -1)) != args.expected_studies:
        raise RuntimeError("Audit study-count mismatch")

    audited = audit.get("candidates", {})
    if not audited:
        raise RuntimeError("Audit contains no candidates")
    for label, state in audited.items():
        if state.get("state") != "completed" or int(state.get("exit_code", -1)) != 0:
            raise RuntimeError(f"Incomplete audited candidate: {label}")
        metrics = _read_json(Path(state["metrics"]))
        if int(metrics.get("n_studies", -1)) != args.expected_studies:
            raise RuntimeError(f"Incomplete study coverage: {label}")
        if int(metrics.get("failures", -1)) != 0:
            raise RuntimeError(f"Audit failures found: {label}")

    grid = pd.read_csv(args.grid)
    grid_candidates = set(grid["candidate"].astype(str))
    if grid_candidates != set(audited):
        raise RuntimeError("Grid candidates differ from audited candidates")

    mask = pd.Series(True, index=grid.index)
    for key, expected in LOCKED_PROFILE.items():
        if isinstance(expected, float):
            mask &= np.isclose(grid[key].astype(float), expected)
        else:
            mask &= grid[key] == expected
    locked = grid.loc[mask].copy()
    counts = locked.groupby("candidate").size().to_dict()
    if counts != {label: 1 for label in audited}:
        raise RuntimeError(f"Frozen profile is not unique per candidate: {counts}")

    history_rows = [
        json.loads(line)
        for line in args.epoch_history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history = pd.DataFrame(history_rows)
    if len(history) != 23 or set(history["epoch"].astype(int)) != set(range(1, 24)):
        raise RuntimeError("Epoch history is not complete 1..23")

    candidate_rows: list[dict] = []
    for _, row in locked.sort_values("candidate").iterrows():
        label = str(row["candidate"])
        checkpoint = _checkpoint_path(label, args.checkpoint_dir)
        mae_pass = float(row["mae_mm"]) <= args.mae_limit
        boundary_pass = float(row["boundary_f1"]) >= args.boundary_floor
        objective_pass = float(row["selection_objective"]) <= args.objective_limit
        candidate_rows.append({
            **{key: _python_value(row[key]) for key in row.index},
            "epoch": _candidate_epoch(label, history),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "gate_results": {
                "mae_pass": bool(mae_pass),
                "boundary_f1_pass": bool(boundary_pass),
                "selection_objective_pass": bool(objective_pass),
                "all_pass": bool(mae_pass and boundary_pass and objective_pass),
            },
        })

    eligible = [row for row in candidate_rows if row["gate_results"]["all_pass"]]
    eligible.sort(key=lambda row: (
        float(row["selection_objective"]),
        float(row["mae_mm"]),
        float(row["rmse_mm"]),
        -float(row["boundary_f1"]),
        str(row["candidate"]),
    ))
    selected = eligible[0] if eligible else None
    grid_summary = _read_json(args.grid_summary)

    payload = {
        "schema_version": 1,
        "run_name": args.run_name,
        "mlflow_run_id": args.mlflow_run_id,
        "fold": args.fold,
        "audit": {
            "compute_policy": audit["compute_policy"],
            "candidate_checkpoints": len(audited),
            "studies_per_candidate": args.expected_studies,
            "total_candidate_studies": len(audited) * args.expected_studies,
            "failures": 0,
            "state": audit["state"],
        },
        "locked_production_profile": LOCKED_PROFILE,
        "historical_reference": {
            "run": args.reference_run,
            "checkpoint": args.reference_checkpoint,
            "mae_mm": args.mae_limit,
            "boundary_f1": args.boundary_floor,
            "selection_objective": args.objective_limit,
        },
        "required_gates": {
            "mae_mm_lte": args.mae_limit,
            "boundary_f1_gte": args.boundary_floor,
            "selection_objective_lte": args.objective_limit,
        },
        "locked_profile_candidates": candidate_rows,
        "eligible_candidates": [row["candidate"] for row in eligible],
        "selected_candidate": selected,
        "same_fold_diagnostic_only": grid_summary.get("global_best_balanced"),
        "decision": (
            f"Promote {selected['candidate']} after package/runtime validation."
            if selected is not None
            else "Retain the historical fold-1 reference; no candidate passed all gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
