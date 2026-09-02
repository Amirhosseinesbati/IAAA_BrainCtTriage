"""Evaluate one preregistered MLS component transfer without model inference.

Both inputs must be full-study prediction CSVs previously produced by CUDA.
Unlike a screening grid, this evaluator accepts exactly one challenger, one
alpha and one component mode. It is intended for an independent-fold transfer
gate where post-hoc checkpoint, weight and component selection are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from screen_mls_crossrun_component_blends import (
    COMPONENT_MODES,
    GATE_ABS_TOLERANCE,
    PROFILES,
    _assert_close,
    _candidate_payloads,
    _load,
    _metrics_row,
    _validate_alignment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--challenger-label", required=True)
    parser.add_argument(
        "--component-mode", choices=tuple(COMPONENT_MODES), required=True
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--expected-studies", type=int, required=True)
    parser.add_argument("--mae-limit", type=float, required=True)
    parser.add_argument("--boundary-floor", type=float, required=True)
    parser.add_argument("--objective-limit", type=float, required=True)
    parser.add_argument("--expected-baseline-mae", type=float, required=True)
    parser.add_argument(
        "--expected-baseline-boundary-f1", type=float, required=True
    )
    parser.add_argument(
        "--expected-baseline-objective", type=float, required=True
    )
    parser.add_argument("--baseline-parity-tolerance", type=float, default=1e-9)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not np.isfinite(args.alpha) or not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    if (
        not np.isfinite(args.baseline_parity_tolerance)
        or args.baseline_parity_tolerance < 0
    ):
        raise ValueError("baseline parity tolerance must be finite and nonnegative")
    if not args.baseline_label.strip() or not args.challenger_label.strip():
        raise ValueError("baseline and challenger labels must be nonempty")
    if args.baseline_label == args.challenger_label:
        raise ValueError("baseline and challenger labels must differ")

    baseline_frame, baseline_decoded = _load(
        args.baseline, args.expected_studies
    )
    challenger_frame, challenger_decoded = _load(
        args.challenger, args.expected_studies
    )
    _validate_alignment(
        baseline_frame,
        baseline_decoded,
        challenger_frame,
        challenger_decoded,
        args.challenger_label,
    )
    truth = baseline_frame["gt_MLS_mm"].to_numpy(float)
    profile = PROFILES["locked_production"]
    baseline = _metrics_row(
        args.baseline_label,
        "none",
        "baseline",
        0.0,
        "locked_production",
        profile,
        truth,
        baseline_decoded,
        args.mae_limit,
        args.boundary_floor,
        args.objective_limit,
    )
    _assert_close(
        "mae_mm",
        baseline["mae_mm"],
        args.expected_baseline_mae,
        args.baseline_parity_tolerance,
    )
    _assert_close(
        "boundary_f1",
        baseline["boundary_f1"],
        args.expected_baseline_boundary_f1,
        args.baseline_parity_tolerance,
    )
    _assert_close(
        "selection_objective",
        baseline["selection_objective"],
        args.expected_baseline_objective,
        args.baseline_parity_tolerance,
    )

    hybrid_payloads = _candidate_payloads(
        baseline_decoded,
        challenger_decoded,
        args.alpha,
        args.component_mode,
    )
    candidate_name = (
        f"{args.baseline_label}__{args.challenger_label}__"
        f"{args.component_mode}__a{str(args.alpha).replace('.', 'p')}"
    )
    hybrid = _metrics_row(
        candidate_name,
        args.challenger_label,
        args.component_mode,
        args.alpha,
        "locked_production",
        profile,
        truth,
        hybrid_payloads,
        args.mae_limit,
        args.boundary_floor,
        args.objective_limit,
    )
    deltas = {
        key: float(hybrid[key] - baseline[key])
        for key in ("mae_mm", "rmse_mm", "bias_mm", "boundary_f1")
    }
    deltas["selection_objective"] = float(
        hybrid["selection_objective"] - baseline["selection_objective"]
    )
    passed = bool(hybrid["all_release_gates_pass"])
    payload = {
        "schema_version": 1,
        "evaluation_design": "preregistered_frozen_crossfold_transfer",
        "compute_policy": "saved_cuda_predictions_cpu_postprocessing_only",
        "warning": (
            "No model or image inference was run by this evaluator. Inputs must "
            "come from completed full-study CUDA audits."
        ),
        "inputs": {
            "baseline": {
                "label": args.baseline_label,
                "path": str(args.baseline),
                "sha256": _sha256(args.baseline),
            },
            "challenger": {
                "label": args.challenger_label,
                "path": str(args.challenger),
                "sha256": _sha256(args.challenger),
            },
        },
        "expected_studies": args.expected_studies,
        "fixed_recipe": {
            "component_mode": args.component_mode,
            "blended_components": sorted(COMPONENT_MODES[args.component_mode]),
            "challenger_alpha": args.alpha,
            "baseline_alpha": 1.0 - args.alpha,
            "profile": asdict(profile),
        },
        "baseline_reproducibility": {
            "absolute_tolerance": args.baseline_parity_tolerance,
            "expected_mae_mm": args.expected_baseline_mae,
            "expected_boundary_f1": args.expected_baseline_boundary_f1,
            "expected_selection_objective": args.expected_baseline_objective,
        },
        "required_gates": {
            "mae_mm_lte": args.mae_limit,
            "boundary_f1_gte": args.boundary_floor,
            "selection_objective_lte": args.objective_limit,
            "absolute_numeric_tolerance": GATE_ABS_TOLERANCE,
        },
        "baseline": baseline,
        "hybrid": hybrid,
        "hybrid_delta_vs_baseline": deltas,
        "passed": passed,
        "decision": (
            "Frozen independent-fold component transfer passed all gates."
            if passed
            else "Frozen independent-fold component transfer failed."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        args.output_dir / "fixed_component_transfer_summary.json",
        json.dumps(payload, indent=2) + "\n",
    )
    report = [
        "# Frozen MLS component-transfer result",
        "",
        f"- Design: `{payload['evaluation_design']}`",
        f"- Fixed mode: `{args.component_mode}`",
        f"- Challenger alpha: `{args.alpha:.2f}`",
        f"- Studies: `{args.expected_studies}`",
        f"- Baseline MAE: `{baseline['mae_mm']:.9f}`",
        f"- Hybrid MAE: `{hybrid['mae_mm']:.9f}`",
        f"- Baseline Boundary-F1: `{baseline['boundary_f1']:.9f}`",
        f"- Hybrid Boundary-F1: `{hybrid['boundary_f1']:.9f}`",
        f"- Baseline objective: `{baseline['selection_objective']:.9f}`",
        f"- Hybrid objective: `{hybrid['selection_objective']:.9f}`",
        f"- Objective delta: `{deltas['selection_objective']:+.9f}`",
        f"- Decision: {payload['decision']}",
        "",
        payload["warning"],
    ]
    _atomic_text(
        args.output_dir / "FROZEN_COMPONENT_TRANSFER_REPORT.md",
        "\n".join(report) + "\n",
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
