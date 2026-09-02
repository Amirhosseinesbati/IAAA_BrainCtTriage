"""CUDA-only 204-study parity audit for the conservative MLS package.

The exact extracted package runtime performs all neural-network forwards. Raw
slice outputs are compared with independent CUDA audit caches, while the
packaged member aggregations and OOF metrics are compared with the frozen
saved-prediction evaluator. Raw per-study rows stay in the audit directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from screen_mls_crossrun_component_blends import (
    PROFILES,
    _candidate_payloads,
    _load,
    _validate_alignment,
)
from search_mls_crossfold_pooling import _metrics, _predict


SLICE_SELECTOR_TOL = 2e-6
SLICE_PEAK_SELECTOR_TOL = 2e-6
SLICE_HEATMAP_TOL = 2e-8
SLICE_MLS_TOL_MM = 2e-5
MEMBER_TOL_MM = 2e-5
OOF_METRIC_TOL = 1e-6


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _runtime(root: Path) -> ModuleType:
    module_path = root / "mls.py"
    if not module_path.is_file() or not (root / "dicom_io.py").is_file():
        raise FileNotFoundError(f"Incomplete package runtime: {root}")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("conservative_package_mls", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_payload(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    raw = _metrics(truth, prediction)
    boundary = float(np.mean([raw["f1_3mm"], raw["f1_5mm"]]))
    return {
        "mae_mm": float(raw["mae_mm"]),
        "rmse_mm": float(raw["rmse_mm"]),
        "bias_mm": float(raw["bias_mm"]),
        "f1_3mm": float(raw["f1_3mm"]),
        "f1_5mm": float(raw["f1_5mm"]),
        "boundary_f1": boundary,
        "selection_objective": float(raw["mae_mm"] + 2.0 * (1.0 - boundary)),
    }


def _value(item: Any, name: str) -> float:
    if name == "peak_probability":
        if hasattr(item, name):
            return float(getattr(item, name))
        return float(item.get(name, item["selector_probability"]))
    return float(getattr(item, name) if hasattr(item, name) else item[name])


def _slice_parity(actual: list[Any], expected: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(actual) != len(expected):
        raise RuntimeError(
            f"Slice-count mismatch: packaged={len(actual)}, expected={len(expected)}"
        )
    result: dict[str, float | int] = {
        "slice_count": len(actual),
        "index_mismatches": 0,
        "max_abs_selector_probability": 0.0,
        "max_abs_peak_probability": 0.0,
        "max_abs_mls_mm": 0.0,
        "max_abs_heatmap_peak": 0.0,
    }
    for packaged, reference in zip(actual, expected, strict=True):
        if int(packaged.index) != int(reference["index"]):
            result["index_mismatches"] = int(result["index_mismatches"]) + 1
        for field, result_key in (
            ("selector_probability", "max_abs_selector_probability"),
            ("peak_probability", "max_abs_peak_probability"),
            ("mls_mm", "max_abs_mls_mm"),
            ("heatmap_peak", "max_abs_heatmap_peak"),
        ):
            residual = abs(_value(packaged, field) - _value(reference, field))
            result[result_key] = max(float(result[result_key]), residual)
    return result


def _flatten_parity(prefix: str, payload: dict[str, float | int]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def _reference_records(
    spec: dict[str, Any], repo_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile = PROFILES[spec["profile"]]
    records: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold_spec in spec["folds"]:
        fold = int(fold_spec["fold"])
        expected = int(fold_spec["expected_studies"])
        baseline_path = repo_root / fold_spec["baseline"]
        baseline_frame, baseline_payloads = _load(baseline_path, expected)
        member_payloads = baseline_payloads
        challenger_payloads: list[list[dict[str, Any]]] | None = None
        challenger_path: Path | None = None
        if fold_spec.get("challenger"):
            challenger_path = repo_root / fold_spec["challenger"]
            challenger_frame, challenger_payloads = _load(challenger_path, expected)
            _validate_alignment(
                baseline_frame,
                baseline_payloads,
                challenger_frame,
                challenger_payloads,
                str(fold_spec["challenger_label"]),
            )
            member_payloads = _candidate_payloads(
                baseline_payloads,
                challenger_payloads,
                float(fold_spec["alpha"]),
                str(fold_spec["component_mode"]),
            )
        hashes.append(
            {"fold": fold, "role": "baseline", "path": str(baseline_path), "sha256": _sha256(baseline_path)}
        )
        if challenger_path is not None:
            hashes.append(
                {"fold": fold, "role": "challenger", "path": str(challenger_path), "sha256": _sha256(challenger_path)}
            )
        for row_index, row in baseline_frame.iterrows():
            study_id = str(row["study_id"])
            if study_id in seen:
                raise RuntimeError(f"OOF study overlap: {study_id}")
            seen.add(study_id)
            records.append(
                {
                    "fold": fold,
                    "study_id": study_id,
                    "gt_MLS_mm": float(row["gt_MLS_mm"]),
                    "baseline": baseline_payloads[row_index],
                    "challenger": None if challenger_payloads is None else challenger_payloads[row_index],
                    "member": member_payloads[row_index],
                    "expected_baseline_value": float(_predict(baseline_payloads[row_index], profile)),
                    "expected_member_value": float(_predict(member_payloads[row_index], profile)),
                }
            )
    return records, hashes


def _render(summary: dict[str, Any]) -> str:
    baseline = summary["micro_oof"]["baseline"]
    candidate = summary["micro_oof"]["packaged_candidate"]
    parity = summary["parity_maxima"]
    return "\n".join(
        [
            "# Conservative MLS package CUDA audit",
            "",
            f"- State: `{summary['state']}`",
            f"- GPU: `{summary['cuda_device']}`",
            f"- Studies: `{summary['n_studies']}`",
            f"- Runtime: `{summary['runtime_total_s']:.3f}s`",
            f"- Peak VRAM: `{summary['peak_vram_gb']:.6f}GiB`",
            f"- Baseline MAE: `{baseline['mae_mm']:.9f}`",
            f"- Packaged candidate MAE: `{candidate['mae_mm']:.9f}`",
            f"- Baseline Boundary-F1: `{baseline['boundary_f1']:.9f}`",
            f"- Packaged candidate Boundary-F1: `{candidate['boundary_f1']:.9f}`",
            f"- Baseline objective: `{baseline['selection_objective']:.9f}`",
            f"- Packaged candidate objective: `{candidate['selection_objective']:.9f}`",
            f"- Max member aggregation residual: `{parity['max_abs_member_value_mm']:.9g}mm`",
            f"- Index mismatches: `{parity['index_mismatches']}`",
            f"- Gate passed: `{summary['passed']}`",
            "",
            "All model forwards were CUDA-only. Raw per-study output remains on Vast.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only package audit found no GPU")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    frozen_oof = json.loads(args.oof_summary.read_text(encoding="utf-8"))
    references, input_hashes = _reference_records(spec, args.repo_root.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    raw_path = output_dir / "study_member_predictions.csv"
    identity = {
        "archive_sha256": _sha256(args.archive),
        "runtime_sha256": _sha256(args.runtime_root / "mls.py"),
        "spec_sha256": _sha256(args.spec),
        "oof_summary_sha256": _sha256(args.oof_summary),
    }
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "started_utc": _utc(),
        "finished_utc": None,
        "compute_policy": "cuda_only_model_forward_saved_prediction_postprocessing",
        "cuda_device": torch.cuda.get_device_name(0),
        "expected_studies": len(references),
        "completed_studies": 0,
        **identity,
    }
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if previous.get(key) != value:
                raise RuntimeError(f"Resume identity mismatch for {key}")
        status["started_utc"] = previous.get("started_utc", status["started_utc"])
    _atomic_json(status_path, status)

    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if raw_path.is_file() and raw_path.stat().st_size:
        old = pd.read_csv(raw_path, dtype={"study_id": str})
        if old["study_id"].duplicated().any():
            raise RuntimeError("Duplicate study in resume CSV")
        rows = old.to_dict("records")
        completed = set(old["study_id"].astype(str))
    runtime = _runtime(args.runtime_root.resolve())
    torch.cuda.reset_peak_memory_stats()
    predictor = runtime.MLSEnsemblePredictor(args.runtime_root / "models" / "mls", device)
    if set(predictor.models) != set(runtime.MODEL_FILES):
        raise RuntimeError("Packaged runtime did not load the five frozen checkpoints")
    audit_started = time.perf_counter()
    try:
        for ordinal, reference in enumerate(references, start=1):
            study_id = reference["study_id"]
            if study_id in completed:
                continue
            study_path = args.data_root / study_id
            if not study_path.is_dir():
                raise FileNotFoundError(study_path)
            started = time.perf_counter()
            detail = predictor.predict_detailed(runtime.DicomStudy(study_path), batch_size=args.batch_size)
            fold = int(reference["fold"])
            baseline_label = f"fold{fold}"
            challenger_label = f"fold{fold}_regression"
            baseline_parity = _slice_parity(
                detail["raw_predictions"][baseline_label], reference["baseline"]
            )
            challenger_parity = {
                "slice_count": 0,
                "index_mismatches": 0,
                "max_abs_selector_probability": 0.0,
                "max_abs_peak_probability": 0.0,
                "max_abs_mls_mm": 0.0,
                "max_abs_heatmap_peak": 0.0,
            }
            if reference["challenger"] is not None:
                challenger_parity = _slice_parity(
                    detail["raw_predictions"][challenger_label], reference["challenger"]
                )
            member_parity = _slice_parity(
                detail["member_predictions"][baseline_label], reference["member"]
            )
            member_value = float(detail["member_values"][baseline_label])
            row = {
                "fold": fold,
                "study_id": study_id,
                "gt_MLS_mm": float(reference["gt_MLS_mm"]),
                "expected_baseline_value": float(reference["expected_baseline_value"]),
                "expected_member_value": float(reference["expected_member_value"]),
                "packaged_member_value": member_value,
                "packaged_ensemble_diagnostic": float(detail["ensemble"]),
                "abs_member_value_delta_mm": abs(member_value - float(reference["expected_member_value"])),
                "runtime_s": time.perf_counter() - started,
                **_flatten_parity("baseline", baseline_parity),
                **_flatten_parity("challenger", challenger_parity),
                **_flatten_parity("member", member_parity),
            }
            rows.append(row)
            completed.add(study_id)
            _atomic_csv(raw_path, pd.DataFrame(rows))
            status["completed_studies"] = len(rows)
            _atomic_json(status_path, status)
            print(f"Conservative package audit {study_id}: {len(rows)}/{len(references)}", flush=True)
            torch.cuda.empty_cache()

        frame = pd.DataFrame(rows).sort_values(["fold", "study_id"]).reset_index(drop=True)
        if len(frame) != len(references) or frame["study_id"].nunique() != len(references):
            raise RuntimeError("Completed audit row contract failed")
        truth = frame["gt_MLS_mm"].to_numpy(float)
        baseline = _metric_payload(truth, frame["expected_baseline_value"].to_numpy(float))
        packaged = _metric_payload(truth, frame["packaged_member_value"].to_numpy(float))
        expected_oof_baseline = frozen_oof["micro_oof"]["baseline"]
        expected_oof_candidate = frozen_oof["micro_oof"]["candidate"]
        metric_residuals: dict[str, float] = {}
        for metric in ("mae_mm", "boundary_f1", "selection_objective"):
            metric_residuals[f"baseline_{metric}"] = abs(
                baseline[metric] - float(expected_oof_baseline[metric])
            )
            metric_residuals[f"candidate_{metric}"] = abs(
                packaged[metric] - float(expected_oof_candidate[metric])
            )
        parity_columns = [
            column for column in frame.columns
            if "max_abs_" in column or "index_mismatches" in column
        ]
        parity_maxima = {
            column: float(frame[column].max()) for column in parity_columns
        }
        parity_maxima["index_mismatches"] = int(
            sum(
                int(frame[column].sum())
                for column in frame.columns if column.endswith("index_mismatches")
            )
        )
        parity_maxima["max_abs_member_value_mm"] = float(
            frame["abs_member_value_delta_mm"].max()
        )
        gates = {
            "index_parity": parity_maxima["index_mismatches"] == 0,
            "selector_parity": max(
                value for key, value in parity_maxima.items()
                if key.endswith("max_abs_selector_probability")
            ) <= SLICE_SELECTOR_TOL,
            "peak_selector_parity": max(
                value for key, value in parity_maxima.items()
                if key.endswith("max_abs_peak_probability")
            ) <= SLICE_PEAK_SELECTOR_TOL,
            "heatmap_parity": max(
                value for key, value in parity_maxima.items()
                if key.endswith("max_abs_heatmap_peak")
            ) <= SLICE_HEATMAP_TOL,
            "slice_mls_parity": max(
                value for key, value in parity_maxima.items()
                if key.endswith("max_abs_mls_mm")
            ) <= SLICE_MLS_TOL_MM,
            "member_value_parity": parity_maxima["max_abs_member_value_mm"] <= MEMBER_TOL_MM,
            "oof_metric_parity": max(metric_residuals.values()) <= OOF_METRIC_TOL,
        }
        passed = bool(all(gates.values()))
        summary = {
            "schema_version": 1,
            "state": "completed" if passed else "failed",
            "started_utc": status["started_utc"],
            "finished_utc": _utc(),
            "compute_policy": status["compute_policy"],
            "cuda_device": status["cuda_device"],
            "n_studies": len(frame),
            "runtime_total_s": float(frame["runtime_s"].sum()),
            "wall_runtime_s": time.perf_counter() - audit_started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
            **identity,
            "input_hashes": input_hashes,
            "micro_oof": {"baseline": baseline, "packaged_candidate": packaged},
            "metric_residuals_vs_frozen_oof": metric_residuals,
            "parity_maxima": parity_maxima,
            "tolerances": {
                "selector": SLICE_SELECTOR_TOL,
                "peak_selector": SLICE_PEAK_SELECTOR_TOL,
                "heatmap_peak": SLICE_HEATMAP_TOL,
                "slice_mls_mm": SLICE_MLS_TOL_MM,
                "member_value_mm": MEMBER_TOL_MM,
                "oof_metric": OOF_METRIC_TOL,
            },
            "gates": gates,
            "passed": passed,
            "raw_artifact": str(raw_path),
        }
        _atomic_json(output_dir / "package_oof_audit_summary.json", summary)
        _atomic_text(output_dir / "PACKAGE_OOF_AUDIT_REPORT.md", _render(summary))
        status.update(
            {
                "state": summary["state"],
                "finished_utc": summary["finished_utc"],
                "completed_studies": len(frame),
                "passed": passed,
            }
        )
        _atomic_json(status_path, status)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if passed else 2
    except BaseException as exc:
        status.update({"state": "failed", "finished_utc": _utc(), "error": repr(exc)})
        _atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
