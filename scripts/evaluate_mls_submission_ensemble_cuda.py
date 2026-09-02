"""CUDA-only integration audit for the deployable MLS ensemble.

The script imports the *packaged* MLS runtime from an extracted submission
directory, runs the three current fold checkpoints plus one challenger in a
single DICOM pass, and compares:

* the current median ensemble,
* the challenger median ensemble, and
* the baseline/challenger checkpoints for one replaced fold on that fold's OOF
  studies.

The ensemble comparison on the replacement fold is diagnostic only: the other
fold models were not held out from it.  The replaced-fold single-model
comparison is the unbiased OOF promotion evidence.  When a reference
slice-prediction CSV is supplied, the challenger's packaged inference is also
checked slice-by-slice against the independent evaluator cache.
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
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.splits import load_fold_manifest


SAFE_MODEL_LABELS = {
    *(f"fold{fold}" for fold in range(3)),
    *(f"fold{fold}_{suffix}" for fold in range(3) for suffix in ("baseline", "challenger")),
}


def _ensemble_contract(
    replacement_fold: int,
) -> tuple[set[str], tuple[str, ...], tuple[str, ...], str, str]:
    if replacement_fold not in {0, 1, 2}:
        raise ValueError(f"Unsupported replacement fold: {replacement_fold}")
    baseline_label = f"fold{replacement_fold}_baseline"
    challenger_label = f"fold{replacement_fold}_challenger"
    required = {
        baseline_label,
        challenger_label,
        *(f"fold{fold}" for fold in range(3) if fold != replacement_fold),
    }
    baseline = tuple(
        baseline_label if fold == replacement_fold else f"fold{fold}"
        for fold in range(3)
    )
    challenger = tuple(
        challenger_label if fold == replacement_fold else f"fold{fold}"
        for fold in range(3)
    )
    return required, baseline, challenger, baseline_label, challenger_label


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=CHECKPOINT")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if label not in SAFE_MODEL_LABELS:
        raise argparse.ArgumentTypeError(
            f"unsupported model label {label!r}; expected one of {sorted(SAFE_MODEL_LABELS)}"
        )
    return label, Path(raw_path).expanduser()


def _load_runtime(runtime_root: Path) -> ModuleType:
    runtime_root = runtime_root.resolve()
    module_path = runtime_root / "mls.py"
    dicom_path = runtime_root / "dicom_io.py"
    if not module_path.is_file() or not dicom_path.is_file():
        raise FileNotFoundError(
            f"Packaged MLS runtime requires mls.py and dicom_io.py under {runtime_root}"
        )
    sys.path.insert(0, str(runtime_root))
    specification = importlib.util.spec_from_file_location(
        "frozen_submission_mls_runtime", module_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import packaged MLS runtime: {module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    f1_3 = float(f1_score(truth >= 3.0, prediction >= 3.0, zero_division=0))
    f1_5 = float(f1_score(truth >= 5.0, prediction >= 5.0, zero_division=0))
    boundary = (f1_3 + f1_5) / 2.0
    mae = float(np.mean(np.abs(prediction - truth)))
    return {
        "mae_mm": mae,
        "rmse_mm": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "bias_mm": float(np.mean(prediction - truth)),
        "f1_3mm": f1_3,
        "f1_5mm": f1_5,
        "boundary_f1": boundary,
        "selection_objective": mae + 2.0 * (1.0 - boundary),
    }


def _load_reference(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype={"study_id": str})
    required = {"study_id", "slice_predictions_json"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Reference cache is missing columns: {sorted(missing)}")
    return {
        str(row.study_id): json.loads(str(row.slice_predictions_json))
        for row in frame.itertuples(index=False)
    }


def _parity(
    packaged: list[Any],
    reference: list[dict[str, Any]],
) -> dict[str, float | int]:
    if len(packaged) != len(reference):
        raise RuntimeError(
            f"Slice-count parity failed: packaged={len(packaged)} reference={len(reference)}"
        )
    max_selector = 0.0
    max_mls = 0.0
    max_peak = 0.0
    index_mismatches = 0
    for actual, expected in zip(packaged, reference, strict=True):
        if int(actual.index) != int(expected["index"]):
            index_mismatches += 1
        max_selector = max(
            max_selector,
            abs(float(actual.selector_probability) - float(expected["selector_probability"])),
        )
        max_mls = max(max_mls, abs(float(actual.mls_mm) - float(expected["mls_mm"])))
        max_peak = max(
            max_peak, abs(float(actual.heatmap_peak) - float(expected["heatmap_peak"]))
        )
    return {
        "slice_count": len(packaged),
        "index_mismatches": index_mismatches,
        "max_abs_selector_probability": max_selector,
        "max_abs_mls_mm": max_mls,
        "max_abs_heatmap_peak": max_peak,
    }


def _load_models(
    runtime: ModuleType,
    model_paths: dict[str, Path],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    models: dict[str, torch.nn.Module] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for label in sorted(model_paths):
        checkpoint = model_paths[label].resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(payload["config"])
        if not bool(config.get("use_selector")):
            raise ValueError(f"Checkpoint lacks selector: {checkpoint}")
        if int(config.get("input_channels", 3)) != 3:
            raise ValueError(f"Packaged runtime supports only three channels: {checkpoint}")
        if int(config.get("image_size", runtime.IMAGE_SIZE)) != int(runtime.IMAGE_SIZE):
            raise ValueError(f"Checkpoint/runtime image-size mismatch: {checkpoint}")
        model = runtime.HRNetMultitask(
            str(config["backbone"]), float(config.get("head_dropout", 0.0))
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model = model.to(device).eval()
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError(f"CUDA guard failed while loading {checkpoint}")
        models[label] = model
        manifest[label] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "backbone": str(config["backbone"]),
            "epoch": int(payload.get("epoch", -1)),
        }
        del payload
    return models, manifest


@torch.inference_mode()
def _predict_study(
    runtime: ModuleType,
    study_dir: Path,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, list[Any]], float]:
    started = time.perf_counter()
    study = runtime.DicomStudy(study_dir)
    volume = study.volume_hwd
    if volume.shape[0] != runtime.IMAGE_SIZE or volume.shape[1] != runtime.IMAGE_SIZE:
        raise RuntimeError(
            "CPU resize is disabled by the audit compute policy; "
            f"found DICOM shape {tuple(volume.shape[:2])}"
        )
    effective_spacing = study.spacing_x * (study.rows / float(runtime.IMAGE_SIZE))
    predictions: dict[str, list[Any]] = {label: [] for label in models}
    for start in range(0, volume.shape[2], batch_size):
        tensors = [
            torch.from_numpy(runtime._input(volume[:, :, index])).unsqueeze(0)
            for index in range(start, min(volume.shape[2], start + batch_size))
        ]
        inputs = torch.cat(tensors).to(device, non_blocking=True)
        if inputs.device.type != "cuda":
            raise RuntimeError("MLS input tensor left CUDA")
        for label, model in models.items():
            heatmap_logits, selector_logits = model.forward_multitask(inputs)
            if heatmap_logits.device.type != "cuda" or selector_logits.device.type != "cuda":
                raise RuntimeError(f"Model output left CUDA: {label}")
            if not torch.isfinite(heatmap_logits).all() or not torch.isfinite(
                selector_logits
            ).all():
                raise FloatingPointError(f"Non-finite MLS output: {label}")
            probabilities = torch.softmax(
                heatmap_logits.flatten(2), dim=-1
            ).reshape_as(heatmap_logits)
            coordinates, peaks = runtime._decode_batch(probabilities.cpu())
            selectors = torch.sigmoid(selector_logits).cpu().numpy()
            for offset, keypoints in enumerate(coordinates):
                measurement = (
                    runtime._measurement(keypoints, effective_spacing)
                    if (keypoints[:, 0] >= 0).all()
                    else 0.0
                )
                predictions[label].append(
                    runtime.SlicePrediction(
                        index=start + offset,
                        selector_probability=float(selectors[offset]),
                        mls_mm=float(measurement),
                        heatmap_peak=float(np.min(peaks[offset])),
                    )
                )
        del inputs
    fold_values = {
        label: float(np.clip(runtime._aggregate(rows), 0.0, 30.0))
        for label, rows in predictions.items()
    }
    return fold_values, predictions, time.perf_counter() - started


def _render_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    delta = result["paired_delta_challenger_minus_baseline"]
    parity = result["reference_parity"]
    baseline_label = result["single_model_labels"]["baseline"]
    challenger_label = result["single_model_labels"]["challenger"]
    replacement_fold = int(result["replacement_fold"])
    lines = [
        "# MLS submission integration audit",
        "",
        f"- Finished UTC: `{result['finished_utc']}`",
        f"- GPU: `{result['cuda_device']}`",
        f"- Studies: `{result['n_studies']}`",
        "- Compute policy: model forward passes are CUDA-only; no CPU resize was allowed.",
        f"- Interpretation: single-model fold{replacement_fold} rows are OOF evidence; "
        "ensemble rows are diagnostic only.",
        "",
        "## Metrics",
        "",
        "| Candidate | MAE (mm) | Boundary F1 | Objective | Bias (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        baseline_label,
        challenger_label,
        "baseline_ensemble_diagnostic",
        "challenger_ensemble_diagnostic",
    ):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['mae_mm']:.6f} | {row['boundary_f1']:.6f} | "
            f"{row['selection_objective']:.6f} | {row['bias_mm']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired diagnostic ensemble delta",
            "",
            f"- MAE delta: `{delta['mae_mm']:+.6f}` mm",
            f"- Boundary-F1 delta: `{delta['boundary_f1']:+.6f}`",
            f"- Improved/worse/tied studies by absolute error: "
            f"`{delta['studies_improved']}/{delta['studies_worse']}/{delta['studies_tied']}`",
            "",
            "## Packaged-runtime parity",
            "",
            f"- Reference available: `{parity['reference_available']}`",
            f"- Studies checked: `{parity['studies_checked']}`",
            f"- Index mismatches: `{parity['index_mismatches']}`",
            f"- Max |selector delta|: `{parity['max_abs_selector_probability']:.9g}`",
            f"- Max |MLS delta|: `{parity['max_abs_mls_mm']:.9g}` mm",
            f"- Max |heatmap-peak delta|: `{parity['max_abs_heatmap_peak']:.9g}`",
            f"- Gate passed: `{parity['passed']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--replacement-fold", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--expected-studies", type=int, default=67)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "Data" / "raw" / "training")
    parser.add_argument("--truth-table", type=Path, default=PROJECT_ROOT / "reports" / "eda" / "deep" / "deep_series_table.csv")
    parser.add_argument("--reference-slice-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mlflow-run-id", default="")
    args = parser.parse_args()

    (
        required_labels,
        baseline_ensemble,
        challenger_ensemble,
        baseline_label,
        challenger_label,
    ) = _ensemble_contract(args.replacement_fold)
    evaluation_fold = args.replacement_fold if args.fold is None else args.fold
    supplied = dict(args.model)
    if set(supplied) != required_labels or len(args.model) != len(required_labels):
        raise ValueError(f"Exactly these model labels are required: {sorted(required_labels)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-only MLS integration audit found no GPU")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    runtime = _load_runtime(args.runtime_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status = {
        "schema_version": 1,
        "state": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "compute_policy": "cuda_only_model_forward_no_cpu_resize",
        "cuda_device": torch.cuda.get_device_name(0),
        "fold": evaluation_fold,
        "replacement_fold": args.replacement_fold,
        "expected_studies": args.expected_studies,
        "completed_studies": 0,
    }
    _atomic_json(status_path, status)

    models, model_manifest = _load_models(runtime, supplied, device)
    reference = _load_reference(args.reference_slice_csv)
    folds = load_fold_manifest()
    manifest = folds.loc[
        folds["fold"] == evaluation_fold, ["study_id", "patient_id", "triage_class"]
    ].copy()
    manifest["study_id"] = manifest["study_id"].astype(str)
    truth = pd.read_csv(
        args.truth_table, dtype={"dicom_series.id": str}
    )[["dicom_series.id", "MLS_mm"]].rename(
        columns={"dicom_series.id": "study_id", "MLS_mm": "gt_MLS_mm"}
    )
    frame = manifest.merge(truth, on="study_id", how="left", validate="one_to_one")
    if len(frame) != args.expected_studies or frame["gt_MLS_mm"].isna().any():
        raise RuntimeError(
            f"Fold contract failed: rows={len(frame)}, missing_truth={int(frame['gt_MLS_mm'].isna().sum())}"
        )

    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, float | int]] = []
    for ordinal, row in enumerate(frame.itertuples(index=False), start=1):
        study_id = str(row.study_id)
        fold_values, slices, runtime_s = _predict_study(
            runtime,
            args.data_root / study_id,
            models,
            device,
            args.batch_size,
        )
        baseline_value = float(np.median([fold_values[label] for label in baseline_ensemble]))
        challenger_value = float(
            np.median([fold_values[label] for label in challenger_ensemble])
        )
        parity_row: dict[str, float | int] = {
            "slice_count": len(slices[challenger_label]),
            "index_mismatches": 0,
            "max_abs_selector_probability": 0.0,
            "max_abs_mls_mm": 0.0,
            "max_abs_heatmap_peak": 0.0,
        }
        if reference:
            if study_id not in reference:
                raise RuntimeError(f"Reference cache lacks study {study_id}")
            parity_row = _parity(slices[challenger_label], reference[study_id])
        parity_rows.append(parity_row)
        rows.append(
            {
                "study_id": study_id,
                "gt_MLS_mm": float(row.gt_MLS_mm),
                **{f"pred_{label}": value for label, value in fold_values.items()},
                "pred_baseline_ensemble": baseline_value,
                "pred_challenger_ensemble": challenger_value,
                "runtime_s": runtime_s,
                **{f"parity_{key}": value for key, value in parity_row.items()},
            }
        )
        _atomic_csv(pd.DataFrame(rows), output_dir / "study_predictions.csv")
        status["completed_studies"] = ordinal
        _atomic_json(status_path, status)
        print(f"MLS packaged integration {study_id}: {ordinal}/{len(frame)}", flush=True)
        torch.cuda.empty_cache()

    predictions = pd.DataFrame(rows)
    truth_values = predictions["gt_MLS_mm"].to_numpy(float)
    metric_inputs = {
        baseline_label: predictions[f"pred_{baseline_label}"].to_numpy(float),
        challenger_label: predictions[f"pred_{challenger_label}"].to_numpy(float),
        "baseline_ensemble_diagnostic": predictions["pred_baseline_ensemble"].to_numpy(float),
        "challenger_ensemble_diagnostic": predictions["pred_challenger_ensemble"].to_numpy(float),
    }
    metrics = {name: _metrics(truth_values, values) for name, values in metric_inputs.items()}
    baseline_error = np.abs(metric_inputs["baseline_ensemble_diagnostic"] - truth_values)
    challenger_error = np.abs(metric_inputs["challenger_ensemble_diagnostic"] - truth_values)
    error_delta = challenger_error - baseline_error
    ensemble_delta = {
        key: metrics["challenger_ensemble_diagnostic"][key]
        - metrics["baseline_ensemble_diagnostic"][key]
        for key in metrics["baseline_ensemble_diagnostic"]
    }
    ensemble_delta.update(
        {
            "studies_improved": int(np.count_nonzero(error_delta < -1e-9)),
            "studies_worse": int(np.count_nonzero(error_delta > 1e-9)),
            "studies_tied": int(np.count_nonzero(np.abs(error_delta) <= 1e-9)),
        }
    )
    parity_summary = {
        "reference_available": bool(reference),
        "studies_checked": len(parity_rows) if reference else 0,
        "index_mismatches": int(sum(int(item["index_mismatches"]) for item in parity_rows)),
        "max_abs_selector_probability": float(
            max(float(item["max_abs_selector_probability"]) for item in parity_rows)
        ),
        "max_abs_mls_mm": float(max(float(item["max_abs_mls_mm"]) for item in parity_rows)),
        "max_abs_heatmap_peak": float(
            max(float(item["max_abs_heatmap_peak"]) for item in parity_rows)
        ),
    }
    parity_summary["passed"] = bool(
        reference
        and parity_summary["index_mismatches"] == 0
        and parity_summary["max_abs_selector_probability"] <= 2e-6
        and parity_summary["max_abs_mls_mm"] <= 2e-5
        and parity_summary["max_abs_heatmap_peak"] <= 2e-8
    )
    if reference and not parity_summary["passed"]:
        status.update({"state": "failed", "finished_utc": _utc_now()})
        _atomic_json(status_path, status)
        raise RuntimeError(f"Packaged-runtime parity gate failed: {parity_summary}")

    result = {
        "schema_version": 1,
        "state": "completed",
        "started_utc": status["started_utc"],
        "finished_utc": _utc_now(),
        "compute_policy": status["compute_policy"],
        "cuda_device": status["cuda_device"],
        "fold": evaluation_fold,
        "replacement_fold": args.replacement_fold,
        "single_model_labels": {
            "baseline": baseline_label,
            "challenger": challenger_label,
        },
        "n_studies": len(predictions),
        "runtime_total_s": float(predictions["runtime_s"].sum()),
        "models": model_manifest,
        "metrics": metrics,
        "paired_delta_challenger_minus_baseline": ensemble_delta,
        "reference_parity": parity_summary,
        "interpretation": {
            f"fold{args.replacement_fold}_single_models": "OOF promotion evidence",
            "ensembles": (
                "diagnostic only because the non-replacement fold models trained with "
                f"fold{args.replacement_fold} studies"
            ),
        },
    }
    _atomic_json(output_dir / "comparison.json", result)
    _atomic_text(output_dir / "comparison_report.md", _render_report(result))
    status.update(
        {
            "state": "completed",
            "finished_utc": result["finished_utc"],
            "completed_studies": len(predictions),
            "parity_passed": parity_summary["passed"],
        }
    )
    _atomic_json(status_path, status)

    if args.mlflow_run_id:
        from mlflow.tracking import MlflowClient

        from src.config import config_section
        from src.mlops.tracking import configure_tracking_environment

        configure_tracking_environment()
        client = MlflowClient()
        for candidate, candidate_metrics in metrics.items():
            for key, value in candidate_metrics.items():
                client.log_metric(
                    args.mlflow_run_id,
                    f"submission_integration_{candidate}_{key}",
                    float(value),
                )
        client.log_metric(
            args.mlflow_run_id,
            "submission_integration_parity_max_abs_mls_mm",
            float(parity_summary["max_abs_mls_mm"]),
        )
        artifact_root = config_section("mlflow", "artifact_paths", "reports")
        for name in ("comparison.json", "comparison_report.md", "status.json"):
            client.log_artifact(
                args.mlflow_run_id,
                str(output_dir / name),
                f"{artifact_root}/submission_integration",
            )
        client.set_tag(args.mlflow_run_id, "submission_runtime_parity", str(parity_summary["passed"]).lower())

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
