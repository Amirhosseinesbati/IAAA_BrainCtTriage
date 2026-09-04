"""Fixed baseline training-only CUDA probe; persist aggregate decoder evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_mls_a2_fold0_resource_screen import _atomic_json
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model
from src.strategies.mls_heatmap.train import (
    differentiable_keypoints_from_heatmaps,
    differentiable_mls_mm,
)
from src.strategies.mls_heatmap.train_multitask import configure_training_determinism
from src.strategies.mls_heatmap.utils import decode_heatmap_dark_batch

EXPECTED_CHECKPOINT = "c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062"
EXPECTED_LABELS = "01512662b62bcaf484f99cb872c40e28e2cfb300adee60db40957db0d06001ad"
EXPECTED_FOLDS = "d3c4640aec8fbfd8a912286bbf40ee39a7f48756c899cafcf8d976ce664ce2b8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(frame, validation_ids: set[str], image_size: int) -> tuple[list[int], str]:
    """Select the frozen sample without model outputs or held-out observations."""
    if set(frame["patient_id"].astype(str)) & validation_ids:
        raise ValueError("Training population overlaps the held-out fold")
    ranked = []
    identities = set()
    for index, row in frame.iterrows():
        coords = np.asarray([row[k] for k in ("x1", "y1", "x2", "y2", "x3", "y3")], dtype=float)
        if float(row["is_target"]) <= 0.5 or not np.isfinite(coords).all():
            continue
        if np.any(coords < 0) or np.any(coords >= image_size):
            continue
        identity = f"20260904|{row['patient_id']}|{row['image_name']}"
        if identity in identities:
            raise ValueError("Duplicate eligible sample identity")
        identities.add(identity)
        ranked.append((hashlib.sha256(identity.encode()).hexdigest(), int(index)))
    ranked.sort()
    if len(ranked) < 64:
        raise ValueError("Fewer than 64 eligible training rows")
    selected = ranked[:128]
    digest = hashlib.sha256("\n".join(item[0] for item in selected).encode()).hexdigest()
    return [item[1] for item in selected], digest


def summarize(soft, dark, truth, spacing) -> dict:
    for value in (soft, dark, truth, spacing):
        if not np.isfinite(value).all():
            raise ValueError("Non-finite diagnostic coordinates")
    if np.any(dark < 0) or np.any(spacing <= 0):
        raise ValueError("Invalid decoded coordinate or spacing")

    def mls(coords):
        a, b, c = coords[:, 0], coords[:, 1], coords[:, 2]
        direction = b - a
        numerator = np.abs(direction[:, 0] * (a[:, 1] - c[:, 1]) - (a[:, 0] - c[:, 0]) * direction[:, 1])
        return numerator / np.maximum(np.linalg.norm(direction, axis=1), 1e-6) * spacing

    true_mls = mls(truth)
    summaries = {}
    for name, coords in (("softargmax", soft), ("dark", dark)):
        distance = np.linalg.norm(coords - truth, axis=2) * spacing[:, None]
        prediction = mls(coords)
        summaries[name] = {
            "landmark_mean_error_mm": distance.mean(axis=0).tolist(),
            "landmark_median_error_mm": np.median(distance, axis=0).tolist(),
            "slice_mls_mae_mm": float(np.abs(prediction - true_mls).mean()),
            "slice_mls_bias_mm": float((prediction - true_mls).mean()),
            "slice_f1_3mm": float(f1_score(true_mls >= 3, prediction >= 3, zero_division=0)),
            "slice_f1_5mm": float(f1_score(true_mls >= 5, prediction >= 5, zero_division=0)),
        }
    gap = np.linalg.norm(soft - dark, axis=2) * spacing[:, None]
    return {
        "decoders": summaries,
        "inter_decoder_coordinate_mean_mm": float(gap.mean()),
        "inter_decoder_coordinate_median_mm": float(np.median(gap)),
        "inter_decoder_coordinate_p90_mm": float(np.quantile(gap, 0.9)),
        "inter_decoder_mls_mean_absolute_difference_mm": float(np.abs(mls(soft) - mls(dark)).mean()),
        "finite_forward_examples": len(soft),
        "valid_coordinate_examples": len(soft),
    }


def log_aggregate(output: Path, result: dict) -> dict:
    """Use the campaign's existing MLS experiment; upload this one aggregate file."""
    run_id = None
    try:
        from mlflow.tracking import MlflowClient
        from src.mlops.tracking import configure_tracking_environment

        configure_tracking_environment()
        client = MlflowClient()
        experiment_id = client.get_run("6983886c0bea419696087a39cc6c8478").info.experiment_id
        run = client.create_run(experiment_id, tags={
            "mlflow.runName": "mls-baseline-training-decoder-probe-20260904",
            "scope": "training_only_mechanistic_diagnostic", "promotion_eligible": "false",
            "checkpoint_sha256": EXPECTED_CHECKPOINT,
        })
        run_id = run.info.run_id
        for name in ("inter_decoder_coordinate_mean_mm", "inter_decoder_coordinate_median_mm", "inter_decoder_coordinate_p90_mm", "inter_decoder_mls_mean_absolute_difference_mm", "sample_count", "represented_studies"):
            client.log_metric(run_id, name, float(result[name]))
        for decoder, values in result["decoders"].items():
            for key, value in values.items():
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        client.log_metric(run_id, f"{decoder}_{key}_{index}", float(item))
                else:
                    client.log_metric(run_id, f"{decoder}_{key}", float(value))
        client.log_artifact(run_id, str(output.resolve()), "reports/training_decoder_probe")
        client.set_terminated(run_id, status="FINISHED")
        return {"status": "logged", "run_id": run_id}
    except Exception as exc:
        return {"status": "deferred", "run_id": run_id, "error_type": type(exc).__name__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status_path = args.output.with_suffix(".status.json")
    if args.output.exists() or status_path.exists():
        raise FileExistsError("Refusing to overwrite a previous diagnostic")
    lock = Path("/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/gpu_training.lock")
    acquired = False
    started = time.monotonic()
    try:
        lock.mkdir()
        acquired = True
        if subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True).strip():
            raise RuntimeError("Another CUDA workload is active")
        _atomic_json(status_path, {"state": "running", "compute_policy": "cuda_only_no_cpu_model_fallback"})
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        configure_training_determinism("strict")
        checkpoint = ROOT / "models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth"
        labels = ROOT / "Data/processed/mls_multitask_v2/mls_labels_multitask.csv"
        folds = ROOT / "config/folds.csv"
        for path, expected in ((checkpoint, EXPECTED_CHECKPOINT), (labels, EXPECTED_LABELS), (folds, EXPECTED_FOLDS)):
            if sha256(path) != expected:
                raise ValueError("A preregistered input hash does not match")
        device = torch.device("cuda:0")
        model, config = load_multitask_model(checkpoint, device)
        if config.fold != 0 or config.seed != 42 or not config.use_competition_folds:
            raise ValueError("Wrong checkpoint training population")
        train, val = create_mls_dataloaders(
            str(labels), str(labels.parent / "images"), img_size=config.image_size,
            heatmap_size=config.image_size // 4, heatmap_sigma=config.heatmap_sigma,
            batch_size=8, augment=False, num_workers=0, fold=0, seed=42,
            use_competition_folds=True, include_negatives=True, return_selector=True,
        )
        indices, sample_digest = select_rows(train.dataset.data, set(val.dataset.data["patient_id"].astype(str)), config.image_size)
        loader = DataLoader(Subset(train.dataset, indices), batch_size=8, shuffle=False, num_workers=2)
        soft_all, dark_all, truth_all, spacing_all = [], [], [], []
        with torch.inference_mode():
            for images, _targets, masks, truth, spacing, is_target, _study_mls, _ids in loader:
                if not (masks > 0.5).all() or not (is_target > 0.5).all():
                    raise ValueError("Selected training sample has invalid annotation masks")
                logits, selector = model.forward_multitask(images.to(device))
                if not torch.isfinite(logits).all() or not torch.isfinite(selector).all():
                    raise FloatingPointError("Non-finite CUDA forward output")
                soft = differentiable_keypoints_from_heatmaps(logits, config.image_size, config.softargmax_temperature)
                probabilities = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
                dark, _peaks = decode_heatmap_dark_batch(probabilities.cpu(), logits.shape[-1], config.image_size)
                # Verify that training-side geometry is finite as well.
                soft_mls = differentiable_mls_mm(soft, spacing.to(device))
                if not torch.isfinite(soft_mls).all():
                    raise FloatingPointError("Non-finite CUDA MLS calculation")
                soft_all.append(soft.cpu().numpy())
                dark_all.append(dark)
                truth_all.append(truth.numpy())
                spacing_all.append(spacing.numpy())
        result = summarize(*(np.concatenate(items) for items in (soft_all, dark_all, truth_all, spacing_all)))
        result.update({
            "schema_version": 1, "status": "completed", "scope": "training_only_mechanistic_diagnostic",
            "compute_policy": "cuda_only_no_cpu_model_fallback", "cuda_device": torch.cuda.get_device_name(0),
            "sample_count": len(indices), "represented_studies": int(train.dataset.data.iloc[indices]["patient_id"].nunique()),
            "sample_sha256": sample_digest, "checkpoint_sha256": EXPECTED_CHECKPOINT,
            "labels_sha256": EXPECTED_LABELS, "fold_manifest_sha256": EXPECTED_FOLDS,
            "probe_source_sha256": sha256(Path(__file__)),
            "dataset_source_sha256": sha256(ROOT / "src/strategies/mls_heatmap/dataset.py"),
            "model_loader_source_sha256": sha256(ROOT / "src/strategies/mls_heatmap/predict_multitask.py"),
            "input_representation": "processed_training_png_no_augmentation",
            "softargmax_source_sha256": sha256(ROOT / "src/strategies/mls_heatmap/train.py"),
            "dark_source_sha256": sha256(ROOT / "src/strategies/mls_heatmap/utils.py"),
            "elapsed_seconds": time.monotonic() - started, "validation_images_used": 0,
            "promotion_eligible": False, "submission_zip_allowed": False,
        })
        _atomic_json(args.output, result)
        result["mlflow"] = log_aggregate(args.output, result)
        _atomic_json(args.output, result)
        _atomic_json(status_path, {"state": "completed", "exit_code": 0})
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        _atomic_json(status_path, {"state": "failed", "error_type": type(exc).__name__, "exit_code": 1})
        print(json.dumps({"state": "failed", "error_type": type(exc).__name__}))
        return 1
    finally:
        if acquired:
            lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
