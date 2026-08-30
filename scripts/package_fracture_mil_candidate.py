"""Assemble a checksum-verified five-fold fracture MIL candidate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics.utils.torch_utils import strip_optimizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_verified(source: Path, target: Path, expected: str | None = None) -> str:
    actual = _sha256(source)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Checksum mismatch for {source}: {actual} != {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied = _sha256(target)
    if copied != actual:
        raise RuntimeError(f"Copy verification failed for {target}")
    return actual


def _strip_detector(source: Path, target: Path, expected_source: str) -> tuple[str, str]:
    source_hash = _sha256(source)
    if source_hash != expected_source:
        raise RuntimeError(
            f"Detector source checksum mismatch: {source_hash} != {expected_source}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    strip_optimizer(str(source), str(target))
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"Optimizer stripping did not create {target}")
    return source_hash, _sha256(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--decision-calibration", type=Path, required=True)
    parser.add_argument("--meta-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = _load_json(args.calibration_manifest)
    decision = _load_json(args.decision_calibration)
    if not isinstance(calibration, list) or len(calibration) != 5:
        raise ValueError("Calibration manifest must contain five folds")
    if not isinstance(decision, dict):
        raise ValueError("Decision calibration must be an object")

    fold_payloads: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, object]] = {}
    for fold in range(5):
        cache_manifest = _load_json(args.cache_root / f"fold_{fold}" / "manifest.json")
        if not isinstance(cache_manifest, dict):
            raise ValueError("Cache manifest must be an object")
        detector_source = Path(str(cache_manifest["checkpoint"]))
        detector_relative = Path(f"fold_{fold}") / "detector.pt"
        detector_source_hash, detector_hash = _strip_detector(
            detector_source,
            args.output / detector_relative,
            str(cache_manifest["checkpoint_sha256"]),
        )
        mil_relatives: list[str] = []
        for source in sorted((args.model_root / f"fold_{fold}_v2").glob("model_seed*.pt")):
            relative = Path(f"fold_{fold}") / source.name
            _copy_verified(source, args.output / relative)
            mil_relatives.append(relative.as_posix())
        if len(mil_relatives) != 3:
            raise RuntimeError(f"Fold {fold} does not contain three MIL heads")
        fold_calibration = calibration[fold]
        if not isinstance(fold_calibration, dict) or int(fold_calibration["fold"]) != fold:
            raise ValueError(f"Calibration order mismatch at fold {fold}")
        fold_payloads.append(
            {
                "fold": fold,
                "detector": detector_relative.as_posix(),
                "detector_source_sha256": detector_source_hash,
                "mil_heads": mil_relatives,
                "candidate_weight": float(fold_calibration["candidate_weight"]),
                "reference_training_scores": fold_calibration[
                    "reference_training_scores"
                ],
                "candidate_training_scores": fold_calibration[
                    "candidate_training_scores"
                ],
            }
        )
        artifacts[detector_relative.as_posix()] = {
            "sha256": detector_hash,
            "bytes": (args.output / detector_relative).stat().st_size,
        }
        for relative in mil_relatives:
            path = args.output / relative
            artifacts[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}

    threshold = float(decision["selected_threshold_all_oof"])
    manifest = {
        "schema_version": 1,
        "candidate": "yolov8s-epoch10-5fold-sa-mil-fixed045-v2",
        "status": "promising_pending_runtime_and_end_to_end_validation",
        "source_meta_mlflow_run_id": args.meta_run_id,
        "inference": {
            "image_size": 512,
            "batch_size": args.batch_size,
            "confidence": 0.001,
            "bone_window_width": 1000.0,
            "bone_window_level": 400.0,
            "jpeg_quality": 95,
            "detector_checkpoint": "optimizer_stripped_inference_only",
        },
        "decision_calibration": {
            "threshold": threshold,
            "mapping": decision["threshold_mapping"],
            "source": "five-fold OOF fixed-weight cross-fit evaluation",
        },
        "folds": fold_payloads,
        "artifacts": artifacts,
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    (args.output / "manifest.json").write_text(rendered + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# Fracture detector + SA-MIL candidate\n\n"
        "Five YOLOv8s epoch-10 outer-fold detectors plus three tiny SA-MIL heads "
        "per fold. Detector checkpoints are optimizer-stripped inference artifacts. "
        "Slice scores and embeddings are mapped through train-only "
        "empirical CDFs, blended, averaged across folds, and decision-aligned so "
        f"the official 0.5 cutoff corresponds to OOF score {threshold:.6f}.\n\n"
        "Offline evidence: deployable OOF AUC 0.9078; leakage-controlled decision "
        "F1 0.5484 (precision 0.50, recall 0.607). This is a best-current candidate, "
        "not a leaderboard-certified final model.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate": manifest["candidate"],
                "artifact_count": len(artifacts),
                "artifact_bytes": sum(int(item["bytes"]) for item in artifacts.values()),
                "decision_threshold": threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
