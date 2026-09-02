"""CUDA-only smoke test for an extracted submission package.

The package is imported from the supplied directory rather than from the
working tree, so this validates the exact archived runtime and model files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import torch


OUTPUT_KEYS = (
    "V_EDH",
    "V_SDH",
    "V_IPH",
    "V_SAH",
    "V_IVH",
    "fracture_prob",
    "MLS_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-mls", type=float)
    parser.add_argument("--mls-atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission_root = args.submission_root.resolve()
    study_dir = args.study_dir.resolve()
    if not (submission_root / "model.py").is_file():
        raise FileNotFoundError(submission_root / "model.py")
    if not study_dir.is_dir():
        raise FileNotFoundError(study_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model fallback is forbidden")

    sys.path.insert(0, str(submission_root))
    runtime = importlib.import_module("model")
    if Path(runtime.__file__).resolve() != submission_root / "model.py":
        raise RuntimeError(f"Imported the wrong package runtime: {runtime.__file__}")

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    models = runtime.load_models(submission_root / "models", "cuda")
    torch.cuda.synchronize()
    load_runtime_s = time.perf_counter() - load_started
    if torch.device(models["device"]).type != "cuda":
        raise RuntimeError("Package did not retain the requested CUDA device")

    inference_started = time.perf_counter()
    with torch.inference_mode():
        prediction = runtime.predict(study_dir, models)
    torch.cuda.synchronize()
    inference_runtime_s = time.perf_counter() - inference_started

    if tuple(prediction) != OUTPUT_KEYS:
        raise RuntimeError(f"Output schema mismatch: {tuple(prediction)}")
    rendered_prediction = {key: float(prediction[key]) for key in OUTPUT_KEYS}
    if not all(math.isfinite(value) for value in rendered_prediction.values()):
        raise FloatingPointError("Package returned a non-finite prediction")

    parity: dict[str, float | bool | None] = {
        "expected_mls_mm": args.expected_mls,
        "absolute_delta_mm": None,
        "atol_mm": args.mls_atol,
        "passed": args.expected_mls is None,
    }
    if args.expected_mls is not None:
        delta = abs(rendered_prediction["MLS_mm"] - args.expected_mls)
        parity.update({"absolute_delta_mm": delta, "passed": delta <= args.mls_atol})
        if delta > args.mls_atol:
            raise RuntimeError(
                f"Packaged MLS parity failed: delta={delta} > atol={args.mls_atol}"
            )

    payload = {
        "schema_version": 1,
        "state": "completed",
        "compute_policy": "cuda_only_model_forward",
        "cuda_device": torch.cuda.get_device_name(0),
        "submission_root": str(submission_root),
        "runtime_module": str(Path(runtime.__file__).resolve()),
        "study_dir": str(study_dir),
        "load_runtime_s": load_runtime_s,
        "inference_runtime_s": inference_runtime_s,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "mls_models": len(models["mls"].models),
        "fracture_folds": len(models["fracture"].folds),
        "prediction": rendered_prediction,
        "mls_reference_parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
