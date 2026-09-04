"""Prove a small A9 trainer optimization is bitwise-equivalent before adoption.

This is a CUDA-only, non-candidate benchmark.  It never creates a model
checkpoint, reads validation data, or chooses a model.  The two permitted
differences are (1) a zero-copy view for the historical input digest and
(2) moving loss-scalar transfer from every step to the end of an epoch.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256
from scripts.train_mls_a7_paired_cuda import move_batch
from scripts.train_mls_a9_frozen_refiner_cuda import (
    BASE,
    BATCH_SIZE,
    EPOCHS,
    STEPS_PER_EPOCH,
    WORK as A9_WORK,
    frozen_forward,
    loader,
    setup,
    verify_frozen,
)
from src.strategies.mls_heatmap.train_multitask import (
    _capture_rng_state,
    multitask_loss,
    seed_training_epoch,
)


OUT = BASE / "a9_speed_equivalence_20260904"
SOURCE_TRAINING_RUN_ID = "bb4a898d61d544c9a450bfcd4ccb4b79"


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _cpu_c_array(tensor: torch.Tensor) -> np.ndarray:
    _require(tensor.device.type == "cpu", "Input digest must be computed before H2D transfer")
    _require(not tensor.requires_grad, "Input digest tensor unexpectedly requires gradients")
    _require(tensor.is_contiguous(), "Input digest tensor is not C-contiguous")
    array = tensor.numpy()
    _require(array.flags.c_contiguous, "NumPy input digest view is not C-contiguous")
    return array


def _update_tensor_legacy(digest: Any, tensor: torch.Tensor) -> None:
    """Exact historical A9 digest behavior, including the temporary bytes copy."""
    digest.update(_cpu_c_array(tensor).tobytes(order="C"))


def _update_tensor_zero_copy(digest: Any, tensor: torch.Tensor) -> None:
    """Hash the same C-order bytes without allocating a temporary bytes object."""
    array = _cpu_c_array(tensor)
    view = memoryview(array).cast("B")
    _require(view.nbytes == array.nbytes, "Zero-copy input digest byte length changed")
    digest.update(view)


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    return value


def _bitwise_equal(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping) or set(first) != set(second):
            return False
        return all(_bitwise_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        if not isinstance(first, type(second)) or len(first) != len(second):
            return False
        return all(_bitwise_equal(left, right) for left, right in zip(first, second))
    return first == second


def _refiner_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return _clone_cpu(model.outer_refinement.state_dict())


def _loss_trace(values: list[float]) -> tuple[str, str, float]:
    _require(len(values) == STEPS_PER_EPOCH, "Loss trace length differs from A9 exposure")
    packed = struct.pack("<" + "d" * len(values), *values)
    mean = float(np.mean(values, dtype=np.float64))
    return hashlib.sha256(packed).hexdigest(), struct.pack("<d", mean).hex(), mean


def _historical_epoch1_digest() -> str:
    history = json.loads((A9_WORK / "candidate/training_history.json").read_text(encoding="utf-8"))
    _require(isinstance(history, list) and history and history[0].get("epoch") == 1, "Missing A9 epoch-1 history")
    digest = history[0].get("input_exposure_sha256")
    _require(isinstance(digest, str) and len(digest) == 64, "A9 historical digest is invalid")
    return digest


def _run_arm(kind: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(kind in {"reference", "optimized"}, "Unknown speed-equivalence arm")
    spec, config, baseline_state, model = setup()
    train_loader = loader(config)
    optimizer = AdamW(model.outer_refinement.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lambda epoch: 0.5 * (1 + math.cos(math.pi * min(epoch / EPOCHS, 1))),
    )
    seed_training_epoch(42, 1)
    legacy = hashlib.sha256()
    values: list[Any] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in train_loader:
        if kind == "reference":
            _update_tensor_legacy(legacy, batch[0])
            _update_tensor_legacy(legacy, batch[3])
        else:
            _update_tensor_zero_copy(legacy, batch[0])
            _update_tensor_zero_copy(legacy, batch[3])
        images, targets, masks, coords, spacing, target, study_mls, _ = move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        refined, selector, _ = frozen_forward(model, images)
        loss, _ = multitask_loss(refined, selector, targets, masks, coords, spacing, target, study_mls, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite speed-equivalence loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.outer_refinement.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if kind == "reference":
            values.append(float(loss.detach()))
        else:
            values.append(loss.detach())
    torch.cuda.synchronize()
    loop_seconds = time.perf_counter() - started
    _require(len(values) == STEPS_PER_EPOCH, "Optimizer exposure changed in speed-equivalence arm")
    scheduler.step()
    verify_frozen(model, baseline_state)
    if kind == "optimized":
        # One post-loop D2H transfer preserves the exact Python float values while
        # removing the per-step scalar synchronization from the timed loop.
        values = [float(item) for item in torch.stack(values).cpu().tolist()]
    losses = [float(value) for value in values]
    loss_trace_sha, mean_hex, train_loss = _loss_trace(losses)
    public = {
        "label": label,
        "kind": kind,
        "optimizer_steps": len(losses),
        "legacy_input_exposure_sha256": legacy.hexdigest(),
        "loss_trace_sha256": loss_trace_sha,
        "train_loss": train_loss,
        "train_loss_float64_hex": mean_hex,
        "loop_seconds": loop_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frozen_baseline_verified": True,
    }
    private = {
        "refiner_state": _refiner_state(model),
        "optimizer_state": _clone_cpu(optimizer.state_dict()),
        "scheduler_state": _clone_cpu(scheduler.state_dict()),
        "rng_state": _clone_cpu(_capture_rng_state()),
    }
    del train_loader, optimizer, scheduler, model, baseline_state
    gc.collect()
    torch.cuda.empty_cache()
    return public, private


def _equivalence(left: tuple[dict[str, Any], dict[str, Any]], right: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, bool]:
    left_public, left_private = left
    right_public, right_private = right
    gates = {
        "optimizer_steps_equal": left_public["optimizer_steps"] == right_public["optimizer_steps"] == STEPS_PER_EPOCH,
        "legacy_input_digest_equal": left_public["legacy_input_exposure_sha256"] == right_public["legacy_input_exposure_sha256"],
        "loss_trace_equal": left_public["loss_trace_sha256"] == right_public["loss_trace_sha256"],
        "train_loss_bitwise_equal": left_public["train_loss_float64_hex"] == right_public["train_loss_float64_hex"],
        "refiner_state_bitwise_equal": _bitwise_equal(left_private["refiner_state"], right_private["refiner_state"]),
        "optimizer_state_bitwise_equal": _bitwise_equal(left_private["optimizer_state"], right_private["optimizer_state"]),
        "scheduler_state_bitwise_equal": _bitwise_equal(left_private["scheduler_state"], right_private["scheduler_state"]),
        "rng_state_bitwise_equal": _bitwise_equal(left_private["rng_state"], right_private["rng_state"]),
    }
    gates["all"] = all(gates.values())
    return gates


def _try_log_mlflow(result: dict[str, Any]) -> dict[str, Any]:
    """Log only public aggregate timing/equivalence metrics; never an artifact."""
    run_id: str | None = None
    try:
        from mlflow.tracking import MlflowClient
        from src.mlops.tracking import configure_tracking_environment

        configure_tracking_environment()
        if not os.getenv("MLFLOW_TRACKING_URI"):
            return {"status": "skipped_missing_remote_tracking"}
        client = MlflowClient()
        source = client.get_run(SOURCE_TRAINING_RUN_ID)
        run_id = client.create_run(source.info.experiment_id, tags={
            "mlflow.runName": "MLS | A9-speed-equivalence | non-candidate",
            "run_type": "speed_equivalence",
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "candidate_model": "false",
            "promotion_eligible": "false",
            "private_predictions_uploaded": "false",
        }).info.run_id
        for key, value in {
            "campaign_id": "mls-deploy-aligned-20260902",
            "experiment_key": "A9-speed-equivalence",
            "fixed_epoch": 1,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "strict_precision": "float32_no_amp_no_tf32",
        }.items():
            client.log_param(run_id, key, value)
        metrics = {
            "reference_median_loop_seconds": result["speed"]["reference_median_loop_seconds"],
            "optimized_median_loop_seconds": result["speed"]["optimized_median_loop_seconds"],
            "speedup_ratio": result["speed"]["speedup_ratio"],
            "adoption_eligible": float(result["adoption_eligible"]),
        }
        metrics.update({f"equivalence_{name}": float(value) for name, value in result["gates"].items()})
        client.log_metrics(run_id, metrics)
        client.set_tag(run_id, "decision", "adopt_only_if_all_equivalent_and_speedup_ge_1_05")
        client.set_tag(run_id, "adoption_eligible", str(bool(result["adoption_eligible"])).lower())
        client.set_terminated(run_id, status="FINISHED")
        return {"status": "finished_metrics_only", "run_id": run_id, "artifacts_uploaded": False}
    except Exception as exc:  # Tracking must never invalidate completed CUDA evidence.
        if run_id is not None:
            try:
                from mlflow.tracking import MlflowClient
                MlflowClient().set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        return {"status": "deferred", "error_type": type(exc).__name__}


def _guard_gpu() -> Path:
    lock = BASE / "gpu_training.lock"
    lock.mkdir()
    try:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if active:
            raise RuntimeError("Concurrent GPU workload")
        if shutil.disk_usage(BASE).free < 10 * 2**30:
            raise RuntimeError("Need 10GiB free disk space")
    except BaseException:
        if lock.exists():
            lock.rmdir()
        raise
    return lock


def _interrupt_as_system_exit(_signum: int, _frame: Any) -> None:
    """Let Supervisor's normal SIGTERM path reach the lock-cleanup finally block."""
    raise SystemExit("Interrupted by Supervisor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-mlflow", action="store_true", help="Do not create the metrics-only MLflow run")
    args = parser.parse_args()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite speed-equivalence evidence: {OUT}")
    lock = _guard_gpu()
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_as_system_exit)
    try:
        OUT.mkdir(parents=True)
        _atomic_json(OUT / "status.json", {
            "status": "running", "compute_policy": "cuda_only_no_cpu_model_fallback",
            "candidate_model": False, "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        })
        expected_digest = _historical_epoch1_digest()
        reference_first = _run_arm("reference", "reference_then_optimized")
        optimized_first = _run_arm("optimized", "optimized_after_reference")
        optimized_second = _run_arm("optimized", "optimized_then_reference")
        reference_second = _run_arm("reference", "reference_after_optimized")
        paired_first = _equivalence(reference_first, optimized_first)
        paired_second = _equivalence(optimized_second, reference_second)
        same_reference = _equivalence(reference_first, reference_second)
        same_optimized = _equivalence(optimized_first, optimized_second)
        public_arms = [
            reference_first[0], optimized_first[0], optimized_second[0], reference_second[0],
        ]
        reference_times = [reference_first[0]["loop_seconds"], reference_second[0]["loop_seconds"]]
        optimized_times = [optimized_first[0]["loop_seconds"], optimized_second[0]["loop_seconds"]]
        reference_median = statistics.median(reference_times)
        optimized_median = statistics.median(optimized_times)
        speedup = reference_median / optimized_median
        gates = {
            "historical_reference_digest_matches": all(
                arm[0]["legacy_input_exposure_sha256"] == expected_digest
                for arm in (reference_first, reference_second)
            ),
            "reference_then_optimized_equivalent": paired_first["all"],
            "optimized_then_reference_equivalent": paired_second["all"],
            "reference_repeatable": same_reference["all"],
            "optimized_repeatable": same_optimized["all"],
            "speedup_ge_1_05": speedup >= 1.05,
        }
        result: dict[str, Any] = {
            "status": "completed",
            "schema_version": 1,
            "purpose": "non_candidate_exact_equivalence_speed_benchmark",
            "compute_policy": "cuda_only_no_cpu_model_fallback",
            "candidate_model": False,
            "validation_images_used": 0,
            "private_predictions_uploaded": False,
            "checkpoints_written": False,
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "source_sha256": {
                "benchmark": _sha256(Path(__file__)),
                "a9_trainer": _sha256(ROOT / "scripts/train_mls_a9_frozen_refiner_cuda.py"),
                "a9_manifest": _sha256(ROOT / "reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_TRAINING_PROTOCOL_20260904.json"),
            },
            "contract": {
                "seed": 42, "fold": 0, "epoch": 1, "batch_size": BATCH_SIZE,
                "steps_per_epoch": STEPS_PER_EPOCH, "precision": "float32_no_amp_no_tf32",
                "allowed_changes": ["zero_copy_input_digest", "post_epoch_loss_scalar_transfer"],
                "equivalence_evidence": "historical image+coordinate digest plus exact loss trace, refiner state, optimizer state, scheduler state, and RNG state",
            },
            "historical_epoch1_input_exposure_sha256": expected_digest,
            "arms": public_arms,
            "pairwise_equivalence": {
                "reference_then_optimized": paired_first,
                "optimized_then_reference": paired_second,
                "reference_repeatability": same_reference,
                "optimized_repeatability": same_optimized,
            },
            "speed": {
                "reference_loop_seconds": reference_times,
                "optimized_loop_seconds": optimized_times,
                "reference_median_loop_seconds": reference_median,
                "optimized_median_loop_seconds": optimized_median,
                "speedup_ratio": speedup,
            },
            "gates": gates,
            "adoption_eligible": all(gates.values()),
            "adoption_rule": "All equivalence gates and speedup_ratio >= 1.05 are required; otherwise keep the historical trainer unchanged.",
        }
        # The local, checksum-addressable evidence must survive a tracker outage
        # or stall.  Release the GPU lock before the best-effort metrics-only
        # projection so no later model work is delayed by MLflow I/O.
        result["mlflow"] = {"status": "pending_best_effort"}
        _atomic_json(OUT / "result.json", result)
        _atomic_json(OUT / "status.json", {
            "status": "compute_completed_tracking_pending", "adoption_eligible": result["adoption_eligible"],
            "mlflow": result["mlflow"], "candidate_model": False,
        })
        if lock.exists():
            lock.rmdir()
        result["mlflow"] = {"status": "not_requested"} if args.no_mlflow else _try_log_mlflow(result)
        _atomic_json(OUT / "result.json", result)
        _atomic_json(OUT / "status.json", {
            "status": "completed", "adoption_eligible": result["adoption_eligible"],
            "mlflow": result["mlflow"], "candidate_model": False,
        })
        print(json.dumps({
            "status": result["status"], "adoption_eligible": result["adoption_eligible"],
            "speedup_ratio": result["speed"]["speedup_ratio"], "mlflow": result["mlflow"],
        }, sort_keys=True))
        return 0
    except Exception as exc:
        if OUT.exists():
            _atomic_json(OUT / "status.json", {
                "status": "failed", "error_type": type(exc).__name__,
                "candidate_model": False, "compute_policy": "cuda_only_no_cpu_model_fallback",
            })
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if lock.exists():
            lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
