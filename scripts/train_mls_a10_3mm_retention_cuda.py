"""A10: frozen A9-style refiner with a training-only 3-mm retention guard.

The qualified fold0 baseline remains bitwise frozen.  A10 adds only a loss on
the refiner output: when that frozen coarse baseline already classifies a valid
training slice correctly at 3 mm *and* the true slice label is at least 0.25 mm
from that boundary, the refined output is penalized for not staying on the
correct side of a fixed 0.25-mm safety margin.  This is a slice-level training
proxy for a study-level observation, not a deployment-time rule.  Inference,
pooling, thresholds, selector, decoder, batch order, precision, and checkpoint
epoch are unchanged.

All model work is CUDA-only.  The experiment is exploratory because fold0's
resource screen informed this hypothesis; no output is promotion eligible.
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
import struct
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256
from scripts.train_mls_a7_paired_cuda import move_batch
from scripts.train_mls_a9_frozen_refiner_cuda import (
    BASE,
    BASELINE,
    BATCH_SIZE,
    EPOCHS,
    STEPS_PER_EPOCH,
    frozen_forward,
    loader,
    setup as a9_setup,
    verify_frozen,
)
from src.mlops.tracking import configure_tracking_environment
from src.strategies.mls_heatmap.train_multitask import (
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
    decode_training_keypoints,
    differentiable_mls_mm,
    multitask_loss,
    seed_training_epoch,
)


WORK = BASE / "a10_frozen_baseline_3mm_retention_20260905"
MANIFEST = ROOT / "reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A10_TRAINING_PROTOCOL_20260905.json"
EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256 = "c27b932c23330b154b63eef8a9cba0d52a282b1760dbf536cafeee55b964352b"
SPEED_EVIDENCE_SHA256 = "3c4e4f4957c10415eeed681f5fb7ab80ff8f9af3f7671ea5cb24b5f44bbcc508"
SOURCE_TRAINING_RUN_ID = "bb4a898d61d544c9a450bfcd4ccb4b79"
THREE_MM = 3.0
RETENTION_MARGIN_MM = 0.25
RETENTION_WEIGHT = 0.10
TRAINABLE_PARAMETERS = 47617


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_protocol() -> dict[str, Any]:
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "experiment": "A10_frozen_qualified_baseline_3mm_retention",
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError("A10 training protocol semantic contract differs")
    sources = spec.get("source_and_input_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("A10 training protocol has no pinned sources")
    for relative, digest in sources.items():
        path = ROOT / str(relative)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError("Pinned A10 source/input changed: " + str(relative))
    if _sha256(BASELINE) != spec.get("baseline_checkpoint_sha256"):
        raise ValueError("Qualified baseline checkpoint changed for A10")
    if spec.get("expected_a9_epoch1_input_exposure_sha256") != EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256:
        raise ValueError("A10 expected A9 input-exposure digest differs")
    if spec.get("runtime_speed_evidence_sha256") != SPEED_EVIDENCE_SHA256:
        raise ValueError("A10 runtime-speed evidence identity differs")
    return spec


def _receipt_provenance(spec: Mapping[str, Any]) -> dict[str, str]:
    """Bind CUDA receipts and MLflow lineage to this exact A10 implementation."""
    return {
        "training_manifest_sha256": _sha256(MANIFEST),
        "baseline_checkpoint_sha256": str(spec["baseline_checkpoint_sha256"]),
        "trainer_source_sha256": _sha256(Path(__file__).resolve()),
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
    }


def _require_receipt_provenance(
    receipt: Mapping[str, Any], spec: Mapping[str, Any], receipt_name: str,
) -> None:
    expected = _receipt_provenance(spec)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(receipt_name + " provenance does not match the current A10 contract")


def setup_a10() -> tuple[dict[str, Any], Any, dict[str, torch.Tensor], torch.nn.Module]:
    """Pin A10 first, then reuse the already pinned exact A9 baseline setup."""
    spec = _load_protocol()
    _a9_spec, config, baseline_state, model = a9_setup()
    if not math.isclose(float(config.threshold_temperature_mm), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("A10 requires the historical 0.5-mm threshold temperature")
    if not math.isclose(float(config.threshold_loss_weight), RETENTION_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("A10 retention coefficient must equal historical threshold-loss weight")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != TRAINABLE_PARAMETERS:
        raise ValueError("A10 changed the frozen-refiner parameter contract")
    return spec, config, baseline_state, model


def _retention_3mm_loss(
    refined: torch.Tensor,
    coarse: torch.Tensor,
    masks: torch.Tensor,
    keypoints_true: torch.Tensor,
    spacing_x: torch.Tensor,
    is_target: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep an unambiguous correct frozen-teacher 3-mm decision after refinement.

    The teacher branch is explicitly detached/no-grad.  The loss is training
    only and uses the same differentiable geometric decoder as ``multitask_loss``
    for the refined branch; no deployment-time score or threshold is altered.
    """
    if coarse.requires_grad:
        raise ValueError("A10 coarse teacher must be gradient-free")
    valid = (is_target > 0.5) & (masks > 0.5).all(dim=1)
    zero = refined.new_zeros(())
    valid_count = valid.sum().detach()
    if not bool(valid.any()):
        return zero, {"valid_count": valid_count, "qualified_count": zero.detach()}

    refined_keypoints = decode_training_keypoints(refined[valid], config)
    refined_mls = differentiable_mls_mm(refined_keypoints, spacing_x[valid])
    with torch.no_grad():
        coarse_keypoints = decode_training_keypoints(coarse[valid], config)
        coarse_mls = differentiable_mls_mm(coarse_keypoints, spacing_x[valid])
        true_mls = differentiable_mls_mm(keypoints_true[valid], spacing_x[valid])
        target_positive = true_mls >= THREE_MM
        teacher_correct = (coarse_mls >= THREE_MM) == target_positive
        # Do not force a label just above/below 3 mm farther away merely because
        # it is technically on one side.  Such slices are not an unambiguous
        # training proxy for the deployed study-level threshold decision.
        target_has_margin = torch.abs(true_mls - THREE_MM) >= RETENTION_MARGIN_MM
        qualified_teacher = teacher_correct & target_has_margin
        direction = torch.where(target_positive, torch.ones_like(refined_mls), -torch.ones_like(refined_mls))
    qualified_count = qualified_teacher.sum().detach()
    signed_distance = direction * (refined_mls - THREE_MM)
    violation = F.relu(RETENTION_MARGIN_MM - signed_distance)
    # A clamp gives an exact zero contribution (and zero gradient) when a rare
    # batch has no correct teacher decision, without a second host-side sync.
    qualified = qualified_teacher.to(refined.dtype)
    return (violation * qualified).sum() / qualified.sum().clamp_min(1.0), {
        "valid_count": valid_count,
        "qualified_count": qualified_count,
    }


def a10_loss(
    refined: torch.Tensor,
    selector: torch.Tensor,
    coarse: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    coords: torch.Tensor,
    spacing: torch.Tensor,
    is_target: torch.Tensor,
    study_mls: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base_loss, parts = multitask_loss(
        refined, selector, targets, masks, coords, spacing, is_target, study_mls, config,
    )
    retention, counts = _retention_3mm_loss(
        refined, coarse, masks, coords, spacing, is_target, config,
    )
    result = dict(parts)
    result["three_mm_retention"] = retention.detach()
    result["three_mm_retention_valid_count"] = counts["valid_count"]
    result["three_mm_retention_qualified_count"] = counts["qualified_count"]
    return base_loss + RETENTION_WEIGHT * retention, result


def _digest_zero_copy(digest: Any, tensor: torch.Tensor) -> None:
    """Exact ``numpy().tobytes`` bytes without the historical full-array copy."""
    if tensor.device.type != "cpu" or tensor.requires_grad or not tensor.is_contiguous():
        raise RuntimeError("A10 zero-copy digest contract is not satisfied")
    array = tensor.numpy()
    if not array.flags.c_contiguous or array.__array_interface__["data"][0] != tensor.data_ptr():
        raise RuntimeError("A10 zero-copy digest cannot prove byte identity")
    digest.update(memoryview(array).cast("B"))


def _digest_legacy(digest: Any, tensor: torch.Tensor) -> None:
    digest.update(tensor.numpy().tobytes())


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
        return (
            isinstance(first, Mapping) and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_bitwise_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        return (
            isinstance(first, type(second)) and len(first) == len(second)
            and all(_bitwise_equal(left, right) for left, right in zip(first, second))
        )
    return first == second


def _loss_trace(values: list[float]) -> tuple[str, str, float]:
    if len(values) != STEPS_PER_EPOCH:
        raise ValueError("A10 loss trace has the wrong optimizer exposure")
    packed = struct.pack("<" + "d" * len(values), *values)
    mean = float(np.mean(values, dtype=np.float64))
    return hashlib.sha256(packed).hexdigest(), struct.pack("<d", mean).hex(), mean


def _require_preflight() -> dict[str, Any]:
    spec = _load_protocol()
    path = WORK / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError("Successful A10 CUDA preflight is required")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "baseline_identity_at_initialization": True,
        "frozen_baseline_unchanged_after_step": True,
        "refiner_updated": True,
        "three_mm_retention_active": True,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "batch_size": BATCH_SIZE,
        "cuda_only": True,
        "validation_images_used": 0,
        "promotion_eligible": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("A10 CUDA preflight semantic contract differs")
    _require_receipt_provenance(receipt, spec, "A10 CUDA preflight")
    return receipt


def _require_equivalence_preflight() -> dict[str, Any]:
    spec = _load_protocol()
    path = WORK / "speed_equivalence_preflight.json"
    if not path.is_file():
        raise FileNotFoundError("Successful A10 exact-state speed-equivalence preflight is required")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or receipt.get("all_gates_passed") is not True:
        raise ValueError("A10 speed-equivalence preflight did not pass")
    _require_receipt_provenance(receipt, spec, "A10 speed-equivalence preflight")
    return receipt


def preflight() -> None:
    if WORK.exists():
        raise FileExistsError("A10 work directory already exists")
    spec, config, baseline_state, model = setup_a10()
    train_loader = loader(config)
    batch = move_batch(next(iter(train_loader)))
    images, targets, masks, coords, spacing, is_target, study_mls, _ = batch
    with torch.no_grad():
        model.eval()
        features = model.backbone(images)[0]
        coarse = model.head(features)
        initialized = model.outer_refinement(features, coarse)
    if not torch.equal(initialized, coarse):
        raise ValueError("A10 zero-initialized refiner does not preserve baseline")
    optimizer = AdamW(model.outer_refinement.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    refined, selector, coarse = frozen_forward(model, images)
    loss, parts = a10_loss(
        refined, selector, coarse, targets, masks, coords, spacing, is_target, study_mls, config,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite A10 preflight loss")
    if int(parts["three_mm_retention_qualified_count"].item()) < 1:
        raise ValueError("A10 3-mm retention guard is inactive on preflight batch")
    before = _clone_cpu(model.outer_refinement.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.outer_refinement.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    if not any(not torch.equal(before[key], value.detach().cpu()) for key, value in model.outer_refinement.state_dict().items()):
        raise ValueError("A10 refiner did not update")
    verify_frozen(model, baseline_state)
    WORK.mkdir()
    result = {
        "status": "completed",
        "baseline_identity_at_initialization": True,
        "frozen_baseline_unchanged_after_step": True,
        "refiner_updated": True,
        "three_mm_retention_active": True,
        "three_mm_retention_margin_mm": RETENTION_MARGIN_MM,
        "three_mm_retention_weight": RETENTION_WEIGHT,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "batch_size": BATCH_SIZE,
        "cuda_only": True,
        "validation_images_used": 0,
        **_receipt_provenance(spec),
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    _atomic_json(WORK / "preflight.json", result)
    print(json.dumps(result, sort_keys=True))


def _run_equivalence_arm(kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind not in {"reference", "optimized"}:
        raise ValueError("Unknown A10 equivalence arm")
    _spec, config, baseline_state, model = setup_a10()
    train_loader = loader(config)
    optimizer = AdamW(model.outer_refinement.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.5 * (1.0 + math.cos(math.pi * min(epoch / EPOCHS, 1.0))))
    seed_training_epoch(42, 1)
    digest = hashlib.sha256()
    values: list[Any] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in train_loader:
        if kind == "reference":
            _digest_legacy(digest, batch[0])
            _digest_legacy(digest, batch[3])
        else:
            _digest_zero_copy(digest, batch[0])
            _digest_zero_copy(digest, batch[3])
        images, targets, masks, coords, spacing, is_target, study_mls, _ = move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        refined, selector, coarse = frozen_forward(model, images)
        loss, _parts = a10_loss(
            refined, selector, coarse, targets, masks, coords, spacing, is_target, study_mls, config,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite A10 equivalence loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.outer_refinement.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        values.append(float(loss.detach()) if kind == "reference" else loss.detach())
    torch.cuda.synchronize()
    loop_seconds = time.perf_counter() - started
    if len(values) != STEPS_PER_EPOCH or digest.hexdigest() != EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256:
        raise ValueError("A10 equivalence exposure does not match qualified A9 epoch1")
    scheduler.step()
    verify_frozen(model, baseline_state)
    if kind == "optimized":
        values = [float(value) for value in torch.stack(values).cpu().tolist()]
    losses = [float(value) for value in values]
    trace_sha, mean_hex, train_loss = _loss_trace(losses)
    public = {
        "kind": kind,
        "optimizer_steps": len(losses),
        "input_exposure_sha256": digest.hexdigest(),
        "loss_trace_sha256": trace_sha,
        "train_loss": train_loss,
        "train_loss_float64_hex": mean_hex,
        "loop_seconds": loop_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frozen_baseline_verified": True,
    }
    private = {
        "refiner_state": _clone_cpu(model.outer_refinement.state_dict()),
        "optimizer_state": _clone_cpu(optimizer.state_dict()),
        "scheduler_state": _clone_cpu(scheduler.state_dict()),
        "rng_state": _clone_cpu(_capture_rng_state()),
    }
    del train_loader, optimizer, scheduler, model, baseline_state
    gc.collect()
    torch.cuda.empty_cache()
    return public, private


def equivalence_preflight() -> None:
    _require_preflight()
    spec = _load_protocol()
    output = WORK / "speed_equivalence_preflight.json"
    if output.exists():
        raise FileExistsError("A10 equivalence preflight already has a receipt")
    reference = _run_equivalence_arm("reference")
    optimized = _run_equivalence_arm("optimized")
    reference_public, reference_private = reference
    optimized_public, optimized_private = optimized
    gates = {
        "optimizer_steps_equal": reference_public["optimizer_steps"] == optimized_public["optimizer_steps"] == STEPS_PER_EPOCH,
        "input_exposure_equal": reference_public["input_exposure_sha256"] == optimized_public["input_exposure_sha256"] == EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256,
        "loss_trace_equal": reference_public["loss_trace_sha256"] == optimized_public["loss_trace_sha256"],
        "train_loss_bitwise_equal": reference_public["train_loss_float64_hex"] == optimized_public["train_loss_float64_hex"],
        "refiner_state_bitwise_equal": _bitwise_equal(reference_private["refiner_state"], optimized_private["refiner_state"]),
        "optimizer_state_bitwise_equal": _bitwise_equal(reference_private["optimizer_state"], optimized_private["optimizer_state"]),
        "scheduler_state_bitwise_equal": _bitwise_equal(reference_private["scheduler_state"], optimized_private["scheduler_state"]),
        "rng_state_bitwise_equal": _bitwise_equal(reference_private["rng_state"], optimized_private["rng_state"]),
        "frozen_baseline_verified": reference_public["frozen_baseline_verified"] and optimized_public["frozen_baseline_verified"],
    }
    all_passed = bool(all(gates.values()))
    result = {
        "status": "completed" if all_passed else "failed_equivalence",
        "scope": "a10_single_epoch_reference_vs_optimized_exact_state_preflight",
        "model_training_candidate_created": False,
        "validation_images_used": 0,
        "private_state_exported": False,
        "a9_four_arm_speed_evidence_sha256": SPEED_EVIDENCE_SHA256,
        "role": "A10 fidelity gate only; four-arm A9 result remains the speed measurement evidence",
        "reference": reference_public,
        "optimized": optimized_public,
        "gates": gates,
        "all_gates_passed": all_passed,
        **_receipt_provenance(spec),
        "promotion_eligible": False,
    }
    _atomic_json(output, result)
    print(json.dumps({"status": result["status"], "all_gates_passed": all_passed}, sort_keys=True))
    if not all_passed:
        raise RuntimeError("A10 optimized training path is not bitwise equivalent")


def _run_card(
    out: Path,
    *,
    status: str,
    tracking_status: str,
    mlflow_run_id: str | None,
    decision: str,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": "mls-deploy-aligned-20260902",
        "experiment_key": "A10",
        "run_type": "candidate_train_and_exploratory_resource_screen",
        "candidate_model": True,
        "candidate_status": status,
        "stage": "training" if status == "running" else "training_complete_pending_exploratory_resource_screen",
        "decision": decision,
        "promotion_eligible": False,
        "submission_zip_allowed": False,
        "fold0_evaluation_role": "exploratory_hypothesis_check_only",
        "hypothesis": "Training-only, margin-enforcing correct-teacher 3-mm retention guard on a frozen qualified baseline refiner; a slice-level proxy for a study-level observation.",
        "three_mm_retention": {
            "threshold_mm": THREE_MM,
            "margin_mm": RETENTION_MARGIN_MM,
            "weight": RETENTION_WEIGHT,
            "qualification": "frozen teacher correct and true slice MLS is at least margin_mm from 3 mm",
            "inference_change": False,
            "study_level_claim": "none; fold0 resource screen is exploratory only",
        },
        "runtime_optimization_evidence_sha256": SPEED_EVIDENCE_SHA256,
        "mlflow_run_id": mlflow_run_id,
        "tracking_status": tracking_status,
        "checkpoint_uploaded_to_mlflow": False,
        "private_predictions_uploaded": False,
        "validation_images_used": 0,
    }
    if history:
        payload["epochs_completed"] = len(history)
        payload["last_epoch"] = history[-1]
    path = out / "MLFLOW_RUN_CARD.json"
    _atomic_json(path, payload)
    return path


def _tracking_lineage_tags(spec: Mapping[str, Any]) -> dict[str, str]:
    """Immutable identity that must survive lifecycle transitions/resume."""
    return {
        "campaign_id": "mls-deploy-aligned-20260902",
        "experiment_key": "A10",
        "candidate_model": "true",
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        **_receipt_provenance(spec),
    }


def _tracking_tags(spec: Mapping[str, Any]) -> dict[str, str]:
    """Human-readable MLflow state plus immutable candidate lineage."""
    return {
        "mlflow.runName": "MLS | A10 | candidate | F0/S42 | RUNNING",
        **_tracking_lineage_tags(spec),
        "run_type": "candidate_train_and_exploratory_resource_screen",
        "candidate_status": "experimental",
        "stage": "training",
        "decision": "pending_exploratory_resource_gate",
        "promotion_eligible": "false",
        "submission_zip_allowed": "false",
        "private_predictions_uploaded": "false",
        "fold0_evaluation_role": "exploratory_hypothesis_check_only",
        "evidence_status": "local_protocol_verified",
        "tracking_lifecycle": "running",
    }


def _mark_tracking_setup_failed(client: Any, run_id: str, error_type: str) -> None:
    """Do not leave a partially-created tracking run falsely marked RUNNING."""
    try:
        client.set_tag(run_id, "stage", "tracking_setup_failed")
        client.set_tag(run_id, "candidate_status", "technical_failure")
        client.set_tag(run_id, "decision", "no_scientific_decision_due_to_tracking_setup_failure")
        client.set_tag(run_id, "tracking_setup_error_type", error_type)
        client.set_terminated(run_id, status="FAILED")
    except Exception:
        pass


def _open_tracking(
    existing_run_id: str | None, spec: Mapping[str, Any], history: list[dict[str, Any]],
) -> tuple[Any | None, str | None, str]:
    """Create/validate one readable MLflow run; CUDA evidence never depends on I/O.

    If tracking becomes available only after a local recovery, its existing
    durable history is backfilled at the original epoch steps, so the MLflow UI
    is not a misleading partial view of the finished candidate.
    """
    client: Any | None = None
    run_id = existing_run_id
    created_here = False
    try:
        from mlflow.tracking import MlflowClient

        configure_tracking_environment()
        if not os.getenv("MLFLOW_TRACKING_URI"):
            return None, run_id, "skipped_missing_remote_tracking"
        client = MlflowClient()
        tags = _tracking_tags(spec)
        params: dict[str, str | int | float] = {
            "baseline_checkpoint_sha256": tags["baseline_checkpoint_sha256"],
            "training_manifest_sha256": tags["training_manifest_sha256"],
            "trainer_source_sha256": tags["trainer_source_sha256"],
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "fixed_epoch": EPOCHS,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "three_mm_retention_margin_mm": RETENTION_MARGIN_MM,
            "three_mm_retention_weight": RETENTION_WEIGHT,
            "runtime_optimization_evidence_sha256": SPEED_EVIDENCE_SHA256,
        }
        if run_id is not None:
            observed = client.get_run(run_id)
            lineage = _tracking_lineage_tags(spec)
            if any(observed.data.tags.get(key) != value for key, value in lineage.items()):
                raise ValueError("A10 recovery points to a different MLflow lineage")
            if observed.info.status != "RUNNING":
                return None, run_id, "remote_tracking_prior_terminal_" + str(observed.info.status).lower()
            client.set_tag(run_id, "mlflow.runName", "MLS | A10 | candidate | F0/S42 | RUNNING")
            client.set_tag(run_id, "stage", "training")
            client.set_tag(run_id, "candidate_status", "experimental")
            client.set_tag(run_id, "tracking_lifecycle", "running")
        else:
            experiment_id = client.get_run(SOURCE_TRAINING_RUN_ID).info.experiment_id
            run_id = client.create_run(experiment_id, tags=tags).info.run_id
            created_here = True
            for key, value in params.items():
                client.log_param(run_id, key, value)
            observed = client.get_run(run_id)
            tags_ok = all(observed.data.tags.get(key) == value for key, value in tags.items())
            params_ok = all(observed.data.params.get(key) == str(value) for key, value in params.items())
            if not tags_ok or not params_ok:
                raise ValueError("A10 MLflow tag/parameter readback failed")

        if history:
            _reconcile_tracking_history(client, run_id, history)
            client.set_tag(run_id, "tracking_backfilled_from_local_history", "true")
            client.set_tag(run_id, "tracking_backfilled_epochs", str(len(history)))
        return client, run_id, "remote_tracking_tags_verified"
    except Exception as exc:  # Tracking must not invalidate durable CUDA evidence.
        if client is not None and run_id is not None:
            # Both a just-created run and a validated A10 recovery run belong to
            # this candidate; terminate it instead of leaving an orphan RUNNING.
            _mark_tracking_setup_failed(client, run_id, type(exc).__name__)
        return None, run_id, "remote_tracking_unavailable_" + type(exc).__name__


EPOCH_METRIC_KEYS = (
    "train_epoch_loss", "train_epoch_seconds", "train_peak_vram_gib",
    "train_three_mm_retention_loss", "train_three_mm_retention_qualified_slices",
    "train_three_mm_retention_valid_slices",
)


def _log_epoch_metrics(client: Any | None, run_id: str | None, row: dict[str, Any]) -> str | None:
    if client is None or run_id is None:
        return None
    try:
        for key in EPOCH_METRIC_KEYS:
            client.log_metric(run_id, key, float(row[key]), step=int(row["epoch"]))
        return None
    except Exception as exc:
        return type(exc).__name__


def _reconcile_tracking_history(client: Any, run_id: str, history: list[dict[str, Any]]) -> None:
    """Make MLflow's visible epoch series match durable local history exactly.

    A transient failure can happen after only some of an epoch's six metrics
    were accepted.  Missing values are backfilled at the original step; an
    existing different value is an immutable provenance conflict, not something
    to overwrite silently.
    """
    for row in history:
        step = int(row["epoch"])
        for key in EPOCH_METRIC_KEYS:
            target = float(row[key])
            observed = [
                float(metric.value) for metric in client.get_metric_history(run_id, key)
                if int(metric.step) == step
            ]
            if observed and not all(math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12) for value in observed):
                raise ValueError("A10 MLflow metric provenance conflict: " + key)
            if not observed:
                client.log_metric(run_id, key, target, step=step)
    for row in history:
        step = int(row["epoch"])
        for key in EPOCH_METRIC_KEYS:
            target = float(row[key])
            observed = [
                float(metric.value) for metric in client.get_metric_history(run_id, key)
                if int(metric.step) == step
            ]
            if not observed or not all(math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12) for value in observed):
                raise ValueError("A10 MLflow metric-history readback failed: " + key)


def _finish_tracking(
    client: Any | None,
    run_id: str | None,
    out: Path,
    history: list[dict[str, Any]],
) -> str:
    if client is None or run_id is None:
        return "local_evidence_only"
    try:
        _reconcile_tracking_history(client, run_id, history)
        tags = {
            "mlflow.runName": "MLS | A10 | candidate | F0/S42 | TRAINED_PENDING_SCREEN",
            "stage": "training_complete_pending_exploratory_resource_screen",
            "candidate_status": "awaiting_exploratory_resource_screen",
            "decision": "pending_exploratory_resource_gate",
            "promotion_eligible": "false",
            "submission_zip_allowed": "false",
            "evidence_status": "completion_readback_pending",
            "tracking_lifecycle": "finalizing",
            "metrics_backfill_status": "verified_against_local_history",
        }
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        client.log_metric(run_id, "training_epochs_completed", float(len(history)))
        client.log_metric(run_id, "decision_promotion_eligible", 0.0)
        client.log_metric(run_id, "decision_submission_zip_allowed", 0.0)
        artifact_status = "uploaded_pretermination"
        for name in ("training_summary.json", "training_history.json", "MLFLOW_RUN_CARD.json"):
            try:
                client.log_artifact(run_id, str(out / name), "public_reports")
            except Exception:
                artifact_status = "best_effort_upload_failed"
                break
        client.set_tag(run_id, "artifact_upload_status", artifact_status)
        client.set_tag(run_id, "evidence_status", "metrics_history_readback_verified")
        client.set_tag(run_id, "tracking_lifecycle", "finalized")
        client.set_terminated(run_id, status="FINISHED")
        verified = client.get_run(run_id)
        expected_tags = {
            **tags,
            "artifact_upload_status": artifact_status,
            "evidence_status": "metrics_history_readback_verified",
            "tracking_lifecycle": "finalized",
        }
        tags_ok = all(verified.data.tags.get(key) == value for key, value in expected_tags.items())
        if verified.info.status != "FINISHED" or not tags_ok:
            raise ValueError("A10 MLflow completion readback failed")
        return "metrics_history_and_terminal_readback_verified"
    except Exception as exc:
        _fail_tracking(client, run_id, type(exc).__name__)
        return "tracking_completion_unverified_" + type(exc).__name__


def _fail_tracking(client: Any | None, run_id: str | None, error_type: str) -> None:
    """Best-effort terminal state so a failed CUDA job is never mislabeled success."""
    if client is None or run_id is None:
        return
    try:
        if client.get_run(run_id).info.status != "RUNNING":
            return
        client.set_tag(run_id, "stage", "failed")
        client.set_tag(run_id, "candidate_status", "technical_failure")
        client.set_tag(run_id, "decision", "no_scientific_decision_due_to_technical_failure")
        client.set_tag(run_id, "failure_type", error_type)
        client.set_tag(run_id, "tracking_lifecycle", "failed")
        client.set_terminated(run_id, status="FAILED")
    except Exception:
        pass


def _mark_tracking_interrupted(client: Any | None, run_id: str | None, reason: str) -> None:
    """Keep a Supervisor-interrupted run resumable rather than falsely failed."""
    if client is None or run_id is None:
        return
    try:
        if client.get_run(run_id).info.status != "RUNNING":
            return
        client.set_tag(run_id, "stage", "interrupted_recoverable")
        client.set_tag(run_id, "candidate_status", "interrupted_recoverable")
        client.set_tag(run_id, "tracking_lifecycle", "interrupted_recoverable")
        client.set_tag(run_id, "interruption_reason", reason)
    except Exception:
        pass


def _validate_resume_history(history: Any, first_epoch: int) -> list[dict[str, Any]]:
    if not isinstance(history, list) or not all(isinstance(row, dict) for row in history):
        raise ValueError("A10 recovery history has the wrong schema")
    expected_epochs = list(range(1, first_epoch))
    if [row.get("epoch") for row in history] != expected_epochs:
        raise ValueError("A10 recovery history epochs are not a contiguous prefix")
    if any(row.get("optimizer_steps") != STEPS_PER_EPOCH for row in history):
        raise ValueError("A10 recovery history optimizer exposure differs")
    if history and history[0].get("input_exposure_sha256") != EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256:
        raise ValueError("A10 recovery epoch1 input exposure differs from A9")
    if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(float(row[key])) for row in history for key in EPOCH_METRIC_KEYS):
        raise ValueError("A10 recovery history contains nonfinite MLflow metrics")
    return history


def _recovery_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: Any,
    history: list[dict[str, Any]],
    spec: Mapping[str, Any],
    mlflow_run_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 10,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": _capture_rng_state(),
        "config": config.model_dump(),
        "history": history,
        **_receipt_provenance(spec),
        "retention_margin_mm": RETENTION_MARGIN_MM,
        "retention_weight": RETENTION_WEIGHT,
        "mlflow_run_id": mlflow_run_id,
        "checkpoint_selection": "fixed_epoch10_no_validation_selection",
    }


def train(resume: bool) -> None:
    _require_preflight()
    _require_equivalence_preflight()
    spec, config, baseline_state, model = setup_a10()
    out = WORK / "candidate"
    if out.exists() and not resume:
        raise FileExistsError("No implicit A10 overwrite or resume")
    if resume and not (out / "recovery.pth").is_file():
        raise FileNotFoundError("A10 explicit recovery checkpoint is missing")
    out.mkdir(exist_ok=resume)
    train_loader = loader(config)
    optimizer = AdamW(model.outer_refinement.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.5 * (1.0 + math.cos(math.pi * min(epoch / EPOCHS, 1.0))))
    history: list[dict[str, Any]] = []
    first_epoch = 1
    existing_run_id: str | None = None
    if resume:
        state = torch.load(out / "recovery.pth", map_location="cpu", weights_only=False)
        required = {
            "training_manifest_sha256": _sha256(MANIFEST),
            "baseline_checkpoint_sha256": spec["baseline_checkpoint_sha256"],
            "trainer_source_sha256": _receipt_provenance(spec)["trainer_source_sha256"],
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "retention_margin_mm": RETENTION_MARGIN_MM,
            "retention_weight": RETENTION_WEIGHT,
        }
        if any(state.get(key) != value for key, value in required.items()) or state.get("config") != config.model_dump():
            raise ValueError("A10 recovery provenance differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        verify_frozen(model, baseline_state)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        _restore_rng_state(state["rng_state"])
        first_epoch = int(state["epoch"]) + 1
        history = _validate_resume_history(state.get("history"), first_epoch)
        existing_run_id = state.get("mlflow_run_id")
        if first_epoch < 1 or first_epoch > EPOCHS:
            raise ValueError("A10 already completed")
    client: Any | None = None
    run_id: str | None = existing_run_id
    try:
        client, run_id, tracking_status = _open_tracking(existing_run_id, spec, history)
        _atomic_json(out / "tracking_binding.json", {
            "schema_version": 1,
            "mlflow_run_id": run_id,
            "tracking_status": tracking_status,
            "training_manifest_sha256": _sha256(MANIFEST),
            "baseline_checkpoint_sha256": spec["baseline_checkpoint_sha256"],
            "trainer_source_sha256": _receipt_provenance(spec)["trainer_source_sha256"],
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "history_epochs_at_binding": len(history),
        })
        if not resume:
            # This epoch-zero checkpoint binds a just-created MLflow run before
            # expensive CUDA work begins; a controlled interrupt can resume from
            # the exact initial state instead of creating a second run.
            _atomic_torch_save(_recovery_payload(
                epoch=0, model=model, optimizer=optimizer, scheduler=scheduler,
                config=config, history=history, spec=spec, mlflow_run_id=run_id,
            ), out / "recovery.pth")
        _run_card(
            out, status="running", tracking_status=tracking_status, mlflow_run_id=run_id,
            decision="pending_exploratory_resource_gate", history=history or None,
        )
        _atomic_json(out / "status.json", {
            "status": "training", "pid": os.getpid(), "mlflow_run_id": run_id,
            "tracking_status": tracking_status,
            "promotion_eligible": False, "submission_zip_allowed": False,
        })
        started = time.monotonic()
        for epoch in range(first_epoch, EPOCHS + 1):
            seed_training_epoch(42, epoch)
            digest = hashlib.sha256()
            losses: list[torch.Tensor] = []
            retention_losses: list[torch.Tensor] = []
            qualified_counts: list[torch.Tensor] = []
            valid_counts: list[torch.Tensor] = []
            epoch_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for batch in train_loader:
                _digest_zero_copy(digest, batch[0])
                _digest_zero_copy(digest, batch[3])
                images, targets, masks, coords, spacing, is_target, study_mls, _ = move_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                refined, selector, coarse = frozen_forward(model, images)
                loss, parts = a10_loss(
                    refined, selector, coarse, targets, masks, coords, spacing, is_target, study_mls, config,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite A10 training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.outer_refinement.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.detach())
                retention_losses.append(parts["three_mm_retention"])
                qualified_counts.append(parts["three_mm_retention_qualified_count"])
                valid_counts.append(parts["three_mm_retention_valid_count"])
            if len(losses) != STEPS_PER_EPOCH:
                raise ValueError("A10 optimizer exposure changed")
            input_digest = digest.hexdigest()
            if epoch == 1 and input_digest != EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256:
                raise ValueError("A10 epoch1 input exposure differs from A9")
            scheduler.step()
            verify_frozen(model, baseline_state)
            loss_values = torch.stack(losses).cpu().tolist()
            retention_values = torch.stack(retention_losses).cpu().tolist()
            qualified_total = float(torch.stack(qualified_counts).sum().cpu().item())
            valid_total = float(torch.stack(valid_counts).sum().cpu().item())
            if qualified_total <= 0.0:
                raise ValueError("A10 3-mm retention guard was inactive for a full epoch")
            row = {
                "epoch": epoch,
                "optimizer_steps": len(loss_values),
                "input_exposure_sha256": input_digest,
                "train_epoch_loss": float(np.mean(loss_values, dtype=np.float64)),
                "train_epoch_seconds": time.monotonic() - epoch_started,
                "train_peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "train_three_mm_retention_loss": float(np.mean(retention_values, dtype=np.float64)),
                "train_three_mm_retention_qualified_slices": qualified_total,
                "train_three_mm_retention_valid_slices": valid_total,
            }
            history.append(row)
            state = _recovery_payload(
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                config=config, history=history, spec=spec, mlflow_run_id=run_id,
            )
            _atomic_torch_save(state, out / "recovery.pth")
            _atomic_json(out / "training_history.json", history)
            metric_error = _log_epoch_metrics(client, run_id, row)
            if metric_error is not None:
                tracking_status = "remote_tracking_degraded_" + metric_error
            print(
                "A10 " + json.dumps({
                    "epoch": epoch,
                    "train_loss": row["train_epoch_loss"],
                    "three_mm_retention": row["train_three_mm_retention_loss"],
                    "seconds": row["train_epoch_seconds"],
                }, sort_keys=True),
                flush=True,
            )
        checkpoint = out / "mls_multitask_epoch_010.pth"
        checkpoint_payload = {key: value for key, value in state.items() if key not in {
            "optimizer_state_dict", "scheduler_state_dict", "rng_state", "history",
        }}
        _atomic_torch_save(checkpoint_payload, checkpoint)
        summary = {
            "status": "completed",
            "epochs_completed": EPOCHS,
            "optimizer_steps": sum(int(row["optimizer_steps"]) for row in history),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "mlflow_run_id": run_id,
            **_receipt_provenance(spec),
            "retention_margin_mm": RETENTION_MARGIN_MM,
            "retention_weight": RETENTION_WEIGHT,
            "runtime_seconds": time.monotonic() - started,
            "validation_images_used": 0,
            "frozen_baseline_verified": True,
            "promotion_eligible": False,
            "submission_zip_allowed": False,
            "checkpoint_uploaded_to_mlflow": False,
            "fold0_evaluation_role": "exploratory_hypothesis_check_only",
        }
        _atomic_json(out / "training_summary.json", summary)
        card = _run_card(
            out, status="awaiting_exploratory_resource_screen",
            tracking_status="completion_pretermination_readback_pending",
            mlflow_run_id=run_id, decision="pending_exploratory_resource_gate", history=history,
        )
        tracking_status = _finish_tracking(client, run_id, out, history)
        card = _run_card(
            out, status="awaiting_exploratory_resource_screen", tracking_status=tracking_status,
            mlflow_run_id=run_id, decision="pending_exploratory_resource_gate", history=history,
        )
        _atomic_json(out / "status.json", {
            "status": "completed", "pid": os.getpid(), "mlflow_run_id": run_id,
            "tracking_status": tracking_status, "run_card": str(card),
            "promotion_eligible": False, "submission_zip_allowed": False,
        })
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            _mark_tracking_interrupted(client, run_id, type(exc).__name__)
            local_status = "interrupted_recoverable"
        else:
            _fail_tracking(client, run_id, type(exc).__name__)
            local_status = "failed"
        _atomic_json(out / "status.json", {
            "status": local_status, "pid": os.getpid(), "error_type": type(exc).__name__,
            "mlflow_run_id": run_id, "promotion_eligible": False,
        })
        raise


def _handle_termination(_signum: int, _frame: Any) -> None:
    raise SystemExit("Supervisor requested controlled A10 termination")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--equivalence-preflight", action="store_true")
    actions.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    lock = BASE / "gpu_training.lock"
    lock.mkdir()
    try:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True,
        ).strip()
        if active:
            raise RuntimeError("Concurrent GPU workload")
        if shutil.disk_usage(BASE).free < 15 * 2**30:
            raise RuntimeError("Need 15 GiB free for A10")
        if args.preflight:
            preflight()
        elif args.equivalence_preflight:
            equivalence_preflight()
        else:
            train(args.resume)
    finally:
        if lock.exists():
            lock.rmdir()


if __name__ == "__main__":
    main()
