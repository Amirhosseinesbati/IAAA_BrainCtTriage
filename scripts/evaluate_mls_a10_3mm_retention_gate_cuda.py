"""One-shot, provenance-bound exploratory CUDA screen for the trained A10 candidate.

This wrapper deliberately does not fork inference logic.  It binds the exact
epoch-10 A10 checkpoint to completed training receipts, stages hash-verified
bytes, then invokes the shared epoch-10 evaluator whose decoder, pooling,
thresholds, precision, split, and CUDA policy were already qualified.  The
fold0 result is exploratory only because fold0 informed the A10 hypothesis.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reconstruct_mls_aligned_cached_screen import gate
from src.mlops.tracking import configure_tracking_environment


BASE = Path("/workspace/iaaa_artifacts/mls_deploy_aligned_20260902")
WORK = BASE / "a10_frozen_baseline_3mm_retention_20260905"
CANDIDATE_DIR = WORK / "candidate"
CANDIDATE = CANDIDATE_DIR / "mls_multitask_epoch_010.pth"
BINDING_DIR = WORK / "evaluation_binding_20260905"
OUT = WORK / "exploratory_fold0_resource_screen_20260905"

REPORTS = ROOT / "reports/mls_experiments/mls-deploy-aligned-upgrade-20260902"
TRAINING_PROTOCOL = REPORTS / "A10_TRAINING_PROTOCOL_20260905.json"
EVALUATION_PROTOCOL = REPORTS / "A10_CANONICAL_EVALUATION_PROTOCOL_20260905.json"
STATIC_CONTRACT = REPORTS / "A10_STATIC_CONTRACT_20260905.json"
STATIC_RECEIPT = REPORTS / "A10_STATIC_CONTRACT_PREFLIGHT_REMOTE_20260905.json"

PREFLIGHT = WORK / "preflight.json"
EQUIVALENCE = WORK / "speed_equivalence_preflight.json"
SUMMARY = CANDIDATE_DIR / "training_summary.json"
HISTORY = CANDIDATE_DIR / "training_history.json"
STATUS = CANDIDATE_DIR / "status.json"
RUN_CARD = CANDIDATE_DIR / "MLFLOW_RUN_CARD.json"
TRACKING_BINDING = CANDIDATE_DIR / "tracking_binding.json"

BASELINE = ROOT / "models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth"
BASELINE_SHA = "c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062"
RUNTIME_REFERENCE = BASE / "reference_refinement_runtime_qualified_20260904.json"
RUNTIME_REFERENCE_SHA = "9255e8387977c97bba19b77aa454403538abaa3ea03ddf184e89b87f136e3b96"
SHARED_EVALUATOR = ROOT / "scripts/evaluate_mls_a9_refinement_resource_cuda.py"
SHARED_EVALUATOR_SHA = "8dc05b99f851ed9597bf9ce9203666398ebfb63d8d601444ee90b5ffdba8f21f"

SOURCE_TRAINING_RUN_ID = "bb4a898d61d544c9a450bfcd4ccb4b79"
EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256 = "c27b932c23330b154b63eef8a9cba0d52a282b1760dbf536cafeee55b964352b"
SPEED_EVIDENCE_SHA256 = "3c4e4f4957c10415eeed681f5fb7ab80ff8f9af3f7671ea5cb24b5f44bbcc508"
METRIC_KEYS = {
    "mae_mm", "rmse_mm", "bias_mm", "f1_1mm", "f1_3mm", "f1_5mm",
    "boundary_f1", "selection_objective",
}
SCREEN_SCOPE = "a10_exploratory_fold0_seed42_resource_screen_only"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified_json(path: Path, *, root: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    if root is not None:
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("Receipt path escapes its required root: " + str(path)) from exc
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return parsed, digest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _receipt_provenance(spec: Mapping[str, Any], manifest_sha: str) -> dict[str, str]:
    trainer_sha = str(spec["source_and_input_sha256"]["scripts/train_mls_a10_3mm_retention_cuda.py"])
    return {
        "training_manifest_sha256": manifest_sha,
        "baseline_checkpoint_sha256": BASELINE_SHA,
        "trainer_source_sha256": trainer_sha,
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
    }


def _require_receipt_provenance(
    receipt: Mapping[str, Any], spec: Mapping[str, Any], manifest_sha: str, name: str,
) -> None:
    expected = _receipt_provenance(spec, manifest_sha)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(name + " provenance differs from the bound A10 contract")


def _verify_protocols() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    training, training_sha = _read_verified_json(TRAINING_PROTOCOL, root=ROOT)
    required_training = {
        "schema_version": 1,
        "experiment": "A10_frozen_qualified_baseline_3mm_retention",
        "baseline_checkpoint_sha256": BASELINE_SHA,
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        "expected_a9_epoch1_input_exposure_sha256": EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256,
        "runtime_speed_evidence_sha256": SPEED_EVIDENCE_SHA256,
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("A10 training protocol semantic contract differs")
    sources = training.get("source_and_input_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("A10 training protocol source pins are missing")
    for relative, expected in sources.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("A10 source pin is not a safe repository-relative path: " + str(relative))
        path = ROOT / relative_path
        resolved = path.resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError as exc:
            # The remote project intentionally exposes the immutable, hash-pinned
            # dataset through a ``Data`` symlink.  Permit that one lexical mount
            # only; source/code pins must still resolve inside the repository.
            if relative_path.parts[:1] != ("Data",):
                raise ValueError("A10 source pin escapes repository: " + str(relative)) from exc
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError("A10 training source/input changed: " + str(relative))
    _require(_sha256(BASELINE) == BASELINE_SHA, "Qualified baseline checkpoint changed")

    evaluation, evaluation_sha = _read_verified_json(EVALUATION_PROTOCOL, root=ROOT)
    required_evaluation = {
        "schema_version": 1,
        "experiment": training["experiment"],
        "stage": "exploratory_fold0_seed42_resource_screen",
    }
    if any(evaluation.get(key) != value for key, value in required_evaluation.items()):
        raise ValueError("A10 evaluation protocol semantic contract differs")
    candidate = evaluation.get("candidate")
    if not isinstance(candidate, dict) or (
        candidate.get("checkpoint_relative_name"), candidate.get("fixed_epoch"),
        candidate.get("fold"), candidate.get("seed"), candidate.get("expected_studies"),
        candidate.get("reference_refinement_enabled"), candidate.get("compute_policy"),
    ) != ("mls_multitask_epoch_010.pth", 10, 0, 42, 70, True, "cuda_only_no_cpu_model_fallback"):
        raise ValueError("A10 evaluation candidate contract differs")
    prohibitions = evaluation.get("prohibitions")
    if not isinstance(prohibitions, dict) or not all(prohibitions.get(key) is True for key in (
        "hyperparameter_or_threshold_tuning", "validation_based_selection", "automatic_replication",
        "promotion_eligible", "submission_zip_allowed", "private_prediction_upload",
    )):
        raise ValueError("A10 evaluation prohibitions differ")
    return training, training_sha, evaluation, evaluation_sha


def _verify_static_gate() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    contract, contract_sha = _read_verified_json(STATIC_CONTRACT, root=ROOT)
    receipt, receipt_sha = _read_verified_json(STATIC_RECEIPT, root=ROOT)
    if (contract.get("experiment_key"), receipt.get("status"), receipt.get("experiment_key")) != ("A10", "passed", "A10"):
        raise ValueError("A10 static-contract receipt did not pass")
    if receipt.get("cuda_or_model_work_performed") is not False or receipt.get("raw_or_processed_data_rehashed") is not False:
        raise ValueError("A10 static receipt scope differs")
    files = contract.get("files")
    checked = receipt.get("checked_files")
    if not isinstance(files, dict) or not isinstance(checked, list):
        raise ValueError("A10 static contract/receipt file schema differs")
    observed = {str(item.get("path")): item.get("sha256") for item in checked if isinstance(item, dict)}
    if any(observed.get(path) != digest for path, digest in files.items()):
        raise ValueError("A10 static receipt no longer covers its contract files")
    return contract, contract_sha, receipt, receipt_sha


def _verify_training() -> dict[str, Any]:
    spec, manifest_sha, evaluation, evaluation_sha = _verify_protocols()
    contract, contract_sha, _receipt, static_receipt_sha = _verify_static_gate()

    preflight, preflight_sha = _read_verified_json(PREFLIGHT, root=WORK)
    required_preflight = {
        "status": "completed", "baseline_identity_at_initialization": True,
        "frozen_baseline_unchanged_after_step": True, "refiner_updated": True,
        "three_mm_retention_active": True, "trainable_parameters": 47617,
        "batch_size": 16, "cuda_only": True, "validation_images_used": 0,
        "promotion_eligible": False, "submission_zip_allowed": False,
    }
    if any(preflight.get(key) != value for key, value in required_preflight.items()):
        raise ValueError("A10 CUDA preflight semantic gate failed")
    _require_receipt_provenance(preflight, spec, manifest_sha, "A10 CUDA preflight")

    equivalence, equivalence_sha = _read_verified_json(EQUIVALENCE, root=WORK)
    if equivalence.get("status") != "completed" or equivalence.get("all_gates_passed") is not True:
        raise ValueError("A10 optimized-path equivalence did not pass")
    if equivalence.get("a9_four_arm_speed_evidence_sha256") != SPEED_EVIDENCE_SHA256:
        raise ValueError("A10 speed-evidence binding differs")
    _require_receipt_provenance(equivalence, spec, manifest_sha, "A10 equivalence preflight")

    summary, summary_sha = _read_verified_json(SUMMARY, root=CANDIDATE_DIR)
    history, history_sha = _read_verified_json(HISTORY, root=CANDIDATE_DIR)
    status, status_sha = _read_verified_json(STATUS, root=CANDIDATE_DIR)
    run_card, run_card_sha = _read_verified_json(RUN_CARD, root=CANDIDATE_DIR)
    binding, binding_sha = _read_verified_json(TRACKING_BINDING, root=CANDIDATE_DIR)
    required_summary = {
        "status": "completed", "epochs_completed": 10, "optimizer_steps": 1690,
        "checkpoint": str(CANDIDATE), "validation_images_used": 0,
        "frozen_baseline_verified": True, "promotion_eligible": False,
        "submission_zip_allowed": False, "checkpoint_uploaded_to_mlflow": False,
        "fold0_evaluation_role": "exploratory_hypothesis_check_only",
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise ValueError("A10 training summary semantic gate failed")
    _require_receipt_provenance(summary, spec, manifest_sha, "A10 training summary")
    checkpoint_sha = summary.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or _sha256(CANDIDATE) != checkpoint_sha:
        raise ValueError("A10 epoch-10 checkpoint digest differs")
    checkpoint_size = CANDIDATE.stat().st_size
    if checkpoint_size <= 0:
        raise ValueError("A10 epoch-10 checkpoint is empty")
    if not isinstance(history, list) or [row.get("epoch") for row in history] != list(range(1, 11)):
        raise ValueError("A10 training epoch history differs")
    if any(row.get("optimizer_steps") != 169 for row in history):
        raise ValueError("A10 training exposure differs")
    if history[0].get("input_exposure_sha256") != EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256:
        raise ValueError("A10 epoch-1 input exposure differs from qualified A9")
    for row in history:
        if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(float(row[key])) for key in (
            "train_epoch_loss", "train_epoch_seconds", "train_peak_vram_gib",
            "train_three_mm_retention_loss", "train_three_mm_retention_qualified_slices",
            "train_three_mm_retention_valid_slices",
        )):
            raise ValueError("A10 training history has nonfinite public metrics")
        if float(row["train_three_mm_retention_qualified_slices"]) <= 0:
            raise ValueError("A10 retention loss was inactive for a completed epoch")
    run_id = summary.get("mlflow_run_id")
    if not isinstance(run_id, str) or len(run_id) != 32:
        raise ValueError("A10 MLflow run identity is invalid")
    if (
        status.get("status"), status.get("mlflow_run_id"), status.get("tracking_status"),
        binding.get("mlflow_run_id"), run_card.get("mlflow_run_id"),
    ) != (
        "completed", run_id, "metrics_history_and_terminal_readback_verified", run_id, run_id,
    ):
        raise ValueError("A10 MLflow/local training identity differs")
    if (
        run_card.get("candidate_status"), run_card.get("promotion_eligible"),
        run_card.get("submission_zip_allowed"), run_card.get("private_predictions_uploaded"),
    ) != ("awaiting_exploratory_resource_screen", False, False, False):
        raise ValueError("A10 run-card eligibility differs")
    return {
        "spec": spec,
        "manifest_sha": manifest_sha,
        "evaluation_protocol": evaluation,
        "evaluation_protocol_sha": evaluation_sha,
        "static_contract": contract,
        "static_contract_sha": contract_sha,
        "static_receipt_sha": static_receipt_sha,
        "preflight_sha": preflight_sha,
        "equivalence_sha": equivalence_sha,
        "training_summary_sha": summary_sha,
        "training_history_sha": history_sha,
        "training_status_sha": status_sha,
        "run_card_sha": run_card_sha,
        "tracking_binding_sha": binding_sha,
        "checkpoint_sha": checkpoint_sha,
        "checkpoint_size": checkpoint_size,
        "mlflow_run_id": run_id,
    }


def _stage_candidate(binding: Mapping[str, Any]) -> tuple[Path, Path]:
    if BINDING_DIR.exists() or OUT.exists():
        raise FileExistsError("A10 evaluation binding/output already exists; do not overwrite or rerun")
    BINDING_DIR.mkdir()
    staged = BINDING_DIR / ("mls_multitask_epoch_010_" + str(binding["checkpoint_sha"]) + ".pth")
    shutil.copyfile(CANDIDATE, staged)
    if _sha256(staged) != binding["checkpoint_sha"]:
        raise ValueError("A10 staged checkpoint digest differs")
    manifest = {
        "schema_version": 1,
        "experiment_key": "A10",
        "completion_status": "completed",
        "checkpoint_relative_path": staged.name,
        "checkpoint_sha256": binding["checkpoint_sha"],
        "checkpoint_size_bytes": binding["checkpoint_size"],
        "fixed_epoch": 10,
        "fold": 0,
        "seed": 42,
        "optimizer_steps": 1690,
        "training_protocol_sha256": binding["manifest_sha"],
        "evaluation_protocol_sha256": binding["evaluation_protocol_sha"],
        "static_contract_sha256": binding["static_contract_sha"],
        "static_contract_receipt_sha256": binding["static_receipt_sha"],
        "trainer_source_sha256": binding["spec"]["source_and_input_sha256"]["scripts/train_mls_a10_3mm_retention_cuda.py"],
        "baseline_checkpoint_sha256": BASELINE_SHA,
        "preflight_receipt_sha256": binding["preflight_sha"],
        "equivalence_receipt_sha256": binding["equivalence_sha"],
        "training_epoch1_input_exposure_sha256": EXPECTED_A9_EPOCH1_INPUT_EXPOSURE_SHA256,
        "mlflow_run_id": binding["mlflow_run_id"],
        "promotion_eligible": False,
        "submission_zip_allowed": False,
    }
    manifest_path = BINDING_DIR / "CANDIDATE_MANIFEST.json"
    _atomic_json(manifest_path, manifest)
    return staged, manifest_path


def _verify_audit(audit: Mapping[str, Any], staged: Path, binding: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, bool], dict[str, float]]:
    reference, reference_sha = _read_verified_json(RUNTIME_REFERENCE, root=BASE)
    if reference_sha != RUNTIME_REFERENCE_SHA:
        raise ValueError("Qualified runtime reference changed")
    expected = {
        "status": "completed", "scope": "canonical_fold0_seed42_resource_screen_only",
        "baseline_self_test": False, "checkpoint_sha256": binding["checkpoint_sha"],
        "checkpoint": str(staged.resolve()), "fold": 0, "seed": 42, "fixed_epoch": 10,
        "studies": 70, "compute_policy": "cuda_only_no_cpu_model_fallback",
        "source_sha256": SHARED_EVALUATOR_SHA, "runtime_reference_sha256": RUNTIME_REFERENCE_SHA,
        "reference_refinement_enabled": True, "promotion_eligible": False,
        "submission_zip_allowed": False, "automatic_replication_allowed": False,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("A10 shared CUDA evaluator output contract differs")
    observed = audit.get("observed")
    if not isinstance(observed, dict) or set(observed) != METRIC_KEYS:
        raise ValueError("A10 aggregate metric schema differs")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in observed.values()):
        raise ValueError("A10 aggregate metrics are nonfinite")
    if float(observed["mae_mm"]) < 0 or float(observed["rmse_mm"]) < 0:
        raise ValueError("A10 aggregate localization metrics are impossible")
    if any(not 0 <= float(observed[key]) <= 1 for key in ("f1_1mm", "f1_3mm", "f1_5mm", "boundary_f1")):
        raise ValueError("A10 aggregate boundary metrics are impossible")
    if abs(float(observed["selection_objective"]) - (float(observed["mae_mm"]) + 2 * (1 - float(observed["boundary_f1"])))) > 1e-8:
        raise ValueError("A10 aggregate objective does not match its components")
    baseline = reference.get("runtime_baseline_metrics")
    bounds = reference.get("prospective_gate_bounds")
    if not isinstance(baseline, dict) or not isinstance(bounds, dict):
        raise ValueError("Qualified runtime reference gate schema differs")
    gates = gate(observed, bounds, 1e-8)
    if audit.get("gate_results") != gates or audit.get("resource_gates_passed") is not bool(all(gates.values())):
        raise ValueError("A10 evaluator gate results are inconsistent")
    return {key: float(value) for key, value in observed.items()}, gates, {key: float(value) for key, value in baseline.items()}


def _create_evaluation_projection(
    parent_run_id: str,
    binding: Mapping[str, Any],
    observed: Mapping[str, float],
    gates: Mapping[str, bool],
    audit_sha: str,
) -> tuple[str | None, str]:
    """Record the immutable audit in a linked, terminal MLflow projection run.

    The A10 training run is deliberately already ``FINISHED``.  MLflow rejects
    metric/tag mutation on terminal runs, so trying to append audit facts there
    would only produce an apparently successful local result with a broken UI.
    This small, explicitly linked run is an evaluation projection, *not* a new
    training candidate or a selection/promotion event.
    """
    projection_run_id: str | None = None
    client = None
    try:
        from mlflow.tracking import MlflowClient

        configure_tracking_environment()
        if not os.getenv("MLFLOW_TRACKING_URI"):
            return None, "evaluation_projection_skipped_missing_remote_tracking"
        client = MlflowClient()
        parent = client.get_run(parent_run_id)
        if parent.info.status != "FINISHED":
            raise ValueError("A10 parent training run is not terminal FINISHED")
        required_parent_tags = {"experiment_key": "A10", "candidate_model": "true"}
        if any(parent.data.tags.get(key) != value for key, value in required_parent_tags.items()):
            raise ValueError("A10 MLflow parent lineage differs")

        tags = {
            "mlflow.runName": "MLS | A10 | exploratory screen | F0/S42 | COMPLETED",
            "campaign_id": "mls-deploy-aligned-20260902",
            "experiment_key": "A10",
            "run_type": "exploratory_resource_screen_projection",
            "candidate_model": "true",
            "parent_training_run_id": parent_run_id,
            "source_training_run_id": SOURCE_TRAINING_RUN_ID,
            "candidate_checkpoint_sha256": str(binding["checkpoint_sha"]),
            "training_manifest_sha256": str(binding["manifest_sha"]),
            "evaluation_protocol_sha256": str(binding["evaluation_protocol_sha"]),
            "static_contract_sha256": str(binding["static_contract_sha"]),
            "audit_aggregate_sha256": audit_sha,
            "stage": "exploratory_resource_screen_completed",
            "candidate_status": "exploratory_screen_completed",
            "decision": "exploratory_result_not_promotion_eligible",
            "promotion_eligible": "false",
            "submission_zip_allowed": "false",
            "private_predictions_uploaded": "false",
            "fold0_evaluation_role": "exploratory_hypothesis_check_only",
            "tracking_lifecycle": "projection_running",
        }
        projection_run_id = client.create_run(parent.info.experiment_id, tags=tags).info.run_id
        for key, value in {
            "fixed_epoch": 10,
            "fold": 0,
            "seed": 42,
            "expected_studies": 70,
            "candidate_checkpoint_sha256": str(binding["checkpoint_sha"]),
        }.items():
            client.log_param(projection_run_id, key, str(value))
        for key, value in observed.items():
            client.log_metric(projection_run_id, "audit_" + key, float(value), step=0)
        for key, value in gates.items():
            client.log_metric(projection_run_id, "gate_" + key, float(value), step=0)
        client.log_metric(projection_run_id, "resource_gates_passed", float(all(gates.values())), step=0)
        client.set_tag(projection_run_id, "tracking_lifecycle", "projection_finalizing")
        client.set_terminated(projection_run_id, status="FINISHED")

        verified = client.get_run(projection_run_id)
        expected_tags = {
            "parent_training_run_id": parent_run_id,
            "run_type": "exploratory_resource_screen_projection",
            "audit_aggregate_sha256": audit_sha,
            "stage": "exploratory_resource_screen_completed",
            "candidate_status": "exploratory_screen_completed",
            "decision": "exploratory_result_not_promotion_eligible",
            "promotion_eligible": "false",
            "submission_zip_allowed": "false",
            "private_predictions_uploaded": "false",
        }
        if verified.info.status != "FINISHED" or any(
            verified.data.tags.get(key) != value for key, value in expected_tags.items()
        ):
            raise ValueError("A10 evaluation projection readback tag/status failed")
        expected_metrics = {
            **{"audit_" + key: float(value) for key, value in observed.items()},
            **{"gate_" + key: float(value) for key, value in gates.items()},
            "resource_gates_passed": float(all(gates.values())),
        }
        if any(
            not math.isclose(float(verified.data.metrics.get(key, float("nan"))), value, rel_tol=0.0, abs_tol=1e-12)
            for key, value in expected_metrics.items()
        ):
            raise ValueError("A10 evaluation projection readback metrics failed")
        return projection_run_id, "evaluation_projection_readback_verified"
    except Exception as exc:
        if projection_run_id is not None and client is not None:
            try:
                current = client.get_run(projection_run_id)
                if current.info.status == "RUNNING":
                    client.set_tag(projection_run_id, "tracking_lifecycle", "projection_tracking_failed")
                    client.set_tag(projection_run_id, "failure_type", type(exc).__name__)
                    client.set_terminated(projection_run_id, status="FAILED")
            except Exception:
                pass
        return projection_run_id, "evaluation_projection_unverified_" + type(exc).__name__


def _write_failure_status(state: Mapping[str, Any], error_type: str) -> None:
    """Persist a privacy-safe, fail-closed receipt if binding has begun.

    No raw exception text, paths to private predictions, or patient data are
    recorded.  A failed invocation is intentionally never retried implicitly:
    an operator must inspect this receipt and choose a new, explicitly approved
    output/binding lineage before rerunning CUDA.
    """
    receipt = {
        "status": "failed",
        "scope": SCREEN_SCOPE,
        "error_type": error_type,
        "provenance_verified": bool(state.get("provenance_verified", False)),
        "binding_created": bool(state.get("binding_created", False)),
        "evaluator_started": bool(state.get("evaluator_started", False)),
        "evaluator_returncode": state.get("evaluator_returncode"),
        "promotion_eligible": False,
        "submission_zip_allowed": False,
        "private_predictions_uploaded": False,
        "automatic_retry_allowed": False,
        "retry_policy": "fail_closed_no_implicit_retry",
    }
    for target in (BINDING_DIR / "evaluation_binding_status.json", OUT / "wrapper_status.json"):
        if target.parent.exists():
            try:
                _atomic_json(target, receipt)
            except Exception:
                pass


def main() -> None:
    state: dict[str, Any] = {
        "provenance_verified": False,
        "binding_created": False,
        "evaluator_started": False,
        "evaluator_returncode": None,
    }
    try:
        binding = _verify_training()
        state["provenance_verified"] = True
        shared_before = _sha256(SHARED_EVALUATOR)
        if shared_before != SHARED_EVALUATOR_SHA:
            raise ValueError("Shared epoch-10 evaluator changed before CUDA")
        staged, manifest_path = _stage_candidate(binding)
        state["binding_created"] = True
        _atomic_json(BINDING_DIR / "evaluation_binding_status.json", {
            "status": "bound_before_cuda", "candidate_manifest": str(manifest_path),
            "checkpoint_sha256": binding["checkpoint_sha"], "promotion_eligible": False,
            "automatic_retry_allowed": False,
        })
        OUT.mkdir()
        state["evaluator_started"] = True
        with (OUT / "candidate.process.log").open("x", encoding="utf-8") as log:
            process = subprocess.run([
                sys.executable, str(SHARED_EVALUATOR), "--checkpoint", str(staged),
                "--checkpoint-sha256", binding["checkpoint_sha"], "--runtime-reference", str(RUNTIME_REFERENCE),
                "--runtime-reference-sha256", RUNTIME_REFERENCE_SHA, "--output-dir", str(OUT / "candidate"),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["evaluator_returncode"] = process.returncode
        shared_after = _sha256(SHARED_EVALUATOR)
        if shared_before != SHARED_EVALUATOR_SHA or shared_after != SHARED_EVALUATOR_SHA:
            raise ValueError("Shared epoch-10 evaluator changed during CUDA evaluation")
        if process.returncode != 0:
            raise RuntimeError("A10 shared CUDA evaluator failed; preserve output and do not rerun")
        audit_path = OUT / "candidate" / "aggregate_summary.json"
        audit, audit_sha = _read_verified_json(audit_path, root=OUT)
        observed, gates, baseline = _verify_audit(audit, staged, binding)
        result = {
            "status": "completed",
            "scope": SCREEN_SCOPE,
            "candidate_checkpoint_sha256": binding["checkpoint_sha"],
            "candidate_manifest_sha256": _sha256(manifest_path),
            "audit_aggregate_sha256": audit_sha,
            "training_manifest_sha256": binding["manifest_sha"],
            "evaluation_protocol_sha256": binding["evaluation_protocol_sha"],
            "static_contract_sha256": binding["static_contract_sha"],
            "static_contract_receipt_sha256": binding["static_receipt_sha"],
            "preflight_receipt_sha256": binding["preflight_sha"],
            "equivalence_receipt_sha256": binding["equivalence_sha"],
            "shared_evaluator_sha256": SHARED_EVALUATOR_SHA,
            "shared_evaluator_sha256_after_cuda": shared_after,
            "runtime_reference_sha256": RUNTIME_REFERENCE_SHA,
            "baseline": baseline,
            "candidate": observed,
            "candidate_minus_baseline": {key: observed[key] - baseline[key] for key in observed},
            "resource_gates_passed": bool(all(gates.values())),
            "gate_results": gates,
            "parent_training_mlflow_run_id": binding["mlflow_run_id"],
            "fold0_informed_hypothesis": True,
            "promotion_eligible": False,
            "submission_zip_allowed": False,
            "private_predictions_uploaded": False,
            "required_next_stage_if_promising": "pre_registered_unused_fold_or_leak_free_multifold_evaluation",
        }
        _atomic_json(OUT / "evaluation_result.json", result)
        projection_run_id, tracking_status = _create_evaluation_projection(
            binding["mlflow_run_id"], binding, observed, gates, audit_sha,
        )
        result["evaluation_mlflow_run_id"] = projection_run_id
        result["mlflow_tracking_status"] = tracking_status
        _atomic_json(OUT / "evaluation_result.json", result)
        _atomic_json(BINDING_DIR / "evaluation_binding_status.json", {
            "status": "completed", "candidate_manifest": str(manifest_path),
            "checkpoint_sha256": binding["checkpoint_sha"], "audit_aggregate_sha256": audit_sha,
            "evaluator_started": True, "evaluator_returncode": 0,
            "promotion_eligible": False, "submission_zip_allowed": False,
            "automatic_retry_allowed": False,
        })
        print(json.dumps({
            "status": result["status"], "candidate": observed, "resource_gates_passed": result["resource_gates_passed"],
            "promotion_eligible": False, "mlflow_tracking_status": tracking_status,
        }, sort_keys=True))
    except BaseException as exc:
        _write_failure_status(state, type(exc).__name__)
        print(json.dumps({
            "status": "failed", "error_type": type(exc).__name__,
            "evaluator_started": state["evaluator_started"],
            "evaluator_returncode": state["evaluator_returncode"],
        }, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
