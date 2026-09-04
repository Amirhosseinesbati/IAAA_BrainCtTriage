#!/usr/bin/env bash
# Run the preregistered A4 fold-0 three-seed CUDA audit and triage gate.
#
# It is intentionally unable to choose checkpoints, pooling, thresholds, or a
# different fold. Per-study files remain under the server-only campaign root.
set -Eeuo pipefail

project_root="${IAAA_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage_mls_da}"
canonical_project_root="${IAAA_CANONICAL_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage}"
campaign_root="${IAAA_MLS_CAMPAIGN_ROOT:-/workspace/iaaa_artifacts/mls_deploy_aligned_20260902}"
report_root="$project_root/reports/mls_experiments/mls-deploy-aligned-upgrade-20260902"
preregistration="$report_root/FOLD0_A4_THREE_SEED_AUDIT_PREREGISTRATION.json"
resource_decision="$report_root/A4_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json"
triage_preregistration="$report_root/FOLD0_A4_TRIAGE_SCREEN_PREREGISTRATION.json"
triage_decision="$report_root/A4_FOLD0_THREE_SEED_TRIAGE_DECISION.json"
candidate_audit_root="$campaign_root/a4_fold0_three_seed_fixed_epoch15_audit"
candidate_private="$candidate_audit_root/study_member_predictions_private.csv"
candidate_summary="$candidate_audit_root/aggregate_summary.json"
baseline_audit_root="$campaign_root/fold0_three_seed_fixed_epoch15_audit"
baseline_private="$baseline_audit_root/study_member_predictions_private.csv"
baseline_summary="$baseline_audit_root/aggregate_summary.json"
triage_root="$campaign_root/a4_fold0_triage_comparison"
triage_summary="$triage_root/aggregate_summary.json"
checkpoint42="$project_root/models/checkpoints/mls_multitask/mls-vast-da-a4-pair-rank-fold0-seed42/mls_multitask_epoch_015.pth"
checkpoint2026="$project_root/models/checkpoints/mls_multitask/mls-vast-da-a4-pair-rank-fold0-seed2026/mls_multitask_epoch_015.pth"
checkpoint3407="$project_root/models/checkpoints/mls_multitask/mls-vast-da-a4-pair-rank-fold0-seed3407/mls_multitask_epoch_015.pth"
status42="$campaign_root/mls-vast-da-a4-pair-rank-fold0-seed42/status.json"
status2026="$campaign_root/mls-vast-da-a4-pair-rank-fold0-seed2026/status.json"
status3407="$campaign_root/mls-vast-da-a4-pair-rank-fold0-seed3407/status.json"
global_gpu_lock="$campaign_root/gpu_training.lock"
frozen_predictions="/workspace/iaaa_artifacts/frozen_champion_branches_20260902/predictions_private.csv"
frozen_sha256="3f58a90c6525644e32eb244d1723210a9dd422b6f33fbbada25afe6bc180a2a9"
truth_table="$canonical_project_root/reports/eda/deep/deep_series_table.csv"
fold_manifest="$project_root/config/folds.csv"

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Project Python is unavailable: $project_root/.venv/bin/python" >&2
  exit 2
fi
for required in \
  "$preregistration" "$resource_decision" "$triage_preregistration" \
  "$project_root/scripts/run_vast_mls_three_seed_audit.sh" \
  "$project_root/scripts/evaluate_mls_deploy_aligned_seed_medians.py" \
  "$project_root/scripts/evaluate_mls_a4_fold0_triage_screen.py" \
  "$baseline_private" "$baseline_summary" "$frozen_predictions" "$truth_table" "$fold_manifest" \
  "$status42" "$status2026" "$status3407" \
  "$checkpoint42" "$checkpoint2026" "$checkpoint3407"; do
  if [[ ! -f "$required" ]]; then
    echo "Required A4 fold-0 gate artifact is missing: $required" >&2
    exit 3
  fi
done
if [[ -e "$candidate_audit_root" || -e "$triage_root" || -e "$triage_decision" ]]; then
  echo "Refusing to overwrite an existing A4 three-seed audit or triage result" >&2
  exit 4
fi

if ! "$project_root/.venv/bin/python" - \
  "$preregistration" "$resource_decision" "$triage_preregistration" \
  "$status42" "$status2026" "$status3407" \
  "$checkpoint42" "$checkpoint2026" "$checkpoint3407" \
  "$baseline_summary" "$baseline_private" "$frozen_predictions" "$truth_table" "$fold_manifest" "$frozen_sha256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    prereg_path, resource_path, triage_prereg_path,
    status42, status2026, status3407,
    checkpoint42, checkpoint2026, checkpoint3407,
    baseline_summary, baseline_private, frozen_path, truth_path, fold_manifest,
) = map(Path, sys.argv[1:-1])
frozen_sha = sys.argv[-1]

def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unreadable A4 JSON contract: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"A4 JSON contract is not an object: {path}")
    return value

prereg = load(prereg_path)
resource = load(resource_path)
triage_prereg = load(triage_prereg_path)
statuses = [load(path) for path in (status42, status2026, status3407)]
checkpoints = [checkpoint42, checkpoint2026, checkpoint3407]
expected_paths = prereg.get("checkpoints", {})
checks = {
    "prereg": prereg.get("status") == "locked_before_any_a4_fold0_three_seed_outcome",
    "stage": prereg.get("candidate_stage") == "a4_pair_rank",
    "protocol": prereg.get("protocol") == "heldout_fold_fixed_epoch15_three_distinct_seed_median",
    "seeds": prereg.get("seeds") == [42, 2026, 3407],
    "resource_status": resource.get("status") == "passed_for_two_remaining_fold0_seed_replications",
    "resource_authorization": resource.get("can_start_only_seeds_2026_and_3407_on_fold0") is True,
    "resource_clean": resource.get("failed_gates") == [],
    "resource_candidate": resource.get("candidate") == "mls-vast-deploy-aligned-a4-pair-rank",
    "resource_scope": resource.get("screen_scope") == "a4_fold0_seed42_resource_screen_only",
    "triage_prereg": triage_prereg.get("status") == "locked_before_any_a4_audit_or_triage_outcome",
    "terminal_statuses": all(item.get("state") == "completed" and item.get("exit_code") == 0 for item in statuses),
    "status_run_names": [item.get("run_name") for item in statuses] == [
        "mls-vast-da-a4-pair-rank-fold0-seed42",
        "mls-vast-da-a4-pair-rank-fold0-seed2026",
        "mls-vast-da-a4-pair-rank-fold0-seed3407",
    ],
    "replica_status_seeds": [item.get("seed") for item in statuses[1:]] == [2026, 3407],
    "fixed_paths": all(str(path) == expected_paths.get(label) for label, path in zip(("seed42", "seed2026", "seed3407"), checkpoints)),
    "baseline_summary": hashlib.sha256(baseline_summary.read_bytes()).hexdigest() == prereg["baseline"]["aggregate_summary_sha256"],
    "baseline_private": hashlib.sha256(baseline_private.read_bytes()).hexdigest() == prereg["baseline"]["private_predictions_sha256"],
    "frozen_path": str(frozen_path) == prereg["frozen_champion"]["predictions"],
    "frozen": hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_sha == prereg["frozen_champion"]["sha256"],
    "truth_path": str(truth_path) == prereg["truth_table"]["path"],
    "truth": hashlib.sha256(truth_path.read_bytes()).hexdigest() == prereg["truth_table"]["sha256"],
    "fold_manifest_path": str(fold_manifest) == prereg["fold_manifest"]["path"],
    "fold_manifest": hashlib.sha256(fold_manifest.read_bytes()).hexdigest() == prereg["fold_manifest"]["sha256"],
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"A4 fold-0 three-seed gate refused: {failed}")
PY
then
  echo "A4 three-seed audit prerequisites were not met" >&2
  exit 5
fi

if [[ -e "$global_gpu_lock" ]]; then
  echo "Refusing A4 three-seed audit while the GPU lock exists" >&2
  exit 73
fi
active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' || true)"
if [[ -n "$active_gpu_pids" ]]; then
  echo "Refusing concurrent GPU compute; active PIDs: $active_gpu_pids" >&2
  exit 74
fi

cd "$project_root"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$project_root/scripts/run_vast_mls_three_seed_audit.sh" \
  --checkpoint "seed42=$checkpoint42" \
  --checkpoint "seed2026=$checkpoint2026" \
  --checkpoint "seed3407=$checkpoint3407" \
  --fold 0 --fixed-epoch 15 --expected-studies 70 --batch-size 6 \
  --output-dir "$candidate_audit_root"

"$project_root/.venv/bin/python" scripts/evaluate_mls_deploy_aligned_seed_medians.py \
  --baseline-fold "0=$baseline_private" --baseline-fold-summary "0=$baseline_summary" \
  --candidate-fold "0=$candidate_private" --candidate-fold-summary "0=$candidate_summary" \
  --fold-manifest "$fold_manifest" \
  --frozen-champion-predictions "$frozen_predictions" \
  --expected-frozen-champion-sha256 "$frozen_sha256" \
  --truth-table "$truth_table" \
  --output-dir "$triage_root"

"$project_root/.venv/bin/python" scripts/evaluate_mls_a4_fold0_triage_screen.py \
  --aggregate-summary "$triage_summary" \
  --preregistration "$triage_preregistration" \
  --output "$triage_decision"
