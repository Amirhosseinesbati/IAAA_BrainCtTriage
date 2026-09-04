#!/usr/bin/env bash
# Strict, one-checkpoint CUDA resource screen for the pre-registered MLS A2 run.
# It deliberately has no path to choose another checkpoint or make a release claim.
set -Eeuo pipefail

project_root="${IAAA_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage}"
campaign_root="${IAAA_MLS_CAMPAIGN_ROOT:-/workspace/iaaa_artifacts/mls_deploy_aligned_20260902}"
run_name="mls-vast-da-a2-signed-geometry-fold0-seed42"
training_artifact_root="$campaign_root/a2_fold0_seed42"
audit_artifact_root="$campaign_root/a2_fold0_seed42_cuda_audit"
report_root="$project_root/reports/mls_experiments/mls-deploy-aligned-upgrade-20260902"
global_gpu_lock="$campaign_root/gpu_training.lock"
training_status="$training_artifact_root/launcher_status.json"
training_report="$project_root/reports/mls_experiments/$run_name/report.md"
checkpoint="$project_root/models/checkpoints/mls_multitask/$run_name/mls_multitask_epoch_015.pth"
audit_status="$audit_artifact_root/audit_status.json"
metrics="$audit_artifact_root/epoch015/metrics.json"
decision="$report_root/A2_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json"
launcher_status="$audit_artifact_root/audit_launcher_status.json"
secrets_file="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"

write_status() {
  local status="$1"
  local code="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$audit_artifact_root"
  printf '{"status":"%s","exit_code":%s,"checkpoint":"%s","audit_status":"%s","decision":"%s","updated_utc":"%s","compute_policy":"cuda_only_no_cpu_fallback"}\n' \
    "$status" "$code" "$checkpoint" "$audit_status" "$decision" "$timestamp" \
    >"$launcher_status.tmp"
  mv -f "$launcher_status.tmp" "$launcher_status"
}

mkdir -p "$audit_artifact_root" "$report_root" "$campaign_root"
write_status "preflight" "null"

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Project Python is unavailable: $project_root/.venv/bin/python" >&2
  write_status "refused_missing_project_python" 2
  exit 2
fi
if [[ ! -f "$training_status" ]] || ! grep -q '"status":"completed"' "$training_status"; then
  echo "A2 training has not completed cleanly; refusing overlapping resource screen" >&2
  write_status "refused_training_not_completed" 3
  exit 3
fi
if [[ ! -f "$checkpoint" || ! -f "$training_report" ]]; then
  echo "Required fixed epoch-15 checkpoint/report is unavailable" >&2
  write_status "refused_missing_fixed_artifact" 4
  exit 4
fi
if [[ ! -f "$secrets_file" || "$(stat -c '%a' "$secrets_file")" != "600" ]]; then
  echo "Root-only MLflow secrets file is unavailable or has unsafe permissions" >&2
  write_status "refused_mlflow_secrets" 5
  exit 5
fi
mlflow_run_id="$(awk -F'`' '/MLflow run id:/ {print $2}' "$training_report" | tail -n 1)"
if [[ ! "$mlflow_run_id" =~ ^[0-9a-f]{32}$ ]]; then
  echo "Could not recover a valid MLflow run id from the completed A2 report" >&2
  write_status "refused_missing_mlflow_run_id" 6
  exit 6
fi
if ! mkdir "$global_gpu_lock" 2>/dev/null; then
  echo "Refusing concurrent MLS audit: $global_gpu_lock exists" >&2
  write_status "refused_gpu_lock_exists" 73
  exit 73
fi
cleanup() {
  rmdir "$global_gpu_lock" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' || true)"
if [[ -n "$active_gpu_pids" ]]; then
  echo "Refusing concurrent GPU compute; active PIDs: $active_gpu_pids" >&2
  write_status "refused_gpu_compute_process_exists" 7
  exit 7
fi

set -a
# shellcheck disable=SC1090
source "$secrets_file"
set +a
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

write_status "running" "null"
cd "$project_root"
set +e
"$project_root/.venv/bin/python" scripts/audit_mls_checkpoints_cuda.py \
  --fold 0 \
  --batch-size 16 \
  --expected-studies 70 \
  --output-root "$audit_artifact_root" \
  --candidate "epoch015=$checkpoint" \
  --mlflow-run-id "$mlflow_run_id"
audit_exit=$?
set -e
if [[ "$audit_exit" -ne 0 ]]; then
  write_status "failed_cuda_audit" "$audit_exit"
  exit "$audit_exit"
fi
"$project_root/.venv/bin/python" scripts/evaluate_mls_a2_fold0_resource_screen.py \
  --audit-status "$audit_status" \
  --metrics "$metrics" \
  --checkpoint "$checkpoint" \
  --output "$decision" \
  --mlflow-run-id "$mlflow_run_id"
write_status "completed" 0
printf 'A2 fold-0 seed-42 resource screen completed: %s\n' "$decision"
