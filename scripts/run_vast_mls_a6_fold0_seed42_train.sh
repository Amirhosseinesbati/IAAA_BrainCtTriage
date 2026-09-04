#!/usr/bin/env bash
# One-GPU, checksum-bound launcher for the preregistered MLS A6 resource screen.
set -Eeuo pipefail

project_root="${IAAA_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage_mls_da}"
campaign_root="${IAAA_MLS_CAMPAIGN_ROOT:-/workspace/iaaa_artifacts/mls_deploy_aligned_20260902}"
run_name="mls-vast-da-a6-local-geometry-fold0-seed42"
manifest="$project_root/config/experiments/mls-vast-deploy-aligned-a6-local-geometry-template.yaml"
artifact_root="$campaign_root/$run_name"
status_path="$artifact_root/status.json"
log_path="$artifact_root/train.log"
manifest_snapshot="$artifact_root/training_manifest_used.yaml"
global_gpu_lock="$campaign_root/gpu_training.lock"
secrets_file="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"

write_status() {
  local state="$1"
  local exit_code="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema_version":1,"state":"%s","exit_code":%s,"run_name":"%s","manifest":"%s","manifest_sha256":"%s","log_path":"%s","started_utc":"%s","finished_utc":"%s","compute_policy":"cuda_only_no_cpu_fallback","auto_destroy":false}\n' \
    "$state" "$exit_code" "$run_name" "$manifest_snapshot" "${manifest_sha256:-}" \
    "$log_path" "${started_utc:-}" "$timestamp" \
    >"$status_path.tmp"
  mv -f "$status_path.tmp" "$status_path"
}

mkdir -p "$artifact_root" "$campaign_root"
if [[ -e "$status_path" ]]; then
  echo "Refusing to overwrite existing A6 terminal or running status: $status_path" >&2
  exit 1
fi
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Project Python is unavailable: $project_root/.venv/bin/python" >&2
  exit 2
fi
if [[ ! -f "$manifest" ]]; then
  echo "A6 manifest is unavailable: $manifest" >&2
  exit 3
fi
if [[ ! -f "$secrets_file" || "$(stat -c '%a' "$secrets_file")" != "600" ]]; then
  echo "Root-only MLflow secrets file is unavailable or has unsafe permissions" >&2
  exit 4
fi
available_kib="$(df --output=avail -k "$project_root" | tail -n 1 | tr -d ' ')"
preflight="$campaign_root/a6_fold0_seed42_cuda_preflight.json"
current_manifest_sha="$(sha256sum "$manifest" | awk '{print $1}')"
if [[ ! -f "$preflight" ]] || ! jq -e --arg sha "$current_manifest_sha" \
  '.status == "ok" and .compute_policy == "cuda_only_no_cpu_fallback" and .manifest_sha256 == $sha and .batch_size == 5 and .optimizer_steps == 1' "$preflight" >/dev/null; then
  echo "Refusing A6 training without matching full-loss CUDA preflight" >&2
  exit 8
fi
minimum_kib=$((20 * 1024 * 1024))
if [[ "$available_kib" -lt "$minimum_kib" ]]; then
  echo "Refusing A6 run: less than 20 GiB free on project filesystem" >&2
  exit 5
fi
if ! mkdir "$global_gpu_lock" 2>/dev/null; then
  echo "Refusing concurrent MLS training: $global_gpu_lock exists" >&2
  exit 73
fi
cleanup() {
  rmdir "$global_gpu_lock" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' || true)"
if [[ -n "$active_gpu_pids" ]]; then
  echo "Refusing concurrent GPU compute; active PIDs: $active_gpu_pids" >&2
  exit 6
fi

cp -- "$manifest" "$manifest_snapshot"
manifest_sha256="$(sha256sum "$manifest_snapshot" | awk '{print $1}')"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "running" "null"
set -a
# shellcheck disable=SC1090
source "$secrets_file"
set +a
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$project_root"
set +e
"$project_root/.venv/bin/python" scripts/run_vast_mls_experiment.py \
  --manifest "$manifest_snapshot" \
  --allow-training \
  >"$log_path" 2>&1
exit_code=$?
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status "completed" 0
else
  write_status "failed" "$exit_code"
fi
exit "$exit_code"
