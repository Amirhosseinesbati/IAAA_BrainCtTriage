#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST ARTIFACT_ROOT" >&2
  exit 64
fi

project_root=/workspace/IAAA_BrainCtTriage
manifest=$1
artifact_root=$2
lock_dir="$artifact_root/run.lock"
campaign_root=/workspace/iaaa_artifacts/mls_deploy_aligned_20260902
global_gpu_lock="$campaign_root/gpu_training.lock"
minimum_free_gib="${IAAA_MLS_MIN_FREE_GIB:-4}"

mkdir -p "$artifact_root" "$campaign_root"
preflight_status="$artifact_root/preflight_status.json"

if ! [[ "$minimum_free_gib" =~ ^[0-9]+$ ]] || [[ "$minimum_free_gib" -lt 1 ]]; then
  echo "IAAA_MLS_MIN_FREE_GIB must be a positive integer" >&2
  exit 65
fi

available_kib=$(df --output=avail -k "$project_root" | tail -n 1 | tr -d ' ')
required_kib=$((minimum_free_gib * 1024 * 1024))
if [[ "$available_kib" -lt "$required_kib" ]]; then
  printf '{"status":"refused","reason":"insufficient_disk","available_kib":%d,"required_kib":%d}\n' \
    "$available_kib" "$required_kib" > "$preflight_status"
  echo "Refusing MLS run: less than ${minimum_free_gib} GiB free" >&2
  exit 4
fi

if ! mkdir "$global_gpu_lock" 2>/dev/null; then
  printf '{"status":"refused","reason":"global_gpu_lock_exists","lock":"%s"}\n' \
    "$global_gpu_lock" > "$preflight_status"
  echo "Refusing concurrent MLS run: $global_gpu_lock exists" >&2
  exit 5
fi

cleanup_locks() {
  rmdir "$lock_dir" 2>/dev/null || true
  rmdir "$global_gpu_lock" 2>/dev/null || true
}
trap cleanup_locks EXIT

active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | tr -d ' ' | sed '/^$/d' || true)
if [[ -n "$active_gpu_pids" ]]; then
  printf '{"status":"refused","reason":"gpu_compute_process_exists","pids":"%s"}\n' \
    "$(printf '%s' "$active_gpu_pids" | tr '\n' ',')" > "$preflight_status"
  echo "Refusing concurrent GPU compute; active PIDs: $active_gpu_pids" >&2
  exit 6
fi

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Refusing duplicate launch: $lock_dir exists" >&2
  exit 2
fi
printf '{"status":"passed","available_kib":%d,"required_kib":%d,"global_gpu_lock":"%s"}\n' \
  "$available_kib" "$required_kib" "$global_gpu_lock" > "$preflight_status"

cd "$project_root"
if [[ ! -f "$manifest" ]]; then
  echo "Manifest not found: $manifest" >&2
  exit 3
fi

set -a
source /root/.config/iaaa/secrets.env
set +a
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"status":"running","manifest":"%s","manifest_sha256":"%s","started_utc":"%s","compute_policy":"cuda_only_no_cpu_fallback"}\n' \
  "$manifest" "$manifest_sha" "$started_utc" > "$artifact_root/launcher_status.json"

set +e
.venv/bin/python scripts/run_vast_mls_experiment.py \
  --manifest "$manifest" \
  --allow-training \
  > "$artifact_root/run.log" 2>&1
exit_code=$?
set -e

finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
status=failed
if [[ $exit_code -eq 0 ]]; then
  status=completed
fi
printf '{"status":"%s","exit_code":%d,"manifest":"%s","manifest_sha256":"%s","started_utc":"%s","finished_utc":"%s","compute_policy":"cuda_only_no_cpu_fallback"}\n' \
  "$status" "$exit_code" "$manifest" "$manifest_sha" "$started_utc" "$finished_utc" \
  > "$artifact_root/launcher_status.json"
exit "$exit_code"
