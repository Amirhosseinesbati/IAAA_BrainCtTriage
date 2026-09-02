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

mkdir -p "$artifact_root"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "Refusing duplicate launch: $lock_dir exists" >&2
  exit 2
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

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

