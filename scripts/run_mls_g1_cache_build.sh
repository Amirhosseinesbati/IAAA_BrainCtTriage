#!/usr/bin/env bash
# Managed, no-model launcher for the immutable G1 MLS 2.5D cache.
#
# The cache is deliberately built before any CUDA training.  It performs DICOM
# decoding/windowing only; it never constructs or runs an ML model.
set -Eeuo pipefail

project_root="${IAAA_G1_PROJECT_ROOT:?IAAA_G1_PROJECT_ROOT is required}"
python_bin="${IAAA_G1_PYTHON:?IAAA_G1_PYTHON is required}"
artifact_root="${IAAA_G1_ARTIFACT_ROOT:?IAAA_G1_ARTIFACT_ROOT is required}"
cache_root="$project_root/Data/processed/mls_2p5d_v1"
status_path="$artifact_root/cache_build_status.json"
lock_path="$artifact_root/cache_build.lock"

write_status() {
  local state="$1"
  local exit_code="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema_version":1,"campaign":"g1_2p5d_deploy_aligned","state":"%s","exit_code":%s,"cache_root":"%s","source_commit":"%s","model_compute":"none","pixel_decode":true,"started_utc":"%s","finished_utc":"%s"}\n' \
    "$state" "$exit_code" "$cache_root" "$(git -C "$project_root" rev-parse HEAD)" \
    "${started_utc:-}" "$timestamp" >"$status_path.tmp"
  mv -f "$status_path.tmp" "$status_path"
}

mkdir -p "$artifact_root"
if [[ ! -x "$python_bin" ]]; then
  echo "G1 Python environment is unavailable: $python_bin" >&2
  exit 2
fi
if ! mkdir "$lock_path" 2>/dev/null; then
  echo "Refusing concurrent G1 cache build: $lock_path exists" >&2
  exit 73
fi
cleanup() {
  rmdir "$lock_path" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

available_kib="$(df --output=avail -k "$project_root" | tail -n 1 | tr -d ' ')"
minimum_kib=$((30 * 1024 * 1024))
if [[ "$available_kib" -lt "$minimum_kib" ]]; then
  echo "Refusing G1 cache build: less than 30 GiB free on project filesystem" >&2
  exit 3
fi

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "running" "null"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Public legacy IDs are not valid DICOM UIDs, but are valid file identifiers
# in this competition dataset.  Suppress only that known pydicom warning so
# the managed log preserves actionable cache failures.
export PYTHONWARNINGS="ignore:Invalid value for VR UI:UserWarning"
export PYTHONUNBUFFERED=1
cd "$project_root"
set +e
"$python_bin" scripts/build_mls_2p5d_cache.py
exit_code=$?
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status "completed" 0
else
  write_status "failed" "$exit_code"
fi
exit "$exit_code"
