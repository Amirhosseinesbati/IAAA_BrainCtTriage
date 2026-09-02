#!/usr/bin/env bash
set -Eeuo pipefail

project_root="${IAAA_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage}"
campaign_root="${IAAA_MLS_CAMPAIGN_ROOT:-/workspace/iaaa_artifacts/mls_deploy_aligned_20260902}"
global_gpu_lock="$campaign_root/gpu_training.lock"

mkdir -p "$campaign_root"
if ! mkdir "$global_gpu_lock" 2>/dev/null; then
  echo "Refusing concurrent MLS audit: $global_gpu_lock exists" >&2
  exit 73
fi

cleanup() {
  rmdir "$global_gpu_lock" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Project Python is unavailable: $project_root/.venv/bin/python" >&2
  exit 2
fi
if [[ ! -f "$project_root/scripts/evaluate_mls_three_seed_fold_cuda.py" ]]; then
  echo "Three-seed CUDA evaluator is unavailable under $project_root" >&2
  exit 2
fi

cd "$project_root"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$project_root/.venv/bin/python" \
  "$project_root/scripts/evaluate_mls_three_seed_fold_cuda.py" "$@"
