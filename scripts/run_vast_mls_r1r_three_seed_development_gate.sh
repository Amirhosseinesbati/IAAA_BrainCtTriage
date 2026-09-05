#!/usr/bin/env bash
# Evaluate the sealed R1R2 fold-1 ensemble, then its development-only triage gate.
#
# Usage:
#   run_vast_mls_r1r_three_seed_development_gate.sh CONTRACT CONTRACT_SHA CAMPAIGN_ROOT
#
# The contract is mandatory so this script cannot silently substitute a model,
# checkpoint, truth table, frozen branch, seed or threshold. It starts no
# training job and performs GPU work only in the two raw-DICOM CUDA audits.
set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 CONTRACT CONTRACT_SHA CAMPAIGN_ROOT" >&2
  exit 2
fi

contract="$1"
contract_sha="$2"
campaign_root="$3"
python_bin="${IAAA_R1R_PYTHON:-/workspace/IAAA_BrainCtTriage_mls_da/.venv/bin/python}"
global_gpu_lock="$campaign_root/gpu_training.lock"
status_path="$campaign_root/three_seed_development_gate_status.json"
control_audit="$campaign_root/control_fold1_three_seed_audit"
candidate_audit="$campaign_root/candidate_fold1_three_seed_audit"
triage_output="$campaign_root/fold1_development_triage"

write_status() {
  local state="$1"
  local exit_code="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema_version":2,"state":"%s","exit_code":%s,"contract":"%s","contract_sha256":"%s","updated_utc":"%s","promotion_eligible":false,"submission_zip_allowed":false}\n' \
    "$state" "$exit_code" "$contract" "$contract_sha" "$timestamp" >"$status_path.tmp"
  mv -f "$status_path.tmp" "$status_path"
}

mkdir -p "$campaign_root"
terminal_status_written=0
lock_acquired=0
cleanup_lock() {
  if [[ "$lock_acquired" -eq 1 ]]; then
    rmdir "$global_gpu_lock" 2>/dev/null || true
    lock_acquired=0
  fi
}
on_exit() {
  local exit_code=$?
  cleanup_lock
  if [[ "$terminal_status_written" -eq 0 ]]; then
    write_status "failed" "$exit_code" || true
  fi
  return "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "$python_bin" || ! -f "$contract" ]]; then
  echo "R1R2 Python or contract is unavailable" >&2
  exit 3
fi
read_contract_path() {
  jq -er "$1" "$contract"
}
contract_project_root="$(read_contract_path '.training_source.root')"
project_root="$(readlink -f "$contract_project_root")"
if [[ -n "${IAAA_PROJECT_ROOT:-}" && "$(readlink -f "$IAAA_PROJECT_ROOT")" != "$project_root" ]]; then
  echo "IAAA_PROJECT_ROOT differs from the R1R2-sealed source root" >&2
  exit 3
fi
project_config="$(read_contract_path '.data.project_config')"
project_config="$(readlink -f "$project_config")"
if [[ ! -f "$project_config" ]]; then
  echo "R1R2 sealed project config is unavailable: $project_config" >&2
  exit 3
fi
export IAAA_CONFIG_PATH="$project_config"
for required in \
  "$project_root/scripts/validate_mls_r1_replication_matrix.py" \
  "$project_root/scripts/evaluate_mls_three_seed_fold_cuda.py" \
  "$project_root/scripts/evaluate_mls_r1r_fold1_development_triage.py"; do
  if [[ ! -f "$required" ]]; then
    echo "Required R1R2 program is missing: $required" >&2
    exit 3
  fi
done
if [[ -e "$triage_output" ]]; then
  echo "Refusing to overwrite R1R2 development triage output: $triage_output" >&2
  exit 4
fi

cd "$project_root"
"$python_bin" scripts/validate_mls_r1_replication_matrix.py \
  --contract "$contract" --contract-sha256 "$contract_sha" --require-checkpoints

if ! mkdir "$global_gpu_lock" 2>/dev/null; then
  echo "Refusing concurrent R1R2 CUDA audit: $global_gpu_lock exists" >&2
  exit 73
fi
lock_acquired=1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for a CUDA-only R1R2 audit" >&2
  exit 74
fi
active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d')"
if [[ -n "$active_gpu_pids" ]]; then
  echo "Refusing R1R CUDA audit while GPU is in use: $active_gpu_pids" >&2
  exit 74
fi

truth_table="$(read_contract_path '.data.truth_table')"
fold_manifest="$(read_contract_path '.data.fold_manifest')"
data_root="$(read_contract_path '.data.raw_dicom.resolved_root')"
audit_batch_size="$(read_contract_path '.protocol.cuda_audit_batch_size')"
control_seed42="$(read_contract_path '.members.control.seed42.checkpoint_path')"
control_seed2026="$(read_contract_path '.members.control.seed2026.expected_checkpoint_path')"
control_seed3407="$(read_contract_path '.members.control.seed3407.expected_checkpoint_path')"
candidate_seed42="$(read_contract_path '.members.candidate.seed42.checkpoint_path')"
candidate_seed2026="$(read_contract_path '.members.candidate.seed2026.expected_checkpoint_path')"
candidate_seed3407="$(read_contract_path '.members.candidate.seed3407.expected_checkpoint_path')"

for required in "$data_root" "$truth_table" "$fold_manifest" "$project_config" \
  "$control_seed42" "$control_seed2026" "$control_seed3407" \
  "$candidate_seed42" "$candidate_seed2026" "$candidate_seed3407"; do
  if [[ ! -e "$required" ]]; then
  echo "R1R2 gate input is missing: $required" >&2
    exit 5
  fi
done

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
write_status "running" "null"

"$python_bin" scripts/evaluate_mls_three_seed_fold_cuda.py \
  --checkpoint "seed42=$control_seed42" \
  --checkpoint "seed2026=$control_seed2026" \
  --checkpoint "seed3407=$control_seed3407" \
  --fold 1 --fixed-epoch 15 --expected-studies 67 --batch-size "$audit_batch_size" \
  --data-root "$data_root" --fold-manifest "$fold_manifest" --truth-table "$truth_table" --output-dir "$control_audit"

"$python_bin" scripts/evaluate_mls_three_seed_fold_cuda.py \
  --checkpoint "seed42=$candidate_seed42" \
  --checkpoint "seed2026=$candidate_seed2026" \
  --checkpoint "seed3407=$candidate_seed3407" \
  --fold 1 --fixed-epoch 15 --expected-studies 67 --batch-size "$audit_batch_size" \
  --data-root "$data_root" --fold-manifest "$fold_manifest" --truth-table "$truth_table" --output-dir "$candidate_audit"

"$python_bin" scripts/evaluate_mls_r1r_fold1_development_triage.py \
  --contract "$contract" --contract-sha256 "$contract_sha" \
  --control-summary "$control_audit/aggregate_summary.json" \
  --control-private "$control_audit/study_member_predictions_private.csv" \
  --candidate-summary "$candidate_audit/aggregate_summary.json" \
  --candidate-private "$candidate_audit/study_member_predictions_private.csv" \
  --output-dir "$triage_output"

write_status "completed" 0
terminal_status_written=1
