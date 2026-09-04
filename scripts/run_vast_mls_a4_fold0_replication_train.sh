#!/usr/bin/env bash
# Launch exactly one of the two preregistered A4 fold-0 seed replications.
#
# This is deliberately not a general MLS launcher.  It can materialize only
# seeds 2026/3407, requires the passed seed-42 resource decision, and makes the
# seed and run name the only textual changes to the locked A4 manifest.
set -Eeuo pipefail

if [[ "$#" -ne 1 || ( "$1" != "2026" && "$1" != "3407" ) ]]; then
  echo "Usage: $0 {2026|3407}" >&2
  exit 64
fi

seed="$1"
project_root="${IAAA_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage_mls_da}"
campaign_root="${IAAA_MLS_CAMPAIGN_ROOT:-/workspace/iaaa_artifacts/mls_deploy_aligned_20260902}"
report_root="$project_root/reports/mls_experiments/mls-deploy-aligned-upgrade-20260902"
template="$project_root/config/experiments/mls-vast-deploy-aligned-a4-pair-rank-template.yaml"
resource_decision="$report_root/A4_FOLD0_SEED42_RESOURCE_SCREEN_DECISION.json"
preregistration="$report_root/FOLD0_A4_SEED_REPLICATION_PREREGISTRATION.json"
run_name="mls-vast-da-a4-pair-rank-fold0-seed${seed}"
artifact_root="$campaign_root/$run_name"
status_path="$artifact_root/status.json"
log_path="$artifact_root/train.log"
manifest_snapshot="$artifact_root/training_manifest_used.yaml"
checkpoint_dir="$project_root/models/checkpoints/mls_multitask/$run_name"
global_gpu_lock="$campaign_root/gpu_training.lock"
secrets_file="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"
core_implementation_commit="6ddd738244cc8b5d702235e64c88b1c8608a93f3"

write_status() {
  local state="$1"
  local exit_code="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema_version":1,"state":"%s","exit_code":%s,"run_name":"%s","seed":%s,"manifest":"%s","manifest_sha256":"%s","template_sha256":"%s","resource_decision_sha256":"%s","preregistration_sha256":"%s","core_implementation_commit":"%s","log_path":"%s","started_utc":"%s","finished_utc":"%s","compute_policy":"cuda_only_no_cpu_fallback","auto_destroy":false}\n' \
    "$state" "$exit_code" "$run_name" "$seed" "$manifest_snapshot" \
    "${manifest_sha256:-}" "${template_sha256:-}" "${resource_decision_sha256:-}" \
    "${preregistration_sha256:-}" "$core_implementation_commit" "$log_path" \
    "${started_utc:-}" "$timestamp" >"$status_path.tmp"
  mv -f "$status_path.tmp" "$status_path"
}

mkdir -p "$campaign_root"
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  echo "Project Python is unavailable: $project_root/.venv/bin/python" >&2
  exit 2
fi
if [[ ! -f "$template" || ! -f "$resource_decision" || ! -f "$preregistration" ]]; then
  echo "A4 template, resource decision, or replication preregistration is unavailable" >&2
  exit 3
fi
if [[ -e "$artifact_root" || -e "$checkpoint_dir" ]]; then
  echo "Refusing to overwrite an existing A4 replication artifact or checkpoint directory" >&2
  exit 4
fi
if [[ ! -f "$secrets_file" || "$(stat -c '%a' "$secrets_file")" != "600" ]]; then
  echo "Root-only MLflow secrets file is unavailable or has unsafe permissions" >&2
  exit 5
fi
available_kib="$(df --output=avail -k "$project_root" | tail -n 1 | tr -d ' ')"
minimum_kib=$((20 * 1024 * 1024))
if [[ "$available_kib" -lt "$minimum_kib" ]]; then
  echo "Refusing A4 replication: less than 20 GiB free on project filesystem" >&2
  exit 6
fi

template_sha256="$(sha256sum "$template" | awk '{print $1}')"
resource_decision_sha256="$(sha256sum "$resource_decision" | awk '{print $1}')"
preregistration_sha256="$(sha256sum "$preregistration" | awk '{print $1}')"

if ! "$project_root/.venv/bin/python" - "$seed" "$template" "$resource_decision" "$preregistration" "$template_sha256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

seed, template_path, decision_path, prereg_path, expected_template_sha = sys.argv[1:]
if seed not in {"2026", "3407"}:
    raise SystemExit("Only the preregistered A4 replication seeds are permitted")

def load(path: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unreadable A4 JSON contract: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"A4 JSON contract is not an object: {path}")
    return value

decision = load(decision_path)
prereg = load(prereg_path)
checks = {
    "decision_status": decision.get("status") == "passed_for_two_remaining_fold0_seed_replications",
    "decision_authorization": decision.get("can_start_only_seeds_2026_and_3407_on_fold0") is True,
    "decision_candidate": decision.get("candidate") == "mls-vast-deploy-aligned-a4-pair-rank",
    "decision_scope": decision.get("screen_scope") == "a4_fold0_seed42_resource_screen_only",
    "decision_fold": decision.get("fold") == 0,
    "decision_studies": decision.get("studies") == 70,
    "decision_epoch": decision.get("fixed_epoch") == 15,
    "decision_compute": decision.get("compute_policy") == "cuda_only_no_cpu_fallback",
    "decision_no_failed_gate": decision.get("failed_gates") == [],
    "prereg_locked": prereg.get("status") == "locked_before_a4_seed42_resource_outcome",
    "prereg_seed": int(seed) in prereg.get("allowed_replication_seeds", []),
    "prereg_fold": prereg.get("fold") == 0,
    "prereg_epoch": prereg.get("fixed_epoch") == 15,
    "prereg_template": prereg.get("template_sha256") == expected_template_sha,
    "prereg_seed_only": prereg.get("only_allowed_manifest_changes") == ["run_name", "training_config.seed"],
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"A4 replication contract refused: {failed}")
PY
then
  echo "A4 seed replication is not authorized by the immutable resource contract" >&2
  exit 7
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
  exit 8
fi

mkdir -p "$artifact_root"
"$project_root/.venv/bin/python" - "$template" "$manifest_snapshot" "$seed" "$run_name" <<'PY'
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
seed = int(sys.argv[3])
run_name = sys.argv[4]
template = template_path.read_text(encoding="utf-8")
expected_name = "mls-vast-da-a4-pair-rank-fold0-seed42"
expected_seed = "  seed: 42\n"
if template.count(f"run_name: {expected_name}") != 1:
    raise SystemExit("A4 template does not contain exactly one seed-42 run name")
if template.count(expected_seed) != 1:
    raise SystemExit("A4 template does not contain exactly one training seed")
materialized = template.replace(
    f"run_name: {expected_name}", f"run_name: {run_name}", 1
).replace(expected_seed, f"  seed: {seed}\n", 1)
if materialized.count(f"run_name: {run_name}") != 1 or materialized.count(f"  seed: {seed}\n") != 1:
    raise SystemExit("A4 replication manifest materialization was not exact")
output_path.write_text(materialized, encoding="utf-8")
PY
manifest_sha256="$(sha256sum "$manifest_snapshot" | awk '{print $1}')"

"$project_root/.venv/bin/python" - "$manifest_snapshot" "$seed" "$run_name" <<'PY'
import sys
from pathlib import Path

import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_seed = int(sys.argv[2])
expected_run_name = sys.argv[3]
config = payload.get("training_config", {})
checks = {
    "task": payload.get("task") == "mls",
    "strategy": payload.get("strategy") == "mls_heatmap",
    "run_name": payload.get("run_name") == expected_run_name,
    "fold": config.get("fold") == 0,
    "seed": config.get("seed") == expected_seed,
    "epoch": config.get("snapshot_start_epoch") == 15,
    "pair_rank": config.get("within_study_rank_loss_weight") == 0.10,
    "study_bag_disabled": config.get("study_bag_loss_weight") == 0.0,
    "cuda_policy": payload.get("tags", {}).get("compute_policy") == "cuda_only_no_cpu_fallback",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Materialized A4 replication manifest failed validation: {failed}")
PY

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
  --allow-training >"$log_path" 2>&1
exit_code=$?
set -e
if [[ "$exit_code" -eq 0 ]]; then
  write_status "completed" 0
else
  write_status "failed" "$exit_code"
fi
exit "$exit_code"
