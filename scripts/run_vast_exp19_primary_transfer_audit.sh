#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/workspace/IAAA_BrainCtTriage
RUN_NAME=mls-vast-exp19-w32-fold0-dual-selector-replication
BASELINE_RUN=mls-vast-exp16-w32-fold0-strict-ensemble-refresh
MLFLOW_RUN_ID=5383a78d31bf4a79a5bf6aff3c086e8c
TRAIN_SESSION=mls_exp19_fold0_dual
SECRETS_FILE="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"

CHECKPOINT_DIR="$PROJECT_DIR/models/checkpoints/mls_multitask/$RUN_NAME"
REPORT_ROOT="$PROJECT_DIR/reports/mls_experiments/$RUN_NAME"
TRAIN_STATUS="/workspace/iaaa_artifacts/logs/$RUN_NAME/status.json"
EPOCH_HISTORY="$REPORT_ROOT/epoch_metrics.jsonl"
BASELINE_PREDICTIONS="$PROJECT_DIR/reports/mls_experiments/$BASELINE_RUN/end_to_end_checkpoint_audit/best_selector_auc/study_slice_predictions.csv"
AUDIT_ROOT="$REPORT_ROOT/primary_epoch21_cuda_audit"
CHALLENGER_PREDICTIONS="$AUDIT_ROOT/epoch021/study_slice_predictions.csv"
TRANSFER_ROOT="$REPORT_ROOT/primary_epoch21_component_transfer"
LOG_DIR="/workspace/iaaa_artifacts/logs/$RUN_NAME/primary_audit"
LOG_PATH="$LOG_DIR/audit.log"
STATUS_PATH="$LOG_DIR/status.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

cd "$PROJECT_DIR"
test -x .venv/bin/python

FINALIZED=0
write_status() {
  local state="$1"
  local evaluation_exit_code="$2"
  local decision="$3"
  .venv/bin/python - "$STATUS_PATH" "$state" "$evaluation_exit_code" "$decision" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
state = sys.argv[2]
exit_code = int(sys.argv[3])
decision = sys.argv[4]
previous = {}
if path.is_file():
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
now = datetime.now(timezone.utc).isoformat()
payload = {
    "schema_version": 1,
    "run_name": "mls-vast-exp19-w32-fold0-dual-selector-replication",
    "state": state,
    "compute_policy": "cuda_only_model_inference_cpu_aggregate_postprocessing",
    "primary_checkpoint": "epoch021",
    "expected_studies": 70,
    "started_utc": previous.get("started_utc", now),
    "updated_utc": now,
    "finished_utc": now if state != "running" else None,
    "evaluation_exit_code": exit_code,
    "decision": decision,
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

on_exit() {
  local exit_code=$?
  if [[ "$FINALIZED" -eq 0 ]]; then
    FINALIZED=1
    write_status "operational_failed" "$exit_code" "Primary audit runner failed before producing a complete frozen decision." || true
  fi
}
trap on_exit EXIT

write_status "running" -1 "Preflight validation is running."

test -f "$TRAIN_STATUS"
test -f "$EPOCH_HISTORY"
test -f "$BASELINE_PREDICTIONS"
test -f "$CHECKPOINT_DIR/mls_multitask_epoch_021.pth"
test -f reports/eda/deep/deep_series_table.csv
test -d Data/raw/training

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "Training session $TRAIN_SESSION is still active; refusing overlapping audit." >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] Validating completed training contract"
.venv/bin/python - "$TRAIN_STATUS" "$EPOCH_HISTORY" <<'PY'
import json
import math
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
history_path = Path(sys.argv[2])
status = json.loads(status_path.read_text(encoding="utf-8"))
if status.get("state") != "completed" or int(status.get("exit_code", -1)) != 0:
    raise RuntimeError(f"training is not completed successfully: {status}")
if status.get("compute_policy") != "cuda_only":
    raise RuntimeError(f"unexpected training compute policy: {status.get('compute_policy')!r}")

rows = [
    json.loads(line)
    for line in history_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
epochs = [int(float(row["epoch"])) for row in rows]
if epochs != list(range(1, 24)):
    raise RuntimeError(f"expected exact epochs 1..23, found {epochs}")
for row in rows:
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite epoch metric: epoch={row['epoch']} {key}={value}")
print(json.dumps({
    "training_state": status["state"],
    "training_exit_code": status["exit_code"],
    "compute_policy": status["compute_policy"],
    "epochs": len(rows),
    "last_epoch": epochs[-1],
}, indent=2))
PY

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing root-only secrets file: $SECRETS_FILE" >&2
  exit 1
fi
if [[ "$(stat -c '%a' "$SECRETS_FILE")" != "600" ]]; then
  echo "Secrets file permissions must be 600: $SECRETS_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

echo "[$(date -u +%FT%TZ)] Frozen Exp19 epoch21 CUDA audit starting"
sha256sum "$CHECKPOINT_DIR/mls_multitask_epoch_021.pth"
.venv/bin/python scripts/audit_mls_checkpoints_cuda.py \
  --fold 0 \
  --batch-size 6 \
  --expected-studies 70 \
  --output-root "$AUDIT_ROOT" \
  --candidate "epoch021=$CHECKPOINT_DIR/mls_multitask_epoch_021.pth"

test -f "$AUDIT_ROOT/audit_status.json"
test -f "$AUDIT_ROOT/epoch021/metrics.json"
test -f "$CHALLENGER_PREDICTIONS"

echo "[$(date -u +%FT%TZ)] Frozen fold0 regression-only transfer gate starting"
set +e
.venv/bin/python scripts/evaluate_mls_fixed_component_transfer.py \
  --baseline "$BASELINE_PREDICTIONS" \
  --baseline-label exp16_best_selector_auc_epoch16 \
  --challenger "$CHALLENGER_PREDICTIONS" \
  --challenger-label exp19_epoch21 \
  --component-mode regression_only \
  --alpha 0.10 \
  --expected-studies 70 \
  --mae-limit 1.604477701016835 \
  --boundary-floor 0.8273325590398761 \
  --objective-limit 1.9398125829370827 \
  --expected-baseline-mae 1.604477701016835 \
  --expected-baseline-boundary-f1 0.8273325590398761 \
  --expected-baseline-objective 1.9498125829370827 \
  --baseline-parity-tolerance 1e-9 \
  --output-dir "$TRANSFER_ROOT"
EVALUATION_EXIT_CODE=$?
set -e

if [[ "$EVALUATION_EXIT_CODE" -ne 0 && "$EVALUATION_EXIT_CODE" -ne 2 ]]; then
  echo "Unexpected frozen evaluator exit code: $EVALUATION_EXIT_CODE" >&2
  exit "$EVALUATION_EXIT_CODE"
fi
test -f "$TRANSFER_ROOT/fixed_component_transfer_summary.json"
test -f "$TRANSFER_ROOT/FROZEN_COMPONENT_TRANSFER_REPORT.md"

echo "[$(date -u +%FT%TZ)] Uploading allowlisted aggregate Exp19 artifacts to MLflow"
.venv/bin/python scripts/log_mls_analysis_artifacts.py \
  --run-id "$MLFLOW_RUN_ID" \
  --experiment-dir "$REPORT_ROOT"

if [[ "$EVALUATION_EXIT_CODE" -eq 0 ]]; then
  DECISION="Frozen independent-fold component transfer passed all primary gates."
else
  DECISION="Frozen independent-fold component transfer completed and failed at least one primary gate."
fi
write_status "completed" "$EVALUATION_EXIT_CODE" "$DECISION"
FINALIZED=1
trap - EXIT

echo "[$(date -u +%FT%TZ)] Exp19 primary audit completed: $DECISION"
exit "$EVALUATION_EXIT_CODE"
