#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/workspace/IAAA_BrainCtTriage
RUN_NAME=mls-vast-exp20-w32-fold2-dual-selector-thirdfold-replication
BASELINE_RUN=mls-vast-exp15r-w32-fold2-strict-repro-control
MLFLOW_RUN_ID=aa4d88acea4246a8a7e5c27a0a33a6c6
TRAIN_SESSION=mls_exp20_fold2_dual
SECRETS_FILE="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"

CHECKPOINT_DIR="$PROJECT_DIR/models/checkpoints/mls_multitask/$RUN_NAME"
REPORT_ROOT="$PROJECT_DIR/reports/mls_experiments/$RUN_NAME"
TRAIN_STATUS="/workspace/iaaa_artifacts/logs/$RUN_NAME/status.json"
BASELINE_PREDICTIONS="$PROJECT_DIR/reports/mls_experiments/$BASELINE_RUN/end_to_end_checkpoint_audit/epoch017/study_slice_predictions.csv"
AUDIT_ROOT="$REPORT_ROOT/postfailure_named_best_cuda_audit"
CHALLENGER_PREDICTIONS="$AUDIT_ROOT/best_objective/study_slice_predictions.csv"
TRANSFER_ROOT="$REPORT_ROOT/postfailure_named_best_component_transfer"
LOG_DIR="/workspace/iaaa_artifacts/logs/$RUN_NAME/named_best_diagnostic"
LOG_PATH="$LOG_DIR/diagnostic.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

cd "$PROJECT_DIR"
test -x .venv/bin/python
test -f "$TRAIN_STATUS"
test "$(jq -r .state "$TRAIN_STATUS")" = completed
test "$(jq -r .exit_code "$TRAIN_STATUS")" = 0
if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "Training session $TRAIN_SESSION is still active; refusing overlap." >&2
  exit 1
fi
test -f "$CHECKPOINT_DIR/mls_multitask_best.pth"
test -f "$BASELINE_PREDICTIONS"

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

echo "[$(date -u +%FT%TZ)] Exp20 named-best CUDA diagnostic starting"
sha256sum "$CHECKPOINT_DIR/mls_multitask_best.pth"
.venv/bin/python scripts/audit_mls_checkpoints_cuda.py \
  --fold 2 \
  --batch-size 6 \
  --expected-studies 67 \
  --output-root "$AUDIT_ROOT" \
  --candidate "best_objective=$CHECKPOINT_DIR/mls_multitask_best.pth"

test -f "$AUDIT_ROOT/audit_status.json"
test -f "$AUDIT_ROOT/best_objective/metrics.json"
test -f "$CHALLENGER_PREDICTIONS"

echo "[$(date -u +%FT%TZ)] Exp20 named-best fixed transfer diagnostic starting"
set +e
.venv/bin/python scripts/evaluate_mls_fixed_component_transfer.py \
  --baseline "$BASELINE_PREDICTIONS" \
  --baseline-label exp15r_epoch17 \
  --challenger "$CHALLENGER_PREDICTIONS" \
  --challenger-label exp20_named_best_epoch11 \
  --component-mode regression_only \
  --alpha 0.10 \
  --expected-studies 67 \
  --mae-limit 1.5483543317709396 \
  --boundary-floor 0.8925925925925926 \
  --objective-limit 1.7531691465857544 \
  --expected-baseline-mae 1.5483543317709396 \
  --expected-baseline-boundary-f1 0.8925925925925926 \
  --expected-baseline-objective 1.7631691465857544 \
  --baseline-parity-tolerance 1e-9 \
  --output-dir "$TRANSFER_ROOT"
EVALUATION_EXIT_CODE=$?
set -e

if [[ "$EVALUATION_EXIT_CODE" -ne 0 && "$EVALUATION_EXIT_CODE" -ne 2 ]]; then
  echo "Unexpected evaluator exit code: $EVALUATION_EXIT_CODE" >&2
  exit "$EVALUATION_EXIT_CODE"
fi
test -f "$TRANSFER_ROOT/fixed_component_transfer_summary.json"
test -f "$TRANSFER_ROOT/FROZEN_COMPONENT_TRANSFER_REPORT.md"

echo "[$(date -u +%FT%TZ)] Uploading named-best aggregate diagnostic artifacts"
.venv/bin/python scripts/log_mls_analysis_artifacts.py \
  --run-id "$MLFLOW_RUN_ID" \
  --experiment-dir "$REPORT_ROOT"

if [[ "$EVALUATION_EXIT_CODE" -eq 0 ]]; then
  echo "[$(date -u +%FT%TZ)] Secondary named-best diagnostic passed its numerical gates; primary remains failed."
else
  echo "[$(date -u +%FT%TZ)] Secondary named-best diagnostic failed; checkpoint diagnostics stop."
fi
exit "$EVALUATION_EXIT_CODE"
