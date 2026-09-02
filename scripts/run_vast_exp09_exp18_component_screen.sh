#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/workspace/IAAA_BrainCtTriage
EXP09_RUN=mls-local-v2-exp09-w32-fold1-hybridsoft-transfer
EXP18_RUN=mls-vast-exp18-w32-fold1-dual-selector-transfer
MLFLOW_RUN_ID=18474f1d10234ca5900caefe3f62c2eb
SECRETS_FILE="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"
EXP09_CHECKPOINT="$PROJECT_DIR/models/checkpoints/mls_multitask/$EXP09_RUN/mls_multitask_epoch_015.pth"
EXP09_AUDIT="$PROJECT_DIR/reports/mls_experiments/$EXP09_RUN/end_to_end_checkpoint_audit"
EXP18_AUDIT="$PROJECT_DIR/reports/mls_experiments/$EXP18_RUN/end_to_end_checkpoint_audit"
EXP18_REPORT="$PROJECT_DIR/reports/mls_experiments/$EXP18_RUN"
SCREEN_DIR="$EXP18_REPORT/crossrun_component_blend_screen_exp09_exp18"
LOG_DIR="/workspace/iaaa_artifacts/logs/$EXP18_RUN/component_screen"
LOG_PATH="$LOG_DIR/component_screen.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

cd "$PROJECT_DIR"
test -x .venv/bin/python
test -f "$EXP09_CHECKPOINT"
test -f "$EXP18_AUDIT/best_objective/study_slice_predictions.csv"
test -f "$EXP18_AUDIT/epoch021/study_slice_predictions.csv"
test -f reports/eda/deep/deep_series_table.csv
test -d Data/raw/training

echo "[$(date -u +%FT%TZ)] Exp09 CUDA re-audit starting"
.venv/bin/python scripts/audit_mls_checkpoints_cuda.py \
  --fold 1 \
  --batch-size 6 \
  --expected-studies 67 \
  --output-root "$EXP09_AUDIT" \
  --candidate "epoch015=$EXP09_CHECKPOINT"

test -f "$EXP09_AUDIT/epoch015/study_slice_predictions.csv"
test -f "$EXP09_AUDIT/epoch015/metrics.json"

echo "[$(date -u +%FT%TZ)] Exp09/Exp18 component screen starting"
.venv/bin/python scripts/screen_mls_crossrun_component_blends.py \
  --baseline "$EXP09_AUDIT/epoch015/study_slice_predictions.csv" \
  --challenger "exp18_epoch21=$EXP18_AUDIT/epoch021/study_slice_predictions.csv" \
  --challenger "exp18_best_objective_epoch12=$EXP18_AUDIT/best_objective/study_slice_predictions.csv" \
  --alpha 0.10 \
  --alpha 0.25 \
  --alpha 0.50 \
  --alpha 0.75 \
  --alpha 1.00 \
  --expected-studies 67 \
  --mae-limit 1.258664866792622 \
  --boundary-floor 0.82 \
  --objective-limit 1.6112072396739778 \
  --expected-baseline-mae 1.258664866792622 \
  --expected-baseline-boundary-f1 0.823728813559322 \
  --expected-baseline-objective 1.6112072396739778 \
  --baseline-parity-tolerance 0.001 \
  --output-dir "$SCREEN_DIR"

test -f "$SCREEN_DIR/crossrun_component_blend_summary.json"
test -f "$SCREEN_DIR/crossrun_component_blend_grid.csv"
test -f "$SCREEN_DIR/CROSSRUN_COMPONENT_BLEND_SCREEN.md"

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

echo "[$(date -u +%FT%TZ)] Uploading allowlisted aggregate screen artifacts"
.venv/bin/python scripts/log_mls_analysis_artifacts.py \
  --run-id "$MLFLOW_RUN_ID" \
  --experiment-dir "$EXP18_REPORT"

echo "[$(date -u +%FT%TZ)] Exp09 CUDA re-audit and Exp18 component screen completed"
