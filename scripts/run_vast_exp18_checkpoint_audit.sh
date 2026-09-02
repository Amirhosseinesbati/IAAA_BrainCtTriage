#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/workspace/IAAA_BrainCtTriage
RUN_NAME=mls-vast-exp18-w32-fold1-dual-selector-transfer
MLFLOW_RUN_ID=18474f1d10234ca5900caefe3f62c2eb
SECRETS_FILE="${IAAA_SECRETS_FILE:-/root/.config/iaaa/secrets.env}"
CHECKPOINT_DIR="$PROJECT_DIR/models/checkpoints/mls_multitask/$RUN_NAME"
REPORT_ROOT="$PROJECT_DIR/reports/mls_experiments/$RUN_NAME"
OUTPUT_ROOT="$REPORT_ROOT/end_to_end_checkpoint_audit"
POOLING_ROOT="$REPORT_ROOT/checkpoint_pooling_expanded"
AUDIT_LOG_DIR="/workspace/iaaa_artifacts/logs/$RUN_NAME/audit"
AUDIT_LOG="$AUDIT_LOG_DIR/audit.log"

mkdir -p "$AUDIT_LOG_DIR"
exec > >(tee -a "$AUDIT_LOG") 2>&1

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

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

cd "$PROJECT_DIR"
test -x .venv/bin/python
test -f "$REPORT_ROOT/epoch_metrics.jsonl"
test -f "$CHECKPOINT_DIR/mls_multitask_best.pth"
test -f "$CHECKPOINT_DIR/mls_multitask_best_mae.pth"
test -f "$CHECKPOINT_DIR/mls_multitask_best_selector_auc.pth"
test -f "$CHECKPOINT_DIR/mls_multitask_best_peak_auc.pth"
test -f "$CHECKPOINT_DIR/mls_multitask_best_study_boundary.pth"
test -f "$CHECKPOINT_DIR/mls_multitask_best_study.pth"

echo "[$(date -u +%FT%TZ)] Exp18 CUDA checkpoint audit starting"
.venv/bin/python scripts/audit_mls_checkpoints_cuda.py \
  --fold 1 \
  --batch-size 6 \
  --expected-studies 67 \
  --output-root "$OUTPUT_ROOT" \
  --candidate "best_objective=$CHECKPOINT_DIR/mls_multitask_best.pth" \
  --candidate "best_mae=$CHECKPOINT_DIR/mls_multitask_best_mae.pth" \
  --candidate "best_selector_auc=$CHECKPOINT_DIR/mls_multitask_best_selector_auc.pth" \
  --candidate "best_peak_auc=$CHECKPOINT_DIR/mls_multitask_best_peak_auc.pth" \
  --candidate "best_study_boundary=$CHECKPOINT_DIR/mls_multitask_best_study_boundary.pth" \
  --candidate "best_study=$CHECKPOINT_DIR/mls_multitask_best_study.pth" \
  --candidate "epoch013=$CHECKPOINT_DIR/mls_multitask_epoch_013.pth" \
  --candidate "epoch015=$CHECKPOINT_DIR/mls_multitask_epoch_015.pth" \
  --candidate "epoch017=$CHECKPOINT_DIR/mls_multitask_epoch_017.pth" \
  --candidate "epoch019=$CHECKPOINT_DIR/mls_multitask_epoch_019.pth" \
  --candidate "epoch021=$CHECKPOINT_DIR/mls_multitask_epoch_021.pth" \
  --candidate "epoch023=$CHECKPOINT_DIR/mls_multitask_epoch_023.pth"

echo "[$(date -u +%FT%TZ)] Exp18 pooling grid starting"
.venv/bin/python scripts/search_mls_checkpoint_pooling.py \
  --output-dir "$POOLING_ROOT" \
  --candidate "best_objective=$OUTPUT_ROOT/best_objective/study_slice_predictions.csv" \
  --candidate "best_mae=$OUTPUT_ROOT/best_mae/study_slice_predictions.csv" \
  --candidate "best_selector_auc=$OUTPUT_ROOT/best_selector_auc/study_slice_predictions.csv" \
  --candidate "best_peak_auc=$OUTPUT_ROOT/best_peak_auc/study_slice_predictions.csv" \
  --candidate "best_study_boundary=$OUTPUT_ROOT/best_study_boundary/study_slice_predictions.csv" \
  --candidate "best_study=$OUTPUT_ROOT/best_study/study_slice_predictions.csv" \
  --candidate "epoch013=$OUTPUT_ROOT/epoch013/study_slice_predictions.csv" \
  --candidate "epoch015=$OUTPUT_ROOT/epoch015/study_slice_predictions.csv" \
  --candidate "epoch017=$OUTPUT_ROOT/epoch017/study_slice_predictions.csv" \
  --candidate "epoch019=$OUTPUT_ROOT/epoch019/study_slice_predictions.csv" \
  --candidate "epoch021=$OUTPUT_ROOT/epoch021/study_slice_predictions.csv" \
  --candidate "epoch023=$OUTPUT_ROOT/epoch023/study_slice_predictions.csv"

echo "[$(date -u +%FT%TZ)] Exp18 locked promotion gate starting"
.venv/bin/python scripts/select_mls_locked_promotion.py \
  --run-name "$RUN_NAME" \
  --mlflow-run-id "$MLFLOW_RUN_ID" \
  --fold 1 \
  --grid "$POOLING_ROOT/checkpoint_pooling_grid.csv" \
  --grid-summary "$POOLING_ROOT/checkpoint_pooling_summary.json" \
  --audit-status "$OUTPUT_ROOT/audit_status.json" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epoch-history "$REPORT_ROOT/epoch_metrics.jsonl" \
  --output "$REPORT_ROOT/promotion_gate.json" \
  --expected-studies 67 \
  --reference-run mls-local-v2-exp09-w32-fold1-hybridsoft-transfer \
  --reference-checkpoint epoch15 \
  --mae-limit 1.258664866792622 \
  --boundary-floor 0.82 \
  --objective-limit 1.6112072396739778

echo "[$(date -u +%FT%TZ)] Uploading aggregate Exp18 analysis artifacts to MLflow"
.venv/bin/python scripts/log_mls_analysis_artifacts.py \
  --run-id "$MLFLOW_RUN_ID" \
  --experiment-dir "$REPORT_ROOT"

echo "[$(date -u +%FT%TZ)] Exp18 CUDA audit, pooling and locked gate completed"
