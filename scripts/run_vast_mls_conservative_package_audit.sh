#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/IAAA_BrainCtTriage
RUN=mls-vast-exp20-w32-fold2-dual-selector-thirdfold-replication
OUTPUT="$PROJECT/reports/mls_experiments/$RUN/conservative_package/full_oof_cuda_audit"
LOG_ROOT=/workspace/iaaa_artifacts/logs/$RUN/conservative_package_audit
ARCHIVE=/workspace/iaaa_artifacts/packages/iaaa_brain_ct_triage_mls_conservative_five_20260902.zip
RUNTIME=/workspace/iaaa_artifacts/package_mls_conservative_five_20260902
SPEC="$PROJECT/config/experiments/mls-conservative-threefold-oof-v1.json"
OOF="$PROJECT/reports/mls_experiments/$RUN/conservative_threefold_oof/conservative_threefold_oof_summary.json"

mkdir -p "$OUTPUT" "$LOG_ROOT"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd "$PROJECT"

if ! test -f "$ARCHIVE" || ! test -f "$RUNTIME/mls.py"; then
  echo "Package archive or extracted runtime is missing" >&2
  exit 3
fi
if pgrep -af "train_multitask.py|audit_mls_checkpoints_cuda.py" | grep -v grep >/dev/null; then
  echo "Refusing to overlap MLS training/checkpoint audit" >&2
  exit 4
fi

set +e
.venv/bin/python scripts/audit_mls_conservative_package_cuda.py \
  --runtime-root "$RUNTIME" \
  --archive "$ARCHIVE" \
  --spec "$SPEC" \
  --oof-summary "$OOF" \
  --repo-root "$PROJECT" \
  --data-root "$PROJECT/Data/raw/training" \
  --output-dir "$OUTPUT" \
  --batch-size 6 \
  >"$LOG_ROOT/audit.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$LOG_ROOT/exit_code"
exit "$code"
