#!/usr/bin/env bash
set -euo pipefail

cd /workspace/project

CHECKPOINT="reports/ich_experiments/2p5d_segmentation/exp61_effnetv2s_hardempty001_fprvolselect_schema4_calonly_f2/best.pth"
MANIFEST="Data/processed/ich_2p5d_schema4/slice_manifest_schema4_server.csv"
OUTPUT_DIR="reports/ich_experiments/2p5d_segmentation/diagnostics/exp67_pre_iph_support_update_epoch1_v1"

test -f "$CHECKPOINT"
test -f "$MANIFEST"

PYTHONPATH=. uv run pytest \
  tests/test_ich_sah_adapter_update_probe.py \
  tests/test_ich_2p5d_segmentation.py \
  -q

PYTHONPATH=. uv run python scripts/diagnose_ich_sah_adapter_updates.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT_DIR/update_probe.json" \
  --run-name "ich-exp67-pre-iph-support-update-epoch1-v1" \
  --outer-fold 2 \
  --calibration-fold 1 \
  --batch-size 16 \
  --probe-batch-size 4 \
  --workers 4 \
  --seed 42 \
  --optimizer-steps 0 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4 \
  --sah-tversky-weight 0 \
  --sah-positive-pixel-weight 0.03 \
  --hidden-channels 16 \
  --maximum-logit-residual 8 \
  --probe-positive-batches 12 \
  --probe-iph-control-batches 12 \
  --probe-negative-batches 12 \
  --maximum-probe-scanned-batches 240 \
  --background-sample-stride 512 \
  --include-incumbent-iph \
  --notify
