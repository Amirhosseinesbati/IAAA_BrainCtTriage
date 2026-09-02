#!/usr/bin/env bash
set -euo pipefail

cd /workspace/project

CHECKPOINT="reports/ich_experiments/2p5d_segmentation/exp61_effnetv2s_hardempty001_fprvolselect_schema4_calonly_f2/best.pth"
MANIFEST="Data/processed/ich_2p5d_schema4/slice_manifest_schema4_server.csv"
OUTPUT_DIR="reports/ich_experiments/2p5d_segmentation/diagnostics/exp68_pre_conditional_subtype_decoder_train_probe_v1"

test -f "$CHECKPOINT"
test -f "$MANIFEST"

PYTHONPATH=. uv run pytest \
  tests/test_ich_conditional_subtype_refiner.py \
  tests/test_ich_2p5d_segmentation.py \
  -q

PYTHONPATH=. uv run python scripts/diagnose_ich_conditional_subtype_refiner.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT_DIR/probe.json" \
  --run-name "ich-exp68-pre-conditional-subtype-decoder-train-probe-v1" \
  --outer-fold 2 \
  --calibration-fold 1 \
  --batch-size 16 \
  --probe-batch-size 16 \
  --workers 4 \
  --seed 42 \
  --epochs 1 \
  --optimizer-steps 0 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --stability-weight 0.25 \
  --conditional-margin 1.0 \
  --notify
