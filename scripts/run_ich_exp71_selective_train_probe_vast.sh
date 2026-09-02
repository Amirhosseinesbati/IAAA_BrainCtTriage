#!/usr/bin/env bash
set -euo pipefail

cd /workspace/project

CHECKPOINT="reports/ich_experiments/2p5d_segmentation/exp61_effnetv2s_hardempty001_fprvolselect_schema4_calonly_f2/best.pth"
MANIFEST="Data/processed/ich_2p5d_schema4/slice_manifest_schema4_server.csv"
OUTPUT_DIR="reports/ich_experiments/2p5d_segmentation/diagnostics/exp71_selective_gated_residual_train_probe_v1"

test -f "$CHECKPOINT"
test -f "$MANIFEST"

PYTHONPATH=. uv run pytest \
  tests/test_ich_conditional_subtype_selective.py \
  tests/test_ich_conditional_subtype_residual.py \
  tests/test_ich_conditional_subtype_refiner.py \
  tests/test_ich_2p5d_segmentation.py \
  -q

PYTHONPATH=. uv run python scripts/diagnose_ich_conditional_subtype_selective.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT_DIR/probe.json" \
  --run-name "ich-exp71-selective-gated-residual-train-probe-v1" \
  --outer-fold 2 \
  --calibration-fold 1 \
  --batch-size 16 \
  --probe-batch-size 16 \
  --workers 4 \
  --seed 42 \
  --epochs 1 \
  --optimizer-steps 0 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4 \
  --correction-weight 4.0 \
  --stability-weight 1.0 \
  --gate-weight 0.25 \
  --gate-positive-weight 200.0 \
  --gate-threshold 0.5 \
  --initial-gate-probability 0.01 \
  --class-weight-power 0.25 \
  --maximum-class-weight 2.0 \
  --hidden-channels 16 \
  --maximum-logit-residual 4.0 \
  --conditional-margin 1.0 \
  --notify
