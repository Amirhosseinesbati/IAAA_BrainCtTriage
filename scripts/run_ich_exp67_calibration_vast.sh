#!/usr/bin/env bash
set -euo pipefail

cd /workspace/project

BASELINE_DIR="reports/ich_experiments/2p5d_segmentation/exp61_effnetv2s_hardempty001_fprvolselect_schema4_calonly_f2"
OUTPUT_DIR="reports/ich_experiments/2p5d_segmentation/exp67_effnetv2s_sahbgiph003_cap8_warm_exp61_schema4_calonly_f2"
MANIFEST="Data/processed/ich_2p5d_schema4/slice_manifest_schema4_server.csv"

test -f "$BASELINE_DIR/best.pth"
test -f "$MANIFEST"
test ! -f "$OUTPUT_DIR/best.pth"

PYTHONPATH=. uv run pytest \
  tests/test_ich_2p5d_segmentation.py \
  tests/test_ich_sah_adapter_update_probe.py \
  tests/test_ich_sah_iph_residual_gate.py \
  -q

PYTHONPATH=. uv run python scripts/train_ich_2p5d_segmentation.py \
  --run-name "ich-exp67-effnetv2s-sahbgiph003-cap8-warm-exp61-schema4-calonly-f2" \
  --output-dir "$OUTPUT_DIR" \
  --manifest-path "$MANIFEST" \
  --architecture unetplusplus \
  --encoder-name tu-efficientnetv2_rw_s \
  --outer-fold 2 \
  --calibration-fold 1 \
  --epochs 6 \
  --batch-size 16 \
  --workers 4 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4 \
  --classification-loss-weight 0 \
  --background-weight 0.15 \
  --empty-foreground-weight 0.05 \
  --empty-foreground-top-fraction 0.001 \
  --checkpoint-selection-strategy fpr_volume_penalized \
  --segmentation-class-weight-power 1 \
  --maximum-segmentation-class-weight 8 \
  --segmentation-class-weight-basis pixel \
  --sampler-study-balance-power 0 \
  --initial-checkpoint "$BASELINE_DIR/best.pth" \
  --sah-residual-adapter \
  --sah-residual-hidden-channels 16 \
  --sah-maximum-logit-residual 8 \
  --sah-include-incumbent-iph \
  --freeze-base-model \
  --sah-tversky-loss-weight 0 \
  --sah-positive-pixel-loss-weight 0.03 \
  --seed 42 \
  --patience 3 \
  --skip-outer-evaluation

PYTHONPATH=. uv run python scripts/evaluate_ich_sah_iph_residual_gate.py \
  --baseline-dir "$BASELINE_DIR" \
  --candidate-dir "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/promotion_gate.json" \
  --notify
