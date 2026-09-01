#!/usr/bin/env bash
set -euo pipefail

cd /workspace/project

BASE_DIR="reports/ich_experiments/2p5d_segmentation/exp61_effnetv2s_hardempty001_fprvolselect_schema4_calonly_f2"
SMOKE_DIR="reports/ich_experiments/2p5d_segmentation/exp65_smoke_effnetv2s_sahbgexpand003_warm_exp61_schema4_f2"
FULL_DIR="reports/ich_experiments/2p5d_segmentation/exp65_effnetv2s_sahbgexpand003_warm_exp61_schema4_calonly_f2"
MANIFEST="Data/processed/ich_2p5d_schema4/slice_manifest_schema4_server.csv"
EXPECTED_CHECKPOINT_SHA="5f304018340e88e1d858d0842e432d8a163c96ed26117796c7068b6d399c18d4"
EXPECTED_MANIFEST_SHA="0455e6d24590a652b324c58730d81750d675b8c8d2442e67c12f11c16531ec37"

checkpoint_sha="$(sha256sum "$BASE_DIR/best.pth" | cut -d' ' -f1)"
manifest_sha="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
[[ "$checkpoint_sha" == "$EXPECTED_CHECKPOINT_SHA" ]]
[[ "$manifest_sha" == "$EXPECTED_MANIFEST_SHA" ]]
[[ ! -e "$SMOKE_DIR/best.pth" ]]
[[ ! -e "$FULL_DIR/best.pth" ]]

COMMON_ARGS=(
  --manifest-path "$MANIFEST"
  --architecture unetplusplus
  --encoder-name tu-efficientnetv2_rw_s
  --outer-fold 2
  --calibration-fold 1
  --batch-size 16
  --workers 4
  --learning-rate 0.0005
  --weight-decay 0.0001
  --dropout 0.2
  --classification-loss-weight 0.0
  --classification-focal-gamma 1.0
  --background-weight 0.15
  --empty-foreground-weight 0.05
  --empty-foreground-top-fraction 0.001
  --checkpoint-selection-strategy fpr_volume_penalized
  --maximum-pos-weight 20
  --segmentation-class-weight-power 1.0
  --maximum-segmentation-class-weight 8.0
  --segmentation-class-weight-basis pixel
  --sampler-study-balance-power 0.0
  --hard-negative-multiplier 1.0
  --initial-checkpoint "$BASE_DIR/best.pth"
  --sah-residual-adapter
  --sah-residual-hidden-channels 16
  --sah-maximum-logit-residual 8.0
  --slice-context-radius 1
  --freeze-base-model
  --ivh-center-loss-weight 0.0
  --physical-volume-loss-weight 0.0
  --diffuse-tversky-loss-weight 0.0
  --sah-tversky-loss-weight 0.03
  --seed 42
  --skip-outer-evaluation
)

mkdir -p "$SMOKE_DIR"
PYTHONPATH=. uv run python scripts/train_ich_2p5d_segmentation.py \
  --run-name ich-exp65-smoke-effnetv2s-sahbgexpand003-warm-exp61-schema4-f2 \
  --output-dir "$SMOKE_DIR" \
  --epochs 1 \
  --patience 1 \
  --max-train-steps 4 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$SMOKE_DIR/console.log"

[[ -f "$SMOKE_DIR/run_summary.json" ]]
[[ ! -e "$SMOKE_DIR/outer_summary.json" ]]

PYTHONPATH=. uv run python scripts/train_ich_2p5d_segmentation.py \
  --run-name ich-exp65-effnetv2s-sahbgexpand003-warm-exp61-schema4-calonly-f2 \
  --output-dir "$FULL_DIR" \
  --epochs 6 \
  --patience 2 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$FULL_DIR/console.log"

[[ -f "$FULL_DIR/run_summary.json" ]]
[[ ! -e "$FULL_DIR/outer_summary.json" ]]

PYTHONPATH=. uv run python scripts/evaluate_ich_sah_residual_gate.py \
  --baseline-dir "$BASE_DIR" \
  --candidate-dir "$FULL_DIR" \
  --output "$FULL_DIR/promotion_gate.json" \
  --notify 2>&1 | tee "$FULL_DIR/promotion_gate.log"
