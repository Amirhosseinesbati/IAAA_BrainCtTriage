#!/usr/bin/env bash
set -euo pipefail

cd /workspace/project

MANIFEST="Data/processed/ich_2p5d/slice_manifest.csv"
OUTPUT_DIR="reports/ich_experiments/2p5d_segmentation/diagnostics/exp69_conditional_subtype_correction_oof_v1"

test -f "$MANIFEST"
test -f "reports/ich_experiments/2p5d_segmentation/exp30_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f0/best.pth"
test -f "reports/ich_experiments/2p5d_segmentation/exp32_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f1/best.pth"
test -f "reports/ich_experiments/2p5d_segmentation/exp26_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f2/best.pth"
test -f "reports/ich_experiments/2p5d_segmentation/exp34_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f3/best.pth"
test -f "reports/ich_experiments/2p5d_segmentation/exp36_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f4/best.pth"

PYTHONPATH=. uv run pytest \
  tests/test_ich_conditional_subtype_refiner.py \
  tests/test_ich_2p5d_segmentation.py \
  -q

PYTHONPATH=. uv run python scripts/diagnose_ich_conditional_subtype_oof.py \
  --fold-checkpoint "0:1:reports/ich_experiments/2p5d_segmentation/exp30_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f0/best.pth" \
  --fold-checkpoint "1:0:reports/ich_experiments/2p5d_segmentation/exp32_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f1/best.pth" \
  --fold-checkpoint "2:1:reports/ich_experiments/2p5d_segmentation/exp26_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f2/best.pth" \
  --fold-checkpoint "3:1:reports/ich_experiments/2p5d_segmentation/exp34_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f3/best.pth" \
  --fold-checkpoint "4:1:reports/ich_experiments/2p5d_segmentation/exp36_ivhcenter003_s11_calonly_hardempty001_fprselect_p1_audited_v3_f4/best.pth" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "ich-exp69-conditional-subtype-correction-oof-v1" \
  --architecture unetplusplus \
  --encoder-name efficientnet-b2 \
  --batch-size 16 \
  --workers 4 \
  --seed 42 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --stability-weight 1.0 \
  --class-weight-power 0.5 \
  --maximum-class-weight 4.0 \
  --conditional-margin 1.0 \
  --notify
