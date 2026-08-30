#!/usr/bin/env bash
set -euo pipefail

project_root="${FRACTURE_PROJECT_ROOT:-/workspace/IAAA_BrainCtTriage}"
cd "$project_root"

folds_root="Data/processed/fracture_v2_balanced_r4"
metadata="Data/metadata/training_df.csv"
output_root="Data/processed/fracture_mil_features/yolov8s_epoch10"
verification_root="reports/fracture_experiments/mil/cache_verification"
mkdir -p "$output_root" "$verification_root"

if (($#)); then
  folds=("$@")
else
  folds=(0 1 2 3 4)
fi

checkpoint_for_fold() {
  case "$1" in
    0) printf '%s\n' 'runs/detect/experiments/fracture_v2_coco/fracture-v2-posr4-f0-coco-lr5e4/weights/epoch10.pt' ;;
    1) printf '%s\n' 'runs/detect/experiments/fracture_v2/fracture-v2-posr4-f1-coco-lr5e4/weights/epoch10.pt' ;;
    2) printf '%s\n' 'runs/detect/experiments/fracture_v2_coco_f2/fracture-v2-posr4-f2-coco-lr5e4/weights/epoch10.pt' ;;
    3) printf '%s\n' 'runs/detect/experiments/fracture_v2/fracture-v2-posr4-f3-coco-lr5e4/weights/epoch10.pt' ;;
    4) printf '%s\n' 'runs/detect/experiments/fracture_v2_coco_f4/fracture-v2-posr4-f4-coco-lr5e4/weights/epoch10.pt' ;;
    *) echo "Unsupported fold: $1" >&2; return 2 ;;
  esac
}

reference_for_fold() {
  case "$1" in
    0) printf '%s\n' 'reports/fracture_experiments/coco_f0_checkpoint_screen/epoch10/study_predictions.csv' ;;
    1) printf '%s\n' 'reports/fracture_experiments/coco_f1_y8s_checkpoint_screen/epoch10/study_predictions.csv' ;;
    2) printf '%s\n' 'reports/fracture_experiments/coco_f2_checkpoint_screen/epoch10/study_predictions.csv' ;;
    3) printf '%s\n' 'reports/fracture_experiments/coco_f3_y8s_checkpoint_screen/epoch10/study_predictions.csv' ;;
    4) printf '%s\n' 'reports/fracture_experiments/coco_f4_checkpoint_screen/epoch10/study_predictions.csv' ;;
    *) echo "Unsupported fold: $1" >&2; return 2 ;;
  esac
}

for fold in "${folds[@]}"; do
  checkpoint="$(checkpoint_for_fold "$fold")"
  reference="$(reference_for_fold "$fold")"
  cache="$output_root/fold_$fold"
  if [[ ! -f "$cache/manifest.json" ]]; then
    echo "extracting fold=$fold checkpoint=$checkpoint"
    .venv/bin/python scripts/extract_fracture_mil_features.py \
      --fold "$fold" \
      --checkpoint "$checkpoint" \
      --folds-root "$folds_root" \
      --metadata "$metadata" \
      --output "$cache" \
      --image-size 512 \
      --batch-size 32 \
      --confidence 0.001 \
      --device 0
  else
    echo "reusing completed cache fold=$fold"
  fi
  .venv/bin/python scripts/verify_fracture_mil_cache.py \
    --cache "$cache" \
    --validation-fold "$fold" \
    --reference-predictions "$reference" \
    --tolerance 0.0002 \
    --auc-tolerance 1e-12 \
    --output "$verification_root/fold_$fold.json"
done

echo "all requested fracture MIL feature caches passed"
