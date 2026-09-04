"""Training-only metadata audit; no images, checkpoint, model or validation statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.splits import load_fold_manifest, normalize_study_id, split_study_ids
from src.strategies.mls_heatmap.dataset import build_mls_sampling_weights

EXPECTED = {
    'Data/processed/mls_multitask_v2/mls_labels_multitask.csv': '01512662b62bcaf484f99cb872c40e28e2cfb300adee60db40957db0d06001ad',
    'config/folds.csv': 'd3c4640aec8fbfd8a912286bbf40ee39a7f48756c899cafcf8d976ce664ce2b8',
    'src/strategies/mls_heatmap/dataset.py': 'df0852f12eba9329e3a65787d65e9649922d6b8c2704b2d7d47192ec33c9ac1c',
    'src/strategies/mls_heatmap/train_multitask.py': '5135d38b193a59079c763ea8003e66e7d5da1a81be94355b82fead1c4eb81cc3',
    'config/experiments/mls-vast-deploy-aligned-a6-local-geometry-template.yaml': '28689c91e9c94a30c1543dde119633a608ff3594e22b9a0ae928fc58ea71822e',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_mm(frame):
    xy = frame[['x1', 'y1', 'x2', 'y2', 'x3', 'y3']].to_numpy(float).reshape(-1, 3, 2)
    spacing = frame.spacing_x.to_numpy(float)
    if not np.isfinite(xy).all() or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid positive geometry/spacing')
    direction = xy[:, 1] - xy[:, 0]
    length = np.linalg.norm(direction, axis=1)
    if (length < 1e-6).any():
        raise ValueError('Degenerate midline annotations')
    cross = direction[:, 0] * (xy[:, 0, 1] - xy[:, 2, 1]) - (xy[:, 0, 0] - xy[:, 2, 0]) * direction[:, 1]
    return np.abs(cross) / length * spacing


def summarize_training(frame, *, batch_size=5):
    frame = frame.reset_index(drop=True).copy()
    if frame.duplicated(['patient_id', 'image_name']).any():
        raise ValueError('Duplicate training row identity')
    if not np.isfinite(frame.study_mls_mm.to_numpy(float)).all() or (frame.study_mls_mm < 0).any():
        raise ValueError('Invalid study truth')
    if (frame.groupby('patient_id').study_mls_mm.nunique() != 1).any():
        raise ValueError('Nonconstant study truth')
    weights = build_mls_sampling_weights(frame, 'slice_class_balanced').numpy()
    p = weights / weights.sum()
    target = frame.is_target.to_numpy() == 1
    if not np.isclose(p[target].sum(), .5):
        raise ValueError('Sampler class mass is not balanced')
    steps = len(frame) // batch_size
    draws = steps * batch_size
    positive = frame.loc[target].copy()
    truth = geometry_mm(positive)
    official = positive.study_mls_mm.to_numpy(float)
    pp = p[target]
    conditional = pp / pp.sum()
    positive['slice_mls_mm'] = truth
    positive['mass'] = conditional
    by_study = positive.groupby('patient_id').agg(
        study_mls_mm=('study_mls_mm', 'first'), max_annotated_mls=('slice_mls_mm', 'max'),
        positive_rows=('slice_mls_mm', 'size'), mass=('mass', 'sum'))
    all_studies = frame.groupby('patient_id').study_mls_mm.first()
    bins = [0., 1., 2.5, 3., 3.5, 4.5, 5., 5.5, float('inf')]
    strata = []
    for low, high in zip(bins[:-1], bins[1:]):
        sm = (truth >= low) & (truth < high)
        study_mask = (all_studies >= low) & (all_studies < high)
        row_study_mask = (official >= low) & (official < high)
        strata.append({
            'range_mm': [low, high if np.isfinite(high) else None],
            'training_studies': int(study_mask.sum()),
            'positive_studies': int(((by_study.study_mls_mm >= low) & (by_study.study_mls_mm < high)).sum()),
            'positive_slices_by_local_mls': int(sm.sum()),
            'expected_positive_draws_per_epoch_by_local_mls': float(draws * pp[sm].sum()),
            'expected_positive_draws_per_epoch_by_study_mls': float(draws * pp[row_study_mask].sum()),
        })
    boundaries = {}
    for threshold in [1., 3., 5.]:
        local_side = truth >= threshold
        study_side = official >= threshold
        near = np.abs(truth - threshold) <= .5
        severe = by_study.study_mls_mm >= threshold
        boundaries[str(threshold)] = {
            'local_within_half_mm_slices': int(near.sum()),
            'local_within_half_mm_studies': int(positive.loc[near, 'patient_id'].nunique()),
            'local_within_half_mm_expected_draws_per_epoch': float(draws * pp[near].sum()),
            'local_within_half_mm_positive_mass': float(conditional[near].sum()),
            'study_within_half_mm_count': int((np.abs(all_studies - threshold) <= .5).sum()),
            'study_positive_but_local_negative_slices': int((study_side & ~local_side).sum()),
            'study_positive_but_local_negative_positive_mass': float(conditional[study_side & ~local_side].sum()),
            'study_negative_but_local_positive_slices': int((~study_side & local_side).sum()),
            'positive_studies_above_threshold': int(severe.sum()),
            'positive_studies_above_threshold_without_annotated_slice_above': int((severe & (by_study.max_annotated_mls < threshold)).sum()),
        }
    xy = positive[['x1', 'y1', 'x2', 'y2', 'x3', 'y3']].to_numpy(float)
    targets = .75 + .25 * np.clip(truth / np.maximum(official, .1), 0., 1.)
    quantiles = [0., .1, .5, .9, 1.]
    gap = by_study.max_annotated_mls - by_study.study_mls_mm
    return {
        'training_rows': len(frame), 'training_studies': len(all_studies),
        'positive_rows': int(target.sum()), 'positive_studies': len(by_study),
        'batch_size': batch_size, 'optimizer_steps_per_epoch': steps,
        'consumed_draws_per_epoch': draws, 'expected_positive_draws_per_epoch': float(draws * p[target].sum()),
        'expected_no_positive_batch_fraction': float(.5 ** batch_size),
        'unaugmented_positive_out_of_bounds_rows': int(((xy < 0) | (xy >= 512)).any(axis=1).sum()),
        'quantile_levels': quantiles,
        'positive_rows_per_study_quantiles': np.quantile(by_study.positive_rows, quantiles).tolist(),
        'positive_study_mass_effective_sample_size': float(1. / np.square(by_study.mass).sum()),
        'positive_selector_target_quantiles': np.quantile(targets, quantiles).tolist(),
        'max_annotated_minus_official_mm_quantiles': np.quantile(gap, quantiles).tolist(),
        'studies_max_annotation_below_official_by_over_half_mm': int((gap < -.5).sum()),
        'studies_max_annotation_above_official_by_over_half_mm': int((gap > .5).sum()),
        'strata': strata, 'boundaries': boundaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a completed audit')
    for relative, expected in EXPECTED.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Changed pinned input: {relative}')
    labels = pd.read_csv(ROOT / 'Data/processed/mls_multitask_v2/mls_labels_multitask.csv')
    labels['patient_id'] = labels.patient_id.map(normalize_study_id)
    train_ids, val_ids = split_study_ids(labels.patient_id.unique(), 0)
    folds = load_fold_manifest()
    actual_train_patients = set(folds.loc[folds.study_id.isin(train_ids), 'patient_id'])
    actual_val_patients = set(folds.loc[folds.study_id.isin(val_ids), 'patient_id'])
    if train_ids & val_ids or actual_train_patients & actual_val_patients:
        raise ValueError('Patient/study leakage')
    training = labels.loc[labels.patient_id.isin(train_ids)].copy()
    # Drop all validation records before statistics; never inspect their labels or predictions.
    del labels
    result = {'status': 'completed', 'scope': 'fold0_training_metadata_only',
              'model_inference': False, 'images_read': 0, 'heldout_statistics': False,
              'inputs_sha256': EXPECTED, 'source_sha256': sha(__file__),
              'patient_and_study_disjoint': True, 'promotion_eligible': False,
              'limitations': ['Unaugmented metadata, not realized stochastic exposure or gradients.',
                              'Expected draws use iid replacement sampler and drop_last.',
                              'Per-positive loss uses a within-batch mean; masses are exposure, not total gradient shares.',
                              'All strata fixed before result; no weight/threshold search.'],
              'summary': summarize_training(training)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
