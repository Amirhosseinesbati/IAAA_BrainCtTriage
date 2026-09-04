"""Metadata-only fold0 training geometry; no images, model calls, or private rows."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/training_pose_audit_20260904.json')
PINS = {
    'Data/processed/mls_multitask_v2/mls_labels_multitask.csv': '01512662b62bcaf484f99cb872c40e28e2cfb300adee60db40957db0d06001ad',
    'config/folds.csv': 'd3c4640aec8fbfd8a912286bbf40ee39a7f48756c899cafcf8d976ce664ce2b8',
    'Data/raw/training_df.pkl': '0e00255ce7dcd6963a00db6c4d5a5dfdc5cff5a6fa8aec6ead5cc5da185cfe7d',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(value):
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r'\d+\.0', text) else text

def describe(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or not len(a):
        raise ValueError('Nonfinite or empty aggregate')
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                    map(float, np.quantile(a, [0, .1, .5, .9, 1]))), mean=float(a.mean()))

def geometry(points, spacing):
    p = points * spacing[:, None, :]
    direction = p[:, 1] - p[:, 0]
    length = np.linalg.norm(direction, axis=1)
    if (length <= 1e-6).any():
        raise ValueError('Degenerate segment')
    unit = direction / length[:, None]
    offset = p[:, 2] - p[:, 0]
    parallel = (offset * unit).sum(1)
    perpendicular = unit[:, 0] * offset[:, 1] - unit[:, 1] * offset[:, 0]
    angle = (np.degrees(np.arctan2(direction[:, 0], direction[:, 1])) + 90) % 180 - 90
    return length, parallel / length, perpendicular, angle

def main():
    if OUT.exists():
        raise FileExistsError('Do not overwrite prior evidence')
    for name, expected in PINS.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Dataset/fold pin mismatch')
    # Read only partition identifiers from the fold manifest, not held-out truth.
    folds = pd.read_csv(ROOT / 'config/folds.csv', usecols=['study_id', 'patient_id', 'fold'],
                        dtype={'study_id': str, 'patient_id': str})
    for col in ('study_id', 'patient_id'):
        folds[col] = folds[col].map(norm)
    if folds.study_id.duplicated().any() or folds.groupby('patient_id').fold.nunique().max() != 1:
        raise ValueError('Nonunique study or patient leakage')
    train_ids = set(folds.loc[folds.fold != 0, 'study_id'])
    val_ids = set(folds.loc[folds.fold == 0, 'study_id'])
    labels = pd.read_csv(ROOT / next(iter(PINS)), dtype={'patient_id': str})
    labels.patient_id = labels.patient_id.map(norm)
    if not set(labels.patient_id) <= set(folds.study_id):
        raise ValueError('Unknown study')
    # Remove held-out rows before any numerical geometry calculation.
    train = labels.loc[labels.patient_id.isin(train_ids)].copy()
    del labels
    if set(train.patient_id) & val_ids or len(train) != 2706 or train.patient_id.nunique() != 268:
        raise ValueError('Training population mismatch')
    pos = train.loc[train.is_target == 1].copy()
    if len(pos) != 1360 or pos.patient_id.nunique() != 138:
        raise ValueError('Positive population mismatch')
    p = pos[['x1','y1','x2','y2','x3','y3']].to_numpy(float).reshape(-1,3,2)
    # Legacy CSV has blank spacing; match deployed scalar spacing_x convention.
    raw = pd.read_pickle(ROOT / 'Data/raw/training_df.pkl')
    raw = raw[['dicom_series.id', 'dicom_series.PixelSpacing1']].copy()
    raw['dicom_series.id'] = raw['dicom_series.id'].map(norm)
    raw = raw.loc[raw['dicom_series.id'].isin(train_ids)]
    spacing_map = raw.groupby('dicom_series.id')['dicom_series.PixelSpacing1'].median()
    scalar = pos.patient_id.map(spacing_map).to_numpy(float)
    spacing = np.repeat(scalar[:, None], 2, axis=1)
    if not np.isfinite(p).all() or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid coordinates or spacing')
    if (p < 0).any() or (p >= 512).any():
        raise ValueError('Coordinates outside 512 image')
    # Analytical sanity check: vertical segment and 3-mm horizontal displacement.
    l, t, d, a = geometry(np.array([[[0.,0.],[0.,10.],[3.,5.]]]), np.ones((1,2)))
    if not np.allclose([l[0], t[0], abs(d[0]), a[0]], [10,.5,3,0]):
        raise ValueError('Geometry self-test failed')
    length, fraction, perp, angle = geometry(p, spacing)
    _, _, scalar_perp, _ = geometry(p, np.repeat(spacing[:, :1], 2, axis=1))
    midpoint_offset = ((p[:,0] + p[:,1]) / 2 - 255.5) * spacing
    metrics = {
        'segment_length_mm': length, 'abs_angle_from_vertical_deg': abs(angle),
        'midpoint_x_offset_mm': midpoint_offset[:,0],
        'midpoint_y_offset_mm': midpoint_offset[:,1],
        'outer_point_parallel_fraction': fraction,
        'abs_perpendicular_mm': abs(perp),
    }
    result = {
        'status': 'completed', 'scope': 'fold0 training positives only, unaugmented',
        'source_sha256': digest(Path(__file__)), 'input_sha256': PINS,
        'training_rows': len(train), 'positive_rows': len(pos), 'positive_studies': 138,
        'images_read': 0, 'model_calls': 0, 'heldout_geometry_used': False,
        'coordinates': '512 pixels; both axes scaled by raw PixelSpacing1 study median, matching current model; centre 255.5',
        'spacing_note': 'Legacy CSV spacing fields blank; raw metadata used. This diagnostic does not establish physical isotropy.',
        'slice_weighted': {k: describe(v) for k,v in metrics.items()},
        'distribution_of_study_means': {k: describe(pd.Series(v).groupby(pos.patient_id.to_numpy()).mean()) for k,v in metrics.items()},
        'counts': {'abs_angle_gt_10_deg': int((abs(angle)>10).sum()),
                   'abs_angle_gt_20_deg': int((abs(angle)>20).sum()),
                   'outer_projection_outside_segment': int(((fraction<0)|(fraction>1)).sum()),
                   'missing_csv_spacing_x': int(pos.spacing_x.isna().sum())},
        'promotion_eligible': False, 'automatic_training_allowed': False,
    }
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))

if __name__ == '__main__':
    main()
