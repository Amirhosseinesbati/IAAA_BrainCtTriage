"""Training-only annotation provenance and extreme-geometry eligibility audit."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_mls_training_pose import PINS, digest, norm, geometry
from src.config import RAW_ANNOTATIONS_DIR, MLS_KEYPOINT_NAMES

OUT = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/training_geometry_provenance_20260904.json')

def main():
    if OUT.exists():
        raise FileExistsError('Evidence already exists')
    for rel, expected in PINS.items():
        if digest(ROOT/rel) != expected:
            raise ValueError('Input checksum changed')
    folds = pd.read_csv(ROOT/'config/folds.csv', usecols=['study_id','fold'], dtype={'study_id':str})
    ids = set(folds.loc[folds.fold != 0, 'study_id'].map(norm))
    labels = pd.read_csv(ROOT/'Data/processed/mls_multitask_v2/mls_labels_multitask.csv', dtype={'patient_id':str})
    labels.patient_id = labels.patient_id.map(norm)
    p = labels.loc[labels.patient_id.isin(ids) & (labels.is_target == 1)].copy().reset_index(drop=True)
    del labels
    if len(p) != 1360 or p.patient_id.nunique() != 138:
        raise ValueError('Population changed')
    raw = pd.read_pickle(ROOT/'Data/raw/training_df.pkl')
    raw = raw[['dicom_series.id','dicom_series.PixelSpacing1','MidlineShiftMM']].copy()
    raw['dicom_series.id'] = raw['dicom_series.id'].map(norm)
    raw = raw.loc[raw['dicom_series.id'].isin(ids)]
    grouped = raw.groupby('dicom_series.id')
    spacing = p.patient_id.map(grouped['dicom_series.PixelSpacing1'].median()).to_numpy(float)
    official = p.patient_id.map(grouped.MidlineShiftMM.max()).to_numpy(float)
    xy = p[['x1','y1','x2','y2','x3','y3']].to_numpy(float).reshape(-1,3,2)
    length, fraction, perp, angle = geometry(xy, np.repeat(spacing[:,None],2,axis=1))
    mls = abs(perp)
    if not np.isfinite(official).all() or not np.isfinite(mls).all():
        raise ValueError('Invalid truth')
    raw_matches = []
    content_digest = hashlib.sha256()
    for i,row in p.iterrows():
        prefix = row.patient_id + '_'
        if not row.image_name.startswith(prefix):
            raise ValueError('Unexpected image naming')
        name = Path(row.image_name[len(prefix):]).with_suffix('.json').name
        path = RAW_ANNOTATIONS_DIR / row.patient_id / name
        if not path.is_file():
            raise ValueError('Missing original annotation')
        data = path.read_bytes()
        content_digest.update(hashlib.sha256(data).digest())
        obj = json.loads(data)
        coords = np.asarray([obj['keypoints'][k][:2] for k in MLS_KEYPOINT_NAMES], float)
        raw_matches.append(bool(np.allclose(coords, xy[i], rtol=0, atol=1e-6)))
    matches = np.array(raw_matches)
    outside = (fraction < 0) | (fraction > 1)
    extreme = mls > 30  # existing deployment clamp; diagnostic only, not exclusion rule
    zeros = (xy == 0).any(axis=(1,2))
    p['geometry_mls'] = mls
    p['official'] = official
    gap = p.groupby('patient_id').geometry_mls.max() - p.groupby('patient_id').official.first()
    def counts(mask):
        return {'rows':int(mask.sum()), 'studies':int(p.loc[mask,'patient_id'].nunique()),
                'raw_coordinate_mismatches':int((mask & ~matches).sum()),
                'rows_with_zero_coordinate':int((mask & zeros).sum()),
                'outside_segment_projection':int((mask & outside).sum()),
                'study_official_over_30mm':int((mask & (official>30)).sum()),
                'local_above_official_by_over_half_mm':int((mask & (mls>official+.5)).sum())}
    result = {'status':'completed','scope':'1360 positive fold0 training slices; no heldout statistics',
        'source_sha256':digest(Path(__file__)), 'input_sha256':PINS,
        'raw_annotation_contents_digest':content_digest.hexdigest(),
        'all_positive':counts(np.ones(len(p),bool)), 'geometry_over_30mm':counts(extreme),
        'outside_projection':counts(outside),
        'max_abs_study_max_minus_official_mm':float(abs(gap).max()),
        'unaugmented_positive_mask_rule':'All three supplied finite in-image keypoints are present; is_target=1, so eligible. No anatomical rejection in current Gaussian mask builder.',
        'model_calls':0,'images_read':0,'labels_modified':0,'promotion_eligible':False,
        'limitations':['Raw agreement proves provenance, not anatomical correctness.',
                       'Official study maximum may be derived from the same annotations; not independent validation.',
                       'No measured gradients or realized sampled exposure; no influence claim.']}
    with OUT.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))

if __name__ == '__main__':
    main()
