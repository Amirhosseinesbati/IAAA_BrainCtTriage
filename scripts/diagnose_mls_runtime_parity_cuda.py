"""Fixed baseline-only numerical runtime diagnostic, not model selection."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_canonical_resource_cuda import (
    CAMPAIGN, CORRECTION, CORRECTION_SHA, SOURCE_HASHES, LEGACY_BASELINE_SHA,
    aggregate, input_fingerprint, migrate_known_baseline, signature, validate_private,
)
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _metrics, _sha256
from src.evaluation.splits import load_fold_manifest
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model, predict_reader_slices

FRESH_ROOT = CAMPAIGN / 'canonical_baseline_residual_diagnostic_20260904'
FRESH_RESULT_SHA = 'd920c3209464c4ab09eccbfefd680b010ed66f9743fb2f694ad18ccfc566b443'
FRESH_PRIVATE_SHA = 'fdfca255e5558459ae3c57f574acc7090bb0595eb9c19ec3e2a44d8f004becd3'
EVALUATOR_SHA = '274243acd0e3b3fc2b3876f89d6daec59f6fde8b72edc7b633d7032c04ec7970'
MODES = [('repeat_default_batch6', 6, True), ('default_batch16', 16, True), ('ieee_convolution_batch6', 6, False)]


def compare(values, reference):
    delta = np.abs(values-reference)
    return {'mean_absolute_difference_mm': float(delta.mean()), 'max_absolute_difference_mm': float(delta.max()),
            'studies_over_1e_minus5_mm': int((delta > 1e-5).sum()),
            'changed_decisions_at_3mm': int(((values >= 3) != (reference >= 3)).sum()),
            'changed_decisions_at_5mm': int(((values >= 5) != (reference >= 5)).sum())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Do not overwrite runtime evidence')
    for path, digest in [(CORRECTION,CORRECTION_SHA),(FRESH_ROOT/'aggregate_summary.json',FRESH_RESULT_SHA),
                         (FRESH_ROOT/'study_predictions_private.json',FRESH_PRIVATE_SHA),
                         (ROOT/'scripts/evaluate_mls_canonical_resource_cuda.py',EVALUATOR_SHA)]:
        if _sha256(path)!=digest:raise ValueError('Runtime diagnostic reference changed')
    for path,digest in SOURCE_HASHES.items():
        if _sha256(ROOT/path)!=digest:raise ValueError('Inference source changed')
    spec=json.loads(CORRECTION.read_text())
    if _sha256(Path(spec['fold_manifest']))!=spec['fold_manifest_sha256']:raise ValueError('Fold hash changed')
    folds=load_fold_manifest(Path(spec['fold_manifest']));ids=sorted(folds.loc[folds.fold==0,'study_id'].astype(str))
    if len(ids)!=70:raise ValueError('Wrong fold coverage')
    fresh=validate_private(json.loads((FRESH_ROOT/'study_predictions_private.json').read_text()),ids)
    if _sha256(Path(spec['baseline_reference_private']))!=spec['baseline_reference_private_sha256']:raise ValueError('Original reference changed')
    with Path(spec['baseline_reference_private']).open(newline='') as f:old=list(csv.DictReader(f))
    old_by_id={r['study_id']:float(r['seed42_MLS_mm']) for r in old}
    if len(old)!=70 or set(old_by_id)!=set(ids):raise ValueError('Reference coverage changed')
    checkpoint=Path(spec['inputs'][0]['checkpoint'])
    if _sha256(checkpoint)!=LEGACY_BASELINE_SHA:raise ValueError('Wrong baseline weights')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    lock=CAMPAIGN/'gpu_training.lock';lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():raise RuntimeError('GPU already occupied')
        model,config=load_multitask_model(checkpoint,torch.device('cuda:0'))
        pool_contract=signature(migrate_known_baseline(config.model_dump(),LEGACY_BASELINE_SHA),spec['canonical_pooling'],spec['clamp_mm'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=False
        # Deidentified UID format warnings contain identifiers. Suppress just
        # their display; do not relax UID existence/uniqueness or reader errors.
        warnings.filterwarnings('ignore',message='Invalid value for VR UI:',category=UserWarning,module='pydicom.valuerep')
        values={name:[] for name,_,_ in MODES};start=time.monotonic()
        for sid in ids:
            reader=BrainDicomReader(str(ROOT/'Data/raw/training'/sid)).load_and_sort()
            if input_fingerprint(reader)!=fresh[sid]['input_fingerprint']:raise ValueError('Raw input or SOP order changed')
            for name,batch,cudnn_tf32 in MODES:
                torch.backends.cudnn.allow_tf32=cudnn_tf32
                slices=predict_reader_slices(reader,model,config,torch.device('cuda:0'),batch_size=batch)
                values[name].append(aggregate(slices,pool_contract))
        reference=np.array([old_by_id[k] for k in ids]);fresh_values=np.array([fresh[k]['MLS_mm'] for k in ids]);truth=np.array([fresh[k]['gt_MLS_mm'] for k in ids])
        results={}
        for name,batch,cudnn_tf32 in MODES:
            pred=np.asarray(values[name])
            results[name]={'batch_size':batch,'cudnn_allow_tf32':cudnn_tf32,'matmul_allow_tf32':False,
                           'versus_immutable_reference':compare(pred,reference),'versus_fresh_default':compare(pred,fresh_values),
                           'metrics':_metrics(truth,pred)}
        result={'status':'completed','scope':'fixed_baseline_runtime_diagnostic_not_promotion','studies':70,
                'checkpoint_sha256':LEGACY_BASELINE_SHA,'fresh_result_sha256':FRESH_RESULT_SHA,
                'source_sha256':_sha256(Path(__file__)),'results':results,'runtime_seconds':time.monotonic()-start,
                'same_raw_content_and_sop_order_verified':True,'no_training':True,'promotion_eligible':False,'submission_zip_allowed':False}
        _atomic_json(args.output,result);print(json.dumps(result))
    finally:lock.rmdir()


if __name__=='__main__':main()
