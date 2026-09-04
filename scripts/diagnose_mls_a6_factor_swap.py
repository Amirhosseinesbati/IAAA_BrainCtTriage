"""Canonical fixed 2x2 scalar-geometry/selector diagnostic; aggregates only."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reconstruct_mls_aligned_cached_screen import aligned, predictions, read_cache
from scripts.evaluate_mls_a2_fold0_resource_screen import _atomic_json, _sha256
from scripts.evaluate_mls_three_seed_fold_cuda import _metrics
from scripts.diagnose_mls_audit_aggregate import _threshold_summary, _truth_strata
from src.strategies.mls_heatmap.predict_multitask import SliceMLSPrediction


def mix(geometry, selector):
    aligned(geometry,selector)
    return {sid:(truth,[SliceMLSPrediction(g.index,s.selector_probability,g.mls_mm,g.heatmap_peak,s.peak_probability) for g,s in zip(slices,selector[sid][1])]) for sid,(truth,slices) in geometry.items()}


def diagnose(protocol_path,reference_path,output):
    if output.exists():raise FileExistsError('Refusing overwrite')
    if _sha256(protocol_path)!='15ffcfe7772bde61b1cc8c4bdd66f925261aebf50c09a73d63b0621fe560ffee':raise ValueError('Protocol mismatch')
    if _sha256(reference_path)!='0dd707cb86b39ef888ce56ee9fb29f455f6b6cfd4b5b870a32f6eeaa7dcf1b70':raise ValueError('Reference mismatch')
    spec=json.loads(protocol_path.read_text())
    reference=json.loads(reference_path.read_text())
    if _sha256(ROOT/'src/strategies/mls_heatmap/predict_multitask.py')!=spec['prediction_source_sha256']:raise ValueError('Pooling source changed')
    if _sha256(Path(spec['fold_manifest']))!=spec['fold_manifest_sha256']:raise ValueError('Fold source changed')
    with Path(spec['fold_manifest']).open(newline='',encoding='utf-8') as f:
        keys=sorted(r['study_id'] for r in csv.DictReader(f) if int(r['fold'])==0)
    if len(keys)!=70 or len(set(keys))!=70:raise ValueError('Wrong fold identity')
    caches={}
    for item in spec['inputs']:
        if item['label'] not in ['baseline','a6']:continue
        if _sha256(Path(item['csv']))!=item['csv_sha256']:raise ValueError('Cache hash mismatch')
        cache=read_cache(item['csv'])
        if set(cache)!=set(keys):raise ValueError('Coverage mismatch')
        caches[item['label']]=cache
    aligned(caches['baseline'],caches['a6'])
    if spec['canonical_pooling']['heatmap_guard_ratio']!=0:raise ValueError('Geometry confidence affects selector')
    truth=np.asarray([caches['baseline'][k][0] for k in keys])
    values={}
    results={}
    # Validate natives first, before evaluating mixed combinations.
    for g,s in [('baseline','baseline'),('a6','a6'),('a6','baseline'),('baseline','a6')]:
        key=f'{g}_geometry__{s}_selector'
        pred=predictions(mix(caches[g],caches[s]),keys,spec['canonical_pooling'],spec['clamp_mm'])
        stats=_metrics(truth,pred)
        if g==s and any(abs(v-reference['results'][g]['aligned_metrics'][k])>1e-7 for k,v in stats.items()):raise ValueError('Native reproduction failed')
        values[key]=pred
        results[key]={'metrics':stats,'thresholds':[_threshold_summary(truth,pred,t) for t in [3.,5.]],'truth_strata':_truth_strata(truth,pred)}
    base=values['baseline_geometry__baseline_selector']
    for key,pred in values.items():
        results[key]['threshold_changes_vs_native_baseline']={str(t):{'changed':int(np.sum((pred>=t)!=(base>=t))),'corrected':int(np.sum(((base>=t)!=(truth>=t))&((pred>=t)==(truth>=t)))),'new_errors':int(np.sum(((base>=t)==(truth>=t))&((pred>=t)!=(truth>=t))))} for t in [3.,5.]}
    result={'status':'completed','scope':'retrospective_factor_sensitivity_not_promotion','studies':70,'canonical_pooling':spec['canonical_pooling'],'clamp_mm':spec['clamp_mm'],'native_metrics_reproduced':True,'slice_alignment_basis':'shared_series_and_sorted_index_not_independently_verified_SOP_UID','results':results,'private_values_exported':False,'model_compute_performed':False,'promotion_eligible':False,'submission_zip_allowed':False,'source_sha256':_sha256(Path(__file__))}
    _atomic_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        r=diagnose(a.protocol,a.reference,a.output)
        print(json.dumps({'status':r['status'],'results':r['results']}))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'reason':str(exc)}))
        return 1


if __name__=='__main__':raise SystemExit(main())
