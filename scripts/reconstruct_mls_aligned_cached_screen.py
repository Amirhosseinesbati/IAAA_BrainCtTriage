"""Repair a pooling-comparison mismatch using immutable server-only slice caches.

No model inference/training. Read trusted checkpoint configuration only; report
aggregate results. Historical decisions are preserved, never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_mls_audit_aggregate import _finite_number, _threshold_summary, _truth_strata
from scripts.evaluate_mls_a2_fold0_resource_screen import _atomic_json, _sha256
from scripts.evaluate_mls_three_seed_fold_cuda import _metrics
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.predict_multitask import SliceMLSPrediction, aggregate_study_mls

POOL_FIELDS = {'selector_threshold':'selector_threshold', 'top_k':'top_k_slices', 'aggregation':'aggregation', 'relative_ratio':'selector_relative_ratio', 'aggregation_quantile':'aggregation_quantile', 'probability_weighted':'aggregation_probability_weighted', 'anchor_window_radius':'anchor_window_radius', 'min_active_slices':'min_active_slices', 'heatmap_guard_ratio':'heatmap_guard_ratio', 'negative_value':'negative_value_mm'}


def read_cache(path):
    result = {}
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            sid = str(row['study_id'])
            if not sid or sid in result or str(row.get('error','')).strip() not in {'','nan','None'}:
                raise ValueError('Duplicate/missing study or inference failure')
            truth = _finite_number(row['gt_MLS_mm'], 'truth')
            slices = []
            seen = set()
            for item in json.loads(row['slice_predictions_json']):
                index = int(item['index'])
                if index in seen or index < 0:
                    raise ValueError('Duplicate/negative slice index')
                seen.add(index)
                p = _finite_number(item['selector_probability'], 'probability')
                peak_p = item.get('peak_probability')
                if peak_p is not None:
                    peak_p = _finite_number(peak_p, 'peak_probability')
                if not 0 <= p <= 1 or (peak_p is not None and not 0 <= peak_p <= 1):
                    raise ValueError('Invalid probability')
                slices.append(SliceMLSPrediction(index, p, _finite_number(item['mls_mm'], 'mls'), _finite_number(item['heatmap_peak'], 'heatmap_peak'), peak_p))
            if not slices or [s.index for s in slices] != list(range(len(slices))):
                raise ValueError('Slice ordering/coverage is not canonical')
            result[sid] = (truth, slices)
    return result


def aligned(cache, reference):
    if set(cache) != set(reference):
        raise ValueError('Study coverage mismatch')
    for sid in reference:
        if abs(cache[sid][0] - reference[sid][0]) > 1e-9:
            raise ValueError('Truth mismatch')
        if [s.index for s in cache[sid][1]] != [s.index for s in reference[sid][1]]:
            raise ValueError('Slice coverage mismatch')


def predictions(cache, keys, pooling, clamp):
    values = np.asarray([aggregate_study_mls(cache[k][1], **pooling) for k in keys])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite aggregation')
    return np.clip(values, *clamp) if clamp is not None else values


def gate(metrics, bounds, tolerance):
    return {key: (metrics[key[:-4]] <= bound + tolerance if key.endswith('_lte') else metrics[key[:-4]] >= bound - tolerance) for key,bound in bounds.items()}


def run(protocol, output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite corrected report')
    spec = json.loads(protocol.read_text())
    if [i['label'] for i in spec['inputs']] != ['baseline','a2','a3','a4','a5','a6']:
        raise ValueError('Cohort changed')
    for field in ['baseline_reference_summary','baseline_reference_private','fold_manifest']:
        if _sha256(Path(spec[field])) != spec[field+'_sha256']:
            raise ValueError('Reference hash mismatch')
    if _sha256(ROOT/'src/strategies/mls_heatmap/predict_multitask.py') != spec['prediction_source_sha256']:
        raise ValueError('Pooling implementation changed')
    with Path(spec['fold_manifest']).open(newline='',encoding='utf-8') as stream:
        fold_ids = {r['study_id'] for r in csv.DictReader(stream) if int(r['fold']) == 0}
    with Path(spec['baseline_reference_private']).open(newline='',encoding='utf-8') as stream:
        reference_rows = list(csv.DictReader(stream))
    reference_values = {r['study_id']:_finite_number(r['seed42_MLS_mm'],'reference') for r in reference_rows}
    if len(reference_values) != len(reference_rows):
        raise ValueError('Duplicate baseline reference study')
    reference_summary = json.loads(Path(spec['baseline_reference_summary']).read_text())['member_metrics']['seed42']
    baseline = None
    results = {}
    for item in spec['inputs']:
        for key in ['csv','metrics','checkpoint']:
            if _sha256(Path(item[key])) != item[key+'_sha256']:
                raise ValueError('Candidate input hash mismatch')
        checkpoint = torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
        config = MLSHeatmapConfig.model_validate(checkpoint['config'])
        if config.fold != 0 or config.seed != 42 or int(checkpoint['epoch']) != 15:
            raise ValueError('Wrong checkpoint identity')
        if {k:getattr(config,v) for k,v in POOL_FIELDS.items()} != spec['canonical_pooling']:
            raise ValueError('Stored pooling differs from intended baseline')
        del checkpoint
        cache = read_cache(item['csv'])
        if set(cache) != fold_ids or len(cache) != spec['expected_studies']:
            raise ValueError('Wrong held-out coverage')
        if baseline is None:
            baseline = cache
        aligned(cache,baseline)
        keys = sorted(cache)
        truth = np.asarray([cache[k][0] for k in keys])
        old_metrics = json.loads(Path(item['metrics']).read_text())
        if Path(old_metrics['checkpoint']).resolve() != Path(item['checkpoint']).resolve() or old_metrics['failures'] != 0 or old_metrics['fold'] != 0 or old_metrics['n_studies'] != 70:
            raise ValueError('Cache provenance mismatch')
        legacy = _metrics(truth,predictions(cache,keys,{'selector_threshold':.5,'top_k':3,'aggregation':'p90'},None))
        for metric in ['mae_mm','rmse_mm','bias_mm','f1_3mm','f1_5mm']:
            if abs(legacy[metric] - old_metrics['fixed_profile_pre_registered'][metric]) > 1e-7:
                raise ValueError('Legacy metric reproduction failed')
        pred = predictions(cache,keys,spec['canonical_pooling'],spec['clamp_mm'])
        metrics = _metrics(truth,pred)
        if item['label'] == 'baseline':
            if set(reference_values) != set(keys) or any(abs(pred[j] - reference_values[k]) > 1e-6 for j,k in enumerate(keys)):
                raise ValueError('Per-study baseline reproduction failed')
            if any(abs(metrics[k] - reference_summary[k]) > 1e-7 for k in metrics):
                raise ValueError('Aggregate baseline reproduction failed')
        gates = gate(metrics,spec['gate_bounds_unchanged'],spec['comparison_tolerance'])
        results[item['label']] = {'aligned_metrics':metrics,'legacy_metrics_reproduced':True,'gate_results':gates,'all_gates_passed':all(gates.values()),'truth_strata':_truth_strata(truth,pred),'thresholds':[_threshold_summary(truth,pred,t) for t in [3.,5.]],'promotion_eligible':False,'automatic_replication_allowed':False}
    result = {'status':'completed','scope':'corrected_same_pooling_resource_comparison_only','protocol_sha256':_sha256(protocol),'source_sha256':_sha256(Path(__file__)),'studies':70,'canonical_pooling':spec['canonical_pooling'],'clamp_mm':spec['clamp_mm'],'baseline_reproduced_per_study':True,'all_legacy_metrics_reproduced':True,'results':results,'model_inference_performed':False,'model_training_performed':False,'private_predictions_exported':False,'promotion_eligible':False,'submission_zip_allowed':False,'historical_reports_preserved':True}
    _atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=run(args.protocol,args.output)
        print(json.dumps({'status':result['status'],'baseline_reproduced_per_study':True,'results':{k:{'metrics':v['aligned_metrics'],'all_gates_passed':v['all_gates_passed']} for k,v in result['results'].items()}}))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'reason':str(exc)}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
