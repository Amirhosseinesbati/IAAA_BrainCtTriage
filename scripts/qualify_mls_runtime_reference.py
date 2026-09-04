"""Qualify two independent IEEE baseline executions without erasing old failures."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _metrics, _sha256
from src.evaluation.splits import load_fold_manifest

BASELINE_SHA = 'c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062'
CORRECTION_SHA = '15ffcfe7772bde61b1cc8c4bdd66f925261aebf50c09a73d63b0621fe560ffee'
EXPECTED_FLAGS = {'matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                  'cudnn_benchmark': False, 'cudnn_deterministic': False, 'matmul_precision': 'highest'}
CAMPAIGN = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
ANCHOR_PRIVATE = CAMPAIGN / 'canonical_baseline_residual_diagnostic_20260904/study_predictions_private.json'
ANCHOR_PRIVATE_SHA = 'fdfca255e5558459ae3c57f574acc7090bb0595eb9c19ec3e2a44d8f004becd3'


def conservative_bounds(old_bounds, current_metrics):
    bounds = {}
    for key, old in old_bounds.items():
        metric = key[:-4]
        current = current_metrics[metric]
        if not math.isfinite(current): raise ValueError('Nonfinite runtime baseline metric')
        if metric == 'selection_objective': current -= .01
        bounds[key] = min(old, current) if key.endswith('_lte') else max(old, current)
    return bounds


def verify_pair(a, b, rows_a, rows_b, old_predictions, anchor_rows):
    """Exact same-runtime reproducibility, including decoded slice outputs."""
    allowed = {'completed', 'failed_baseline_reproduction'}
    for run in [a, b]:
        if run['status'] not in allowed or not run['baseline_self_test'] or run['checkpoint_sha256'] != BASELINE_SHA:
            raise ValueError('Not a completed baseline-only inference audit')
        if (run['fold'], run['seed'], run['fixed_epoch'], run['studies']) != (0, 42, 15, len(old_predictions)):
            raise ValueError('Wrong baseline population')
        if run['compute_policy'] != 'cuda_only_no_cpu_model_fallback': raise ValueError('Wrong compute policy')
        if run['inference_signature']['precision_flags'] != EXPECTED_FLAGS: raise ValueError('Wrong runtime precision')
        if not run.get('execution_id') or not run.get('process_id'): raise ValueError('Missing independent execution identity')
    if a['process_id'] == b['process_id'] or a['execution_id'] == b['execution_id']:
        raise ValueError('Two independent processes are required')
    for key in ['inference_signature', 'source_sha256', 'hardware_signature', 'truth_sha256', 'fold_manifest_sha256', 'reference_summary_sha256']:
        if a[key] != b[key]: raise ValueError('Independent execution contracts differ')
    if rows_a != rows_b: raise ValueError('Repeated study/slice predictions or input fingerprints are not EXACTLY equal')
    ids = [r['study_id'] for r in rows_a]
    if len(ids) != len(set(ids)) or set(ids) != set(old_predictions): raise ValueError('Incorrect repeated coverage')
    anchors = {r['study_id']: r for r in anchor_rows}
    if len(anchors) != len(anchor_rows) or set(anchors) != set(ids): raise ValueError('Incorrect input anchor coverage')
    for row in rows_a:
        sid = row['study_id']
        if row['input_fingerprint'] != anchors[sid]['input_fingerprint'] or row['gt_MLS_mm'] != anchors[sid]['gt_MLS_mm']:
            raise ValueError('Raw input, ordered SOP identity or truth changed')
        if not math.isfinite(row['MLS_mm']) or not math.isfinite(old_predictions[sid]): raise ValueError('Nonfinite predictions')
        if any((row['MLS_mm'] >= t) != (old_predictions[sid] >= t) for t in [1., 3., 5.]):
            raise ValueError('Runtime migration changed an old baseline boundary decision')
    return True


def qualify(first_dir, second_dir, correction_path):
    if first_dir.resolve() == second_dir.resolve(): raise ValueError('Distinct execution directories required')
    if _sha256(correction_path) != CORRECTION_SHA: raise ValueError('Correction contract changed')
    spec = json.loads(correction_path.read_text())
    for key in ['fold_manifest', 'baseline_reference_summary', 'baseline_reference_private']:
        if _sha256(Path(spec[key])) != spec[key+'_sha256']: raise ValueError('Immutable reference changed')
    if _sha256(ANCHOR_PRIVATE) != ANCHOR_PRIVATE_SHA: raise ValueError('Input anchor changed')
    folds = load_fold_manifest(Path(spec['fold_manifest']))
    ids = set(folds.loc[folds.fold == 0, 'study_id'].astype(str))
    if len(ids) != 70: raise ValueError('Wrong immutable fold')
    with Path(spec['baseline_reference_private']).open(newline='') as f: original = list(csv.DictReader(f))
    reference = {r['study_id']: float(r['seed42_MLS_mm']) for r in original}
    if len(original) != 70 or set(reference) != ids: raise ValueError('Wrong original reference coverage')
    runs, private_rows, bindings = [], [], []
    for folder in [first_dir, second_dir]:
        result_path = folder / 'aggregate_summary.json'; result = json.loads(result_path.read_text())
        private_path = folder / 'study_predictions_private.json'
        if _sha256(private_path) != result['private_predictions_sha256']: raise ValueError('Private result changed')
        if _sha256(Path(result['checkpoint'])) != BASELINE_SHA: raise ValueError('Baseline checkpoint changed')
        if _sha256(ROOT/'scripts/evaluate_mls_canonical_resource_cuda.py') != result['source_sha256']:
            raise ValueError('Audit evaluator source changed')
        runs.append(result); private_rows.append(json.loads(private_path.read_text()))
        bindings.append({'directory':str(folder.resolve()), 'aggregate_sha256':_sha256(result_path), 'private_sha256':_sha256(private_path)})
    verify_pair(*runs, *private_rows, reference, json.loads(ANCHOR_PRIVATE.read_text()))
    rows = private_rows[0]
    metrics = _metrics(np.array([r['gt_MLS_mm'] for r in rows]), np.array([r['MLS_mm'] for r in rows]))
    if any(metrics[k] != runs[0]['observed'][k] or metrics[k] != runs[1]['observed'][k] for k in metrics):
        raise ValueError('Recorded aggregate metrics fail recomputation')
    return {'status':'qualified_same_runtime_reference', 'scope':'fold0_seed42_only_not_model_upgrade',
            'historical_cross_runtime_parity_passed':all(r['baseline_reproduction']['passed'] for r in runs),
            'same_runtime_predictions_exact':True, 'old_boundary_decisions_unchanged':True,
            'raw_inputs_and_ordered_sop_ids_unchanged':True, 'studies':70, 'bindings':bindings,
            'correction_protocol_sha256':CORRECTION_SHA, 'input_anchor_private_sha256':ANCHOR_PRIVATE_SHA,
            'qualifier_source_sha256':_sha256(Path(__file__)), 'evaluator_source_sha256':runs[0]['source_sha256'],
            'inference_signature':runs[0]['inference_signature'], 'hardware_signature':runs[0]['hardware_signature'],
            'runtime_baseline_metrics':metrics, 'old_gate_bounds':spec['gate_bounds_unchanged'],
            'prospective_gate_bounds':conservative_bounds(spec['gate_bounds_unchanged'],metrics),
            'promotion_eligible':False, 'submission_zip_allowed':False, 'automatic_replication_allowed':False}


def load_qualified_reference(path, expected_sha, correction_path):
    if _sha256(path) != expected_sha: raise ValueError('Runtime qualification receipt changed')
    recorded = json.loads(path.read_text())
    bindings = recorded['bindings']
    rebuilt = qualify(Path(bindings[0]['directory']), Path(bindings[1]['directory']), correction_path)
    if rebuilt != recorded: raise ValueError('Runtime qualification no longer verifies')
    rows = json.loads((Path(bindings[0]['directory'])/'study_predictions_private.json').read_text())
    return recorded, {r['study_id']:r for r in rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--first-dir',type=Path,required=True);parser.add_argument('--second-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('No qualification overwrite')
    correction=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json'
    result=qualify(args.first_dir,args.second_dir,correction)
    _atomic_json(args.output,result)
    print(json.dumps(result))


if __name__=='__main__':main()
