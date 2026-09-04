"""Require exact baseline predictions across the explicit refinement source migration."""
from pathlib import Path
import json
import sys
import subprocess
from argparse import Namespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _sha256,_atomic_json
from scripts.qualify_mls_runtime_reference import load_qualified_reference as load_previous

BASE=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
OLD=BASE/'same_runtime_baseline_qualification_20260904.json'
OLD_SHA='59743c79a788839940f73e6d9e81cbd564c0fae9421239e2382debf3b56e5b19'
FOLDER=BASE/'reference_refinement_baseline_qualification_20260904'
RECEIPT=BASE/'reference_refinement_runtime_qualified_20260904.json'
CORRECTION=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json'
EVALUATOR=ROOT/'scripts/evaluate_mls_refinement_resource_cuda.py'

def rebuild(folder,correction):
    old,old_rows=load_previous(OLD,OLD_SHA,correction)
    result_path=folder/'aggregate_summary.json'
    result=json.loads(result_path.read_text())
    if result['status'] not in {'completed','failed_baseline_reproduction'} or not result['baseline_self_test']:
        raise ValueError('Incomplete baseline audit')
    if result['reference_refinement_enabled'] or result['studies']!=70:
        raise ValueError('Wrong model/coverage')
    if result['checkpoint_sha256']!='c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062':
        raise ValueError('Wrong baseline')
    if result['source_sha256']!=_sha256(EVALUATOR):raise ValueError('Evaluator changed')
    if _sha256(Path(result['checkpoint']))!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    private=folder/'study_predictions_private.json'
    if _sha256(private)!=result['private_predictions_sha256']:raise ValueError('Private rows changed')
    rows=json.loads(private.read_text()); by_id={r['study_id']:r for r in rows}
    if len(rows)!=70 or len(by_id)!=70 or by_id!=old_rows:
        raise ValueError('Baseline study/slice predictions, truth or input fingerprints not exactly reproduced')
    a=dict(result['inference_signature']);b=dict(old['inference_signature'])
    source=a.pop('source_sha256');b.pop('source_sha256')
    if a!=b or result['hardware_signature']!=old['hardware_signature']:
        raise ValueError('Non-source runtime contract changed')
    for rel,digest in source.items():
        if _sha256(ROOT/rel)!=digest:raise ValueError('Qualified source changed')
    if result['observed']!=old['runtime_baseline_metrics']:raise ValueError('Aggregate metrics differ')
    return {'status':'qualified_same_runtime_reference','scope':'refinement_source_migration_only_not_model_upgrade',
        'baseline_directory':str(folder),'baseline_aggregate_sha256':_sha256(result_path),
        'previous_reference_sha256':OLD_SHA,'baseline_private_sha256':_sha256(private),
        'source_migration_exact_predictions':True,'studies':70,
        'inference_signature':result['inference_signature'],'hardware_signature':result['hardware_signature'],
        'runtime_baseline_metrics':old['runtime_baseline_metrics'],'prospective_gate_bounds':old['prospective_gate_bounds'],
        'evaluator_source_sha256':_sha256(EVALUATOR),'qualifier_source_sha256':_sha256(Path(__file__)),
        'promotion_eligible':False,'submission_zip_allowed':False}

def load_qualified_reference(path,expected_sha,correction):
    if _sha256(path)!=expected_sha:raise ValueError('Qualification receipt changed')
    recorded=json.loads(Path(path).read_text())
    rebuilt=rebuild(Path(recorded['baseline_directory']),correction)
    if rebuilt!=recorded:raise ValueError('Qualification no longer verifies')
    rows=json.loads((Path(recorded['baseline_directory'])/'study_predictions_private.json').read_text())
    return recorded,{r['study_id']:r for r in rows}

def main():
    if RECEIPT.exists() or FOLDER.exists():raise FileExistsError('Do not rerun existing qualification')
    from scripts.evaluate_mls_refinement_resource_cuda import run
    checkpoint=ROOT/'models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth'
    run(Namespace(checkpoint=checkpoint,checkpoint_sha256='c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062',
        output_dir=FOLDER,baseline_self_test=True,verified_baseline_dir=None,verified_baseline_sha256=None,
        runtime_reference=None,runtime_reference_sha256=None))
    result=rebuild(FOLDER,CORRECTION);_atomic_json(RECEIPT,result)
    print(json.dumps(result))

if __name__=='__main__':main()
