"""Audit both completed A8 arms with the frozen canonical evaluator, once each."""
from __future__ import annotations
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256
from scripts.reconstruct_mls_aligned_cached_screen import gate

CAMPAIGN=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK=CAMPAIGN/'a8_reference_refinement_20260904'
PROTOCOL=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A8_TRAINING_PROTOCOL_20260904.json'
PROTOCOL_SHA='199e344e9d823aec2fb7295fea59b9b55b5ac227ed90df9ffe1f43e45f456298'
EVALUATOR=ROOT/'scripts/evaluate_mls_refinement_resource_cuda.py'
EVALUATOR_SHA='be30c23ba52d1ddfca17bd17f233f4ae4426e03aeab9ea0353b8e6a63882177e'
REFERENCE=CAMPAIGN/'reference_refinement_runtime_qualified_20260904.json'
REFERENCE_SHA='9255e8387977c97bba19b77aa454403538abaa3ea03ddf184e89b87f136e3b96'
ARMS=('control','refinement')


def verify_configs(configs):
    if len(configs)!=2 or configs[0].get('use_reference_refinement') is not False or configs[1].get('use_reference_refinement') is not True:
        raise ValueError('Wrong control/candidate model variants')
    stripped=[{k:v for k,v in c.items() if k!='use_reference_refinement'} for c in configs]
    if stripped[0]!=stripped[1]:raise ValueError('Unplanned arm configuration difference')
    return True


def verify_training(summaries,histories,initial_sha):
    """Rebuild exposure checks; a saved matched=true flag alone is insufficient."""
    for arm in ARMS:
        summary,history=summaries[arm],histories[arm]
        if summary['status']!='completed' or summary['arm']!=arm:
            raise ValueError('Incomplete or mislabeled training arm')
        if summary['epochs_completed']!=15 or summary['optimizer_steps']!=8115:
            raise ValueError('Not the fixed training budget')
        if summary['manifest_sha256']!=PROTOCOL_SHA or summary['initialization_sha256']!=initial_sha:
            raise ValueError('Training provenance mismatch')
        if [r['epoch'] for r in history]!=list(range(1,16)) or any(r['optimizer_steps']!=541 for r in history):
            raise ValueError('Missing, duplicated or uneven training epochs')
        exposure=[r['input_exposure_sha256'] for r in history]
        if exposure!=summary['exposure_sha256_by_epoch'] or any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in exposure):
            raise ValueError('Exposure record mismatch')
    if summaries['control']['exposure_sha256_by_epoch']!=summaries['refinement']['exposure_sha256_by_epoch']:
        raise ValueError('Arms saw different augmented inputs')
    return True


def compare_audits(audits,reference,checkpoints):
    bounds=reference['prospective_gate_bounds']
    for arm in ARMS:
        audit=audits[arm]
        if audit.get('reference_refinement_enabled') is not (arm=='refinement'):
            raise ValueError('Audited model variant differs')
        if (audit['status'],audit['fold'],audit['seed'],audit['fixed_epoch'],audit['studies'])!=('completed',0,42,15,70):
            raise ValueError('Wrong audit scope')
        if audit['checkpoint_sha256']!=checkpoints[arm] or audit['baseline_self_test']:
            raise ValueError('Wrong audited checkpoint')
        if audit['runtime_reference_sha256']!=REFERENCE_SHA or audit['source_sha256']!=EVALUATOR_SHA:
            raise ValueError('Wrong evaluator or runtime reference')
        if audit['inference_signature']!=reference['inference_signature'] or audit['hardware_signature']!=reference['hardware_signature']:
            raise ValueError('Runtime differs between comparator and candidate')
        if audit['effective_gate_bounds']!=bounds or audit['runtime_baseline_metrics']!=reference['runtime_baseline_metrics']:
            raise ValueError('Changed gate bounds or baseline')
        metrics=audit['observed']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite audit metric')
        if any(not 0<=metrics[k]<=1 for k in ['f1_3mm','f1_5mm','boundary_f1']) or metrics['mae_mm']<0:
            raise ValueError('Impossible metric range')
        if abs(metrics['selection_objective']-(metrics['mae_mm']+2*(1-metrics['boundary_f1'])))>1e-8:
            raise ValueError('Objective does not match its components')
        gates=gate(metrics,bounds,1e-8)
        if gates!=audit['gate_results'] or bool(all(gates.values()))!=audit['resource_gates_passed']:
            raise ValueError('Saved pass flag is inconsistent with metrics')
        if any(audit[k] for k in ['automatic_replication_allowed','promotion_eligible','submission_zip_allowed']):
            raise ValueError('Unauthorized promotion flag')
    c,a=(audits[arm]['observed'] for arm in ARMS)
    better=a['selection_objective']<c['selection_objective']-1e-8
    return {'status':'completed','matched_training_verified':True,
        'baseline':reference['runtime_baseline_metrics'],
        'control':c,'refinement':a,'refinement_minus_control':{k:a[k]-c[k] for k in a},
        'control_resource_gates_passed':audits['control']['resource_gates_passed'],
        'refinement_resource_gates_passed':audits['refinement']['resource_gates_passed'],
        'refinement_objective_better_than_control':better,
        'refinement_replication_review_eligible':bool(better and audits['refinement']['resource_gates_passed']),
        'automatic_replication_allowed':False,'promotion_eligible':False,'submission_zip_allowed':False}


def main():
    for p,h in [(PROTOCOL,PROTOCOL_SHA),(EVALUATOR,EVALUATOR_SHA),(REFERENCE,REFERENCE_SHA)]:
        if _sha256(p)!=h:raise ValueError('Audit contract/source checksum changed')
    status=json.loads((WORK/'pair_completion.json').read_text())
    if status['status']!='completed':raise RuntimeError('Wait for both training arms; no GPU work launched')
    out=WORK/'canonical_pair_audit'
    if out.exists():raise FileExistsError('No rerun, overwrite or implicit audit resume')
    spec=json.loads(PROTOCOL.read_text())
    for relative,digest in spec['source_and_input_sha256'].items():
        if _sha256(ROOT/relative)!=digest:raise ValueError('Training source changed')
    summaries={arm:json.loads((WORK/arm/'training_summary.json').read_text()) for arm in ARMS}
    histories={arm:json.loads((WORK/arm/'training_history.json').read_text()) for arm in ARMS}
    initial_sha=_sha256(CAMPAIGN/'a7_paired_translation_20260904/initialization.pth')
    if initial_sha!=spec['initialization_sha256']:raise ValueError('Initialization changed from protocol')
    verify_training(summaries,histories,initial_sha)
    # Metadata loading only on CPU; all model inference is in CUDA evaluator.
    import torch
    configs=[]
    for arm in ARMS:
        checkpoint=WORK/arm/'mls_multitask_epoch_015.pth'
        if Path(summaries[arm]['checkpoint']).resolve()!=checkpoint.resolve() or _sha256(checkpoint)!=summaries[arm]['checkpoint_sha256']:
            raise ValueError('Checkpoint location or bytes mismatch')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if (state['epoch'],state['arm'],state['manifest_sha256'],state['initialization_sha256'])!=(15,arm,PROTOCOL_SHA,initial_sha):
            raise ValueError('Checkpoint embedded provenance mismatch')
        if state['mlflow_run_id']!=summaries[arm]['mlflow_run_id']:raise ValueError('MLflow provenance mismatch')
        configs.append(state['config'])
        del state
    verify_configs(configs)
    out.mkdir()
    audits={}
    for arm in ARMS:
        with (out/(arm+'.process.log')).open('x') as log:
            process=subprocess.run([sys.executable,str(EVALUATOR),'--checkpoint',summaries[arm]['checkpoint'],
                '--checkpoint-sha256',summaries[arm]['checkpoint_sha256'],
                '--runtime-reference',str(REFERENCE),'--runtime-reference-sha256',REFERENCE_SHA,
                '--output-dir',str(out/arm)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode!=0:raise RuntimeError('Canonical audit failed; preserve existing outputs')
        audits[arm]=json.loads((out/arm/'aggregate_summary.json').read_text())
    reference=json.loads(REFERENCE.read_text())
    result=compare_audits(audits,reference,{arm:summaries[arm]['checkpoint_sha256'] for arm in ARMS})
    result.update({'protocol_sha256':PROTOCOL_SHA,'evaluator_sha256':EVALUATOR_SHA,
        'wrapper_sha256':_sha256(Path(__file__)),'runtime_reference_sha256':REFERENCE_SHA,
        'training_summary_sha256':{a:_sha256(WORK/a/'training_summary.json') for a in ARMS},
        'audit_summary_sha256':{a:_sha256(out/a/'aggregate_summary.json') for a in ARMS},
        'checkpoint_sha256':{a:summaries[a]['checkpoint_sha256'] for a in ARMS}})
    _atomic_json(out/'pair_aggregate_summary.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()


