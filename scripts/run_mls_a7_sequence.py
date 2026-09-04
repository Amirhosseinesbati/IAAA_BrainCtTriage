"""Two finite preregistered GPU jobs, sequential; no evaluation or replication."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256

WORK=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902/a7_paired_translation_20260904')


def main():
    status=WORK/'sequence_status.json'
    if status.exists():raise FileExistsError('No automatic sequence restart')
    preflight=json.loads((WORK/'preflight.json').read_text())
    manifest=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A7_PAIRED_TRAINING_PROTOCOL_20260904.json'
    if preflight['status']!='passed' or preflight['manifest_sha256']!=_sha256(manifest):
        raise RuntimeError('Matching passed preflight required')
    try:
        for arm in ['control','consistency']:
            with (WORK/(arm+'.process.log')).open('x') as log:
                process=subprocess.Popen([sys.executable,str(ROOT/'scripts/train_mls_a7_paired_cuda.py'),
                    '--mode','train','--arm',arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                _atomic_json(status,{'status':'running','arm':arm,'pid':os.getpid(),'child_pid':process.pid})
                code=process.wait()
            if code!=0:raise RuntimeError('Training arm exited nonzero: '+arm)
        summaries=[json.loads((WORK/arm/'training_summary.json').read_text()) for arm in ['control','consistency']]
        matched=(summaries[0]['initialization_sha256']==summaries[1]['initialization_sha256']==preflight['initialization_sha256']
                 and summaries[0]['exposure_sha256_by_epoch']==summaries[1]['exposure_sha256_by_epoch']
                 and len(summaries[0]['exposure_sha256_by_epoch'])==15
                 and all(s['optimizer_steps']==8115 and s['epochs_completed']==15 for s in summaries))
        _atomic_json(WORK/'pair_completion_summary.json',{'status':'completed','matched_training_verified':matched,
            'arms':summaries,'promotion_eligible':False,'submission_zip_allowed':False})
        if not matched:raise RuntimeError('Paired exposure/initialization mismatch; no causal interpretation')
        _atomic_json(status,{'status':'completed','pid':os.getpid(),'matched_training_verified':True})
    except Exception as exc:
        _atomic_json(status,{'status':'failed','pid':os.getpid(),'error_type':type(exc).__name__})
        raise


if __name__=='__main__':main()
