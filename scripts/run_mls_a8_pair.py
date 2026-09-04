"""One sequential paired run, no automatic retry, evaluation or promotion."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_mls_a8_reference_cuda import WORK,MANIFEST
from scripts.evaluate_mls_three_seed_fold_cuda import _sha256,_atomic_json

def main():
    if WORK.exists():raise FileExistsError('Pair already exists; inspect before any resume')
    for arm in ['control','refinement']:
        subprocess.run([sys.executable,str(ROOT/'scripts/train_mls_a8_reference_cuda.py'),'--arm',arm],cwd=ROOT,check=True)
    results=[json.loads((WORK/arm/'training_summary.json').read_text()) for arm in ['control','refinement']]
    for r in results:
        if r['status']!='completed' or r['optimizer_steps']!=8115 or r['epochs_completed']!=15 or r['manifest_sha256']!=_sha256(MANIFEST):
            raise ValueError('Incomplete or mismatched training')
        if _sha256(Path(r['checkpoint']))!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if results[0]['exposure_sha256_by_epoch']!=results[1]['exposure_sha256_by_epoch'] or results[0]['initialization_sha256']!=results[1]['initialization_sha256']:
        raise ValueError('Matched exposure/initialization failed')
    summary={'status':'completed','matched_training_verified':True,'validation_images_used':0,
        'run_ids':[r['mlflow_run_id'] for r in results],'promotion_eligible':False,'automatic_evaluation':False}
    _atomic_json(WORK/'pair_completion.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
