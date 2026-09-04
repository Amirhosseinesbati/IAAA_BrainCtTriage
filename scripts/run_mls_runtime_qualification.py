"""Run the two preregistered CUDA controls sequentially, then qualify aggregates."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _sha256
from scripts.qualify_mls_runtime_reference import BASELINE_SHA, CAMPAIGN, qualify
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json


def main():
    pins={'scripts/evaluate_mls_canonical_resource_cuda.py':'fa51c31a6fb44964cb172b8001a6945b0e9689ad1deebf1d04987f69f0c95f41',
          'scripts/qualify_mls_runtime_reference.py':'2d1e99a6ba50be5e66b9bd6e67590037546a99aa771d79cb2a61360b94ad631e',
          'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/SAME_RUNTIME_BASELINE_QUALIFICATION_PROTOCOL_20260904.md':'cbcc7c46ca58130a8292fe1c5b391a712dece4e0d8c1dcab827e906020ce2f9c'}
    for p,h in pins.items():
        if _sha256(ROOT/p)!=h:raise ValueError('Qualification source/protocol changed')
    outputs=[CAMPAIGN/f'ieee_baseline_independent_{label}_20260904' for label in ['a','b']]
    qualified=CAMPAIGN/'same_runtime_baseline_qualification_20260904.json'
    if qualified.exists() or any(p.exists() for p in outputs):raise FileExistsError('Refusing to rerun existing controls')
    checkpoint=Path('/workspace/IAAA_BrainCtTriage/models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth')
    for output in outputs:
        with output.with_suffix('.process.log').open('x') as log:
            process=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_mls_canonical_resource_cuda.py'),
                                    '--baseline-self-test','--checkpoint',str(checkpoint),
                                    '--checkpoint-sha256',BASELINE_SHA,'--output-dir',str(output)],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((output/'aggregate_summary.json').read_text())
        expected_code=0 if result['status']=='completed' else 1
        if result['status'] not in {'completed','failed_baseline_reproduction'} or process.returncode!=expected_code or result['studies']!=70:
            raise RuntimeError('A control did not complete all fixed studies')
    correction=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json'
    result=qualify(*outputs,correction)
    _atomic_json(qualified,result)
    print(json.dumps(result))


if __name__=='__main__':main()
