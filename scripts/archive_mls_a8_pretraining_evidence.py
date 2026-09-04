"""Archive checksum-pinned A8 pretraining evidence without touching training."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mlops.tracking import configure_tracking_environment
from mlflow.tracking import MlflowClient

BASE=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
RUN='451470b102064180add9d0d21f2e45fe'
PINS={
'reference_refinement_preflight_20260904.json':'78aa095ec8c1dfd6ac31f03854f02e52a0b7a4a10c71c619afdfe8381f31f281',
'reference_integration_cuda_20260904.json':'b2646da4ac1bf47f0083db699d65933e15ac76093a1e0831295c2be08ff7f459',
'reference_real_batch_preflight_20260904.json':'4dff6c89b3a7c8cf09b2cbed254fbd5547b5010dcd9ab5df7ea0901598bcb446',
'reference_refinement_runtime_qualified_20260904.json':'9255e8387977c97bba19b77aa454403538abaa3ea03ddf184e89b87f136e3b96',
'reference_refinement_baseline_qualification_20260904/aggregate_summary.json':'e91d240619c4b9d81818f3987315b8f7b9eaad0e13f2f6c58ab84d82cb55efc3',
}
PREFIX='reports/a8_pretraining_evidence'
OUT=BASE/'a8_pretraining_archive_receipt_20260904.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists(): raise FileExistsError('Receipt already exists')
    configure_tracking_environment()
    client=MlflowClient()
    existing={a.path for a in client.list_artifacts(RUN,PREFIX)}
    records=[]
    for name,expected_sha in PINS.items():
        source=BASE/name
        if sha(source)!=expected_sha: raise ValueError('Pinned aggregate changed')
        obj=json.loads(source.read_text())
        if obj.get('status') not in {'completed','qualified_same_runtime_reference','failed_baseline_reproduction'} or obj.get('promotion_eligible') is not False:
            raise ValueError('Unexpected diagnostic payload')
        destination=PREFIX+'/'+source.name
        if destination not in existing:
            client.log_artifact(RUN,str(source),PREFIX)
        with tempfile.TemporaryDirectory(prefix='mls_diagnostic_verify_') as directory:
            downloaded=client.download_artifacts(RUN,destination,directory)
            if sha(downloaded)!=sha(source): raise ValueError('Archive checksum mismatch; do not overwrite')
        records.append({'artifact_path':destination,'sha256':sha(source),'download_verified':True})
    result={'status':'completed','run_id':RUN,'artifacts':records,'private_rows_uploaded':False}
    with OUT.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__': main()


