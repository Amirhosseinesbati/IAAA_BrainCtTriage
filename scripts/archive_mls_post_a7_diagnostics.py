"""Archive aggregate-only post-A7 diagnostics and verify downloads."""
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
RUN='5746a793c5d04da994935651fbaae5d4'
NAMES=['training_pose_audit_20260904.json','training_geometry_provenance_20260904.json','training_error_components_20260904.json']
PREFIX='reports/post_a7_diagnostics'
OUT=BASE/'post_a7_diagnostics_archive_receipt_20260904.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists(): raise FileExistsError('Receipt already exists')
    configure_tracking_environment()
    client=MlflowClient()
    existing={a.path for a in client.list_artifacts(RUN,PREFIX)}
    records=[]
    for name in NAMES:
        source=BASE/name
        obj=json.loads(source.read_text())
        if obj.get('status')!='completed' or obj.get('promotion_eligible') is not False:
            raise ValueError('Unexpected diagnostic payload')
        destination=PREFIX+'/'+name
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
