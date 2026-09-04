"""Archive and independently re-download checksum-pinned A8 final evidence."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mlops.tracking import configure_tracking_environment
from mlflow.tracking import MlflowClient

BASE = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK = BASE / 'a8_reference_refinement_20260904'
AUDIT = WORK / 'canonical_pair_audit'
OUT = BASE / 'a8_final_archive_receipt_20260904.json'
RUNS = {
    'control': '451470b102064180add9d0d21f2e45fe',
    'refinement': '02f60665f699444cb0b9500c6a2eaf9f',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Final archive receipt already exists')
    pair = AUDIT / 'pair_aggregate_summary.json'
    pair_obj = json.loads(pair.read_text())
    if pair_obj['status'] != 'completed' or pair_obj['promotion_eligible'] or pair_obj['submission_zip_allowed']:
        raise ValueError('Unexpected A8 final decision')
    configure_tracking_environment()
    client = MlflowClient()
    records = []
    for arm, run_id in RUNS.items():
        summary = json.loads((WORK / arm / 'training_summary.json').read_text())
        if summary['mlflow_run_id'] != run_id or summary['checkpoint_sha256'] != pair_obj['checkpoint_sha256'][arm]:
            raise ValueError('Training/MLflow/checkpoint identity mismatch')
        sources = {
            'reports/a8_final_audit/aggregate_summary.json': AUDIT / arm / 'aggregate_summary.json',
            'reports/a8_final_audit/pair_aggregate_summary.json': pair,
        }
        for destination, source in sources.items():
            parent, name = destination.rsplit('/', 1)
            existing = {item.path for item in client.list_artifacts(run_id, parent)}
            if destination not in existing:
                client.log_artifact(run_id, str(source), parent)
            with tempfile.TemporaryDirectory(prefix='a8_final_verify_') as directory:
                downloaded = client.download_artifacts(run_id, destination, directory)
                if sha(downloaded) != sha(source):
                    raise ValueError('Final audit archive checksum mismatch')
            records.append({'arm': arm, 'run_id': run_id, 'artifact_path': destination,
                            'sha256': sha(source), 'download_verified': True})
        for destination, source in {
            'reports/training_summary.json': WORK / arm / 'training_summary.json',
            'checkpoints/mls_multitask_epoch_015.pth': WORK / arm / 'mls_multitask_epoch_015.pth',
        }.items():
            with tempfile.TemporaryDirectory(prefix='a8_training_verify_') as directory:
                downloaded = client.download_artifacts(run_id, destination, directory)
                if sha(downloaded) != sha(source):
                    raise ValueError('Training artifact archive checksum mismatch')
            records.append({'arm': arm, 'run_id': run_id, 'artifact_path': destination,
                            'sha256': sha(source), 'download_verified': True})
    result = {'status': 'completed', 'pair_summary_sha256': sha(pair), 'artifacts': records,
              'private_rows_uploaded': False, 'rejected_checkpoints_copied_locally': False,
              'promotion_eligible': False, 'submission_zip_allowed': False}
    with OUT.open('x') as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
