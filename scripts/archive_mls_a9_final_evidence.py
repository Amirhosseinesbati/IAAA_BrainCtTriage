"""Create a readable, checksum-verified MLflow audit run for rejected A9."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlflow.tracking import MlflowClient
from src.mlops.tracking import configure_tracking_environment

BASE = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK = BASE / 'a9_frozen_baseline_refiner_20260904'
AUDIT = WORK / 'canonical_audit'
TRAINING_RUN_ID = 'bb4a898d61d544c9a450bfcd4ccb4b79'
CHECKPOINT_SHA = '853dc584f0c2baef731ddcf8d8b0ba2eef0914606285c7375039a1ea7d6bd8fe'
PAIR_SHA = 'd67d0e9b1cd3fa84e31e053c11fc521b75e636ff25d72eef884b8a4de594e0d9'
AUDIT_SHA = '81f3950211252783ee42c6c069cc9c5fd0550d0756009392122f7a91b0a6094f'
OUT = BASE / 'a9_final_archive_receipt_20260904.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_download(client, run_id, artifact_path, source):
    with tempfile.TemporaryDirectory(prefix='a9_mlflow_verify_') as directory:
        downloaded = client.download_artifacts(run_id, artifact_path, directory)
        if sha(downloaded) != sha(source):
            raise ValueError('MLflow artifact checksum mismatch: ' + artifact_path)


def main():
    if OUT.exists():
        raise FileExistsError('A9 final archive receipt already exists')
    pair = AUDIT / 'pair_aggregate_summary.json'
    audit = AUDIT / 'candidate/aggregate_summary.json'
    if sha(pair) != PAIR_SHA or sha(audit) != AUDIT_SHA:
        raise ValueError('A9 final audit bytes changed')
    pair_obj, audit_obj = json.loads(pair.read_text()), json.loads(audit.read_text())
    if (pair_obj['status'], pair_obj['resource_gates_passed'],
            pair_obj['promotion_eligible'], pair_obj['submission_zip_allowed']) != (
                'completed', False, False, False):
        raise ValueError('A9 decision is not the expected rejected resource screen')
    if pair_obj['checkpoint_sha256'] != CHECKPOINT_SHA or audit_obj['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('A9 training/audit checkpoint mismatch')
    if audit_obj['resource_gates_passed'] or audit_obj['gate_results']['f1_3mm_gte']:
        raise ValueError('A9 gate decision unexpectedly changed')

    configure_tracking_environment()
    if not os.getenv('MLFLOW_TRACKING_URI'):
        raise RuntimeError('Refusing local MLflow fallback; source the secure tracking environment first')
    client = MlflowClient()
    training = client.get_run(TRAINING_RUN_ID)
    experiment_id = training.info.experiment_id
    audit_run_id = client.create_run(experiment_id, tags={
        'mlflow.runName': 'A9 canonical audit — rejected at F1@3mm',
        'source_training_run_id': TRAINING_RUN_ID,
        'experiment_stage': 'canonical_fold0_seed42_resource_screen',
        'canonical_decision': 'rejected_resource_gate',
        'failed_gate': 'f1_3mm_gte',
        'promotion_eligible': 'false',
        'submission_zip_allowed': 'false',
        'private_predictions_uploaded': 'false',
    }).info.run_id
    try:
        params = {
            'fold': 0, 'seed': 42, 'fixed_epoch': 10, 'studies': 70,
            'candidate_checkpoint_sha256': CHECKPOINT_SHA,
            'pair_aggregate_sha256': PAIR_SHA, 'canonical_audit_sha256': AUDIT_SHA,
            'training_manifest_sha256': pair_obj['training_manifest_sha256'],
            'runtime_reference_sha256': pair_obj['runtime_reference_sha256'],
            'evaluation_protocol_sha256': pair_obj['evaluation_protocol_sha256'],
            'resource_gates_passed': False, 'promotion_eligible': False,
            'submission_zip_allowed': False,
        }
        for key, value in params.items():
            client.log_param(audit_run_id, key, value)
        for prefix, values in [
            ('baseline', pair_obj['baseline']), ('candidate', pair_obj['candidate']),
            ('delta', pair_obj['candidate_minus_baseline']),
        ]:
            for key, value in values.items():
                client.log_metric(audit_run_id, prefix + '_' + key, value)
        for key, value in pair_obj['gate_results'].items():
            client.log_metric(audit_run_id, 'gate_' + key, float(value))
        client.log_metric(audit_run_id, 'resource_gates_passed', 0.0)

        public_sources = {
            'reports/a9_canonical_audit/training_protocol.json':
                ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_TRAINING_PROTOCOL_20260904.json',
            'reports/a9_canonical_audit/evaluation_protocol.json':
                ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_CANONICAL_EVALUATION_PROTOCOL_20260904.json',
            'reports/a9_canonical_audit/preflight.json': WORK / 'preflight.json',
            'reports/a9_canonical_audit/training_summary.json': WORK / 'candidate/training_summary.json',
            'reports/a9_canonical_audit/training_history.json': WORK / 'candidate/training_history.json',
            'reports/a9_canonical_audit/candidate_aggregate.json': audit,
            'reports/a9_canonical_audit/pair_aggregate.json': pair,
            'reports/a9_canonical_audit/result.md':
                ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_PAIRED_RESULT_20260904.md',
        }
        records = []
        for artifact_path, source in public_sources.items():
            if 'private' in artifact_path.lower() or 'private' in source.name.lower():
                raise ValueError('Refusing a private A9 MLflow artifact')
            parent = artifact_path.rsplit('/', 1)[0]
            client.log_artifact(audit_run_id, str(source), parent)
            verify_download(client, audit_run_id, artifact_path, source)
            records.append({
                'artifact_path': artifact_path, 'sha256': sha(source),
                'download_verified': True,
            })

        for artifact_path, source in {
            'reports/training_summary.json': WORK / 'candidate/training_summary.json',
            'reports/training_history.json': WORK / 'candidate/training_history.json',
            'checkpoints/mls_multitask_epoch_010.pth': WORK / 'candidate/mls_multitask_epoch_010.pth',
        }.items():
            verify_download(client, TRAINING_RUN_ID, artifact_path, source)
            records.append({
                'training_artifact_path': artifact_path, 'sha256': sha(source),
                'download_verified': True,
            })

        client.set_tag(TRAINING_RUN_ID, 'canonical_audit_run_id', audit_run_id)
        client.set_tag(TRAINING_RUN_ID, 'canonical_decision', 'rejected_resource_gate')
        client.set_tag(TRAINING_RUN_ID, 'failed_gate', 'f1_3mm_gte')
        client.set_tag(TRAINING_RUN_ID, 'promotion_eligible', 'false')
        result = {
            'status': 'completed', 'training_run_id': TRAINING_RUN_ID,
            'canonical_audit_run_id': audit_run_id, 'decision': 'rejected_resource_gate',
            'failed_gate': 'f1_3mm_gte', 'pair_aggregate_sha256': PAIR_SHA,
            'canonical_audit_sha256': AUDIT_SHA, 'artifacts': records,
            'private_rows_uploaded': False, 'rejected_checkpoint_copied_locally': False,
            'promotion_eligible': False, 'submission_zip_allowed': False,
        }
        with OUT.open('x') as file:
            json.dump(result, file, indent=2)
        client.set_terminated(audit_run_id, status='FINISHED')
        print(json.dumps(result))
    except Exception:
        client.set_terminated(audit_run_id, status='FAILED')
        raise


if __name__ == '__main__':
    main()
