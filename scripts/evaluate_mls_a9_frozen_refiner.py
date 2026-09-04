"""One-shot canonical A9 audit with explicit frozen-baseline provenance."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256
from scripts.reconstruct_mls_aligned_cached_screen import gate

BASE = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK = BASE / 'a9_frozen_baseline_refiner_20260904'
OUT = WORK / 'canonical_audit'
MANIFEST = ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_TRAINING_PROTOCOL_20260904.json'
MANIFEST_SHA = '9df64033b2139e78c861bb56f7977945d0314fbb612cc289e7fa3b50a9dd6794'
EVALUATION_PROTOCOL = ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_CANONICAL_EVALUATION_PROTOCOL_20260904.json'
EVALUATION_PROTOCOL_SHA = '63a206321afedb58183cd62446363ad4c437608cc50d36648f6748cb5afed95f'
PREFLIGHT = WORK / 'preflight.json'
PREFLIGHT_SHA = 'e7bcd33d2461221521a715ef15b00ccd4cac281e6e6a94d6bb6b47ca6e062ab6'
BASELINE = ROOT / 'models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth'
BASELINE_SHA = 'c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062'
REFERENCE = BASE / 'reference_refinement_runtime_qualified_20260904.json'
REFERENCE_SHA = '9255e8387977c97bba19b77aa454403538abaa3ea03ddf184e89b87f136e3b96'
GENERIC_EVALUATOR = ROOT / 'scripts/evaluate_mls_refinement_resource_cuda.py'
GENERIC_EVALUATOR_SHA = 'be30c23ba52d1ddfca17bd17f233f4ae4426e03aeab9ea0353b8e6a63882177e'
EVALUATOR = ROOT / 'scripts/evaluate_mls_a9_refinement_resource_cuda.py'
EVALUATOR_SHA = '8dc05b99f851ed9597bf9ce9203666398ebfb63d8d601444ee90b5ffdba8f21f'
CORRECTION = ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json'
CORRECTION_SHA = '15ffcfe7772bde61b1cc8c4bdd66f925261aebf50c09a73d63b0621fe560ffee'
CANDIDATE = WORK / 'candidate/mls_multitask_epoch_010.pth'
CANDIDATE_SHA = '853dc584f0c2baef731ddcf8d8b0ba2eef0914606285c7375039a1ea7d6bd8fe'
SUMMARY = WORK / 'candidate/training_summary.json'
SUMMARY_SHA = '5f857eb4347b76c8b841a8d37fa72ae60b6c4110731c9eba6e70949dfc55c6f3'
HISTORY = WORK / 'candidate/training_history.json'
HISTORY_SHA = 'a56ed661d748997eb14429eecf59cff44d89e2c49767352cdc37bec7f04edf11'
MLFLOW_RUN_ID = 'bb4a898d61d544c9a450bfcd4ccb4b79'
METRIC_KEYS = {
    'mae_mm', 'rmse_mm', 'bias_mm', 'f1_1mm', 'f1_3mm', 'f1_5mm',
    'boundary_f1', 'selection_objective',
}


def _is_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _verify_preflight():
    if _sha256(PREFLIGHT) != PREFLIGHT_SHA:
        raise ValueError('A9 preflight receipt changed')
    receipt = json.loads(PREFLIGHT.read_text())
    required = {
        'status': 'completed', 'baseline_identity_at_initialization': True,
        'frozen_baseline_unchanged_after_step': True, 'refiner_updated': True,
        'trainable_parameters': 47617, 'batch_size': 16, 'cuda_only': True,
        'validation_images_used': 0, 'manifest_sha256': MANIFEST_SHA,
        'baseline_checkpoint_sha256': BASELINE_SHA, 'promotion_eligible': False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError('A9 preflight semantic gate failed')
    return receipt


def _verify_training_receipts():
    if _sha256(SUMMARY) != SUMMARY_SHA or _sha256(HISTORY) != HISTORY_SHA:
        raise ValueError('A9 training receipt changed')
    summary, history = json.loads(SUMMARY.read_text()), json.loads(HISTORY.read_text())
    required = {
        'status': 'completed', 'epochs_completed': 10, 'optimizer_steps': 1690,
        'checkpoint': str(CANDIDATE), 'checkpoint_sha256': CANDIDATE_SHA,
        'mlflow_run_id': MLFLOW_RUN_ID, 'manifest_sha256': MANIFEST_SHA,
        'baseline_checkpoint_sha256': BASELINE_SHA, 'validation_images_used': 0,
        'frozen_baseline_verified': True, 'promotion_eligible': False,
    }
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError('A9 training summary semantic gate failed')
    if [row.get('epoch') for row in history] != list(range(1, 11)):
        raise ValueError('A9 training epoch history differs')
    if any(row.get('optimizer_steps') != 169 for row in history):
        raise ValueError('A9 training exposure differs')
    for row in history:
        if not _is_digest(row.get('input_exposure_sha256')):
            raise ValueError('A9 input exposure fingerprint is invalid')
        if not all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                   for key in ('train_loss', 'seconds', 'peak_vram_gib')):
            raise ValueError('A9 training history contains nonfinite values')
    return summary


def _verify_checkpoint(summary):
    if _sha256(CANDIDATE) != CANDIDATE_SHA:
        raise ValueError('A9 candidate checkpoint changed')
    # Metadata/tensor identity verification only; model inference remains CUDA-only below.
    import torch
    from scripts.evaluate_mls_refinement_resource_cuda import migrate_known_baseline
    from src.strategies.config_models import MLSHeatmapConfig

    saved = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    baseline = torch.load(BASELINE, map_location='cpu', weights_only=False)
    required = (10, 9, MANIFEST_SHA, BASELINE_SHA, MLFLOW_RUN_ID,
                'fixed_epoch10_no_validation_selection')
    observed = (saved.get('epoch'), saved.get('schema_version'), saved.get('manifest_sha256'),
                saved.get('baseline_checkpoint_sha256'), saved.get('mlflow_run_id'),
                saved.get('checkpoint_selection'))
    if observed != required or saved['mlflow_run_id'] != summary['mlflow_run_id']:
        raise ValueError('A9 checkpoint embedded provenance failed')
    expected_config = MLSHeatmapConfig.model_validate(
        migrate_known_baseline(baseline['config'], BASELINE_SHA)
    ).model_dump()
    candidate_config = saved['config']
    expected_stripped = {key: value for key, value in expected_config.items()
                         if key != 'use_reference_refinement'}
    stripped_config = {key: value for key, value in candidate_config.items()
                       if key != 'use_reference_refinement'}
    if candidate_config.get('use_reference_refinement') is not True or stripped_config != expected_stripped:
        raise ValueError('A9 altered baseline config beyond reference refiner')
    candidate_state, baseline_state = saved['model_state_dict'], baseline['model_state_dict']
    if set(baseline_state) - set(candidate_state):
        raise ValueError('A9 checkpoint lost frozen baseline tensors')
    if any(not torch.equal(candidate_state[key], value) for key, value in baseline_state.items()):
        raise ValueError('A9 changed a frozen baseline tensor')
    refiner_keys = set(candidate_state) - set(baseline_state)
    if not refiner_keys or any(not key.startswith('outer_refinement.') for key in refiner_keys):
        raise ValueError('A9 contains unexpected non-refiner tensors')
    return {'baseline_tensor_keys_verified': len(baseline_state),
            'refiner_tensor_keys': sorted(refiner_keys)}


def _verify_evaluator_epoch_contract_diff():
    """Prove the executable inference body differs only in A9's epoch contract."""
    generic = GENERIC_EVALUATOR.read_text()
    candidate = EVALUATOR.read_text()
    marker = 'from __future__ import annotations'
    generic_body = generic[generic.index(marker):].rstrip() + '\n'
    candidate_body = candidate[candidate.index(marker):].rstrip() + '\n'
    replacements = [
        ("payload['epoch'] != 10 or config.fold != 0 or config.seed != 42 or not config.use_competition_folds",
         "payload['epoch'] != 15 or config.fold != 0 or config.seed != 42 or not config.use_competition_folds"),
        ("Requires heldout fold0/seed42/epoch10 competition checkpoint",
         "Requires heldout fold0/seed42/epoch15 competition checkpoint"),
        ("'fixed_epoch': 10", "'fixed_epoch': 15"),
    ]
    for actual, expected in replacements:
        if candidate_body.count(actual) != 1:
            raise ValueError('A9 evaluator epoch-contract diff is not unique')
        candidate_body = candidate_body.replace(actual, expected, 1)
    if candidate_body != generic_body:
        raise ValueError('A9 evaluator inference body differs beyond epoch contract')
    return True


def _verify_static_contract():
    for path, digest in [
        (MANIFEST, MANIFEST_SHA), (EVALUATION_PROTOCOL, EVALUATION_PROTOCOL_SHA),
        (BASELINE, BASELINE_SHA), (REFERENCE, REFERENCE_SHA),
        (GENERIC_EVALUATOR, GENERIC_EVALUATOR_SHA), (EVALUATOR, EVALUATOR_SHA),
        (CORRECTION, CORRECTION_SHA),
    ]:
        if _sha256(path) != digest:
            raise ValueError('A9 audit contract/source changed: ' + str(path))
    manifest = json.loads(MANIFEST.read_text())
    for relative, digest in manifest['source_and_input_sha256'].items():
        if _sha256(ROOT / relative) != digest:
            raise ValueError('Pinned A9 training source/input changed: ' + relative)
    _verify_preflight()
    summary = _verify_training_receipts()
    checkpoint = _verify_checkpoint(summary)
    checkpoint['evaluator_epoch_contract_normalized_diff_verified'] = _verify_evaluator_epoch_contract_diff()
    return summary, checkpoint


def _verify_audit(audit, reference, correction):
    expected_scope = ('completed', 'canonical_fold0_seed42_resource_screen_only', 0, 42, 10, 70)
    observed_scope = (audit.get('status'), audit.get('scope'), audit.get('fold'),
                      audit.get('seed'), audit.get('fixed_epoch'), audit.get('studies'))
    if observed_scope != expected_scope:
        raise ValueError('Wrong A9 audit scope')
    if audit.get('reference_refinement_enabled') is not True:
        raise ValueError('A9 evaluator did not load reference refinement')
    if audit.get('baseline_self_test') is not False or audit.get('verified_baseline_sha256') is not None:
        raise ValueError('A9 must use only the qualified runtime reference')
    if audit.get('checkpoint_sha256') != CANDIDATE_SHA or Path(audit.get('checkpoint', '')).resolve() != CANDIDATE.resolve():
        raise ValueError('Wrong A9 audit checkpoint')
    if (audit.get('runtime_reference_sha256'), audit.get('source_sha256')) != (REFERENCE_SHA, EVALUATOR_SHA):
        raise ValueError('Wrong A9 runtime reference/evaluator')
    expected_hashes = {
        'correction_protocol_sha256': CORRECTION_SHA,
        'truth_sha256': '70a3551d9460c73e665cdd3ca6037407f1854152b211e7dfee09394bae149a94',
        'fold_manifest_sha256': correction['fold_manifest_sha256'],
        'reference_summary_sha256': correction['baseline_reference_summary_sha256'],
    }
    if any(audit.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError('A9 immutable audit input differs')
    if audit.get('compute_policy') != 'cuda_only_no_cpu_model_fallback':
        raise ValueError('A9 audit violated CUDA-only inference policy')
    if audit.get('inference_signature') != reference['inference_signature']:
        raise ValueError('A9 inference signature differs from qualified comparator')
    if audit.get('hardware_signature') != reference['hardware_signature']:
        raise ValueError('A9 hardware differs from qualified comparator')
    if audit.get('runtime_baseline_metrics') != reference['runtime_baseline_metrics']:
        raise ValueError('A9 runtime baseline differs from qualified comparator')
    legacy_baseline = json.loads(
        Path(correction['baseline_reference_summary']).read_text()
    )['member_metrics']['seed42']
    if audit.get('baseline_metrics') != legacy_baseline:
        raise ValueError('A9 evaluator legacy baseline metrics differ from correction contract')
    if audit.get('effective_gate_bounds') != reference['prospective_gate_bounds']:
        raise ValueError('A9 gate bounds differ from qualified comparator')
    if not _is_digest(audit.get('private_predictions_sha256')):
        raise ValueError('A9 private prediction receipt is invalid')
    metrics = audit.get('observed')
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise ValueError('A9 metric schema differs')
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()):
        raise ValueError('A9 metrics contain nonfinite values')
    if metrics['mae_mm'] < 0 or metrics['rmse_mm'] < 0:
        raise ValueError('A9 localization metric is impossible')
    if any(not 0 <= metrics[key] <= 1 for key in ('f1_1mm', 'f1_3mm', 'f1_5mm', 'boundary_f1')):
        raise ValueError('A9 boundary metric is impossible')
    if abs(metrics['selection_objective'] - (metrics['mae_mm'] + 2 * (1 - metrics['boundary_f1']))) > 1e-8:
        raise ValueError('A9 objective does not match its components')
    gates = gate(metrics, reference['prospective_gate_bounds'], 1e-8)
    if audit.get('gate_results') != gates or audit.get('resource_gates_passed') is not bool(all(gates.values())):
        raise ValueError('A9 resource gates are inconsistent with metrics')
    if any(audit.get(key) for key in ('automatic_replication_allowed', 'promotion_eligible', 'submission_zip_allowed')):
        raise ValueError('A9 evaluator authorized an impermissible promotion')
    return metrics, gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--finalize-existing', action='store_true')
    args = parser.parse_args()
    summary, checkpoint = _verify_static_contract()
    if args.preflight_only:
        print(json.dumps({
            'status': 'completed', 'stage': 'A9_provenance_preflight_only',
            'checkpoint_sha256': summary['checkpoint_sha256'],
            'frozen_baseline_exactly_verified': True, **checkpoint,
        }))
        return
    if OUT.exists():
        if ((OUT / 'pair_aggregate_summary.json').exists() or not args.finalize_existing
                or not (OUT / 'candidate' / 'aggregate_summary.json').is_file()):
            raise FileExistsError('Existing A9 output is not eligible for no-rerun finalization')
        audit_reused_without_inference = True
    else:
        if args.finalize_existing:
            raise FileNotFoundError('No existing A9 candidate audit to finalize')
        OUT.mkdir()
        with (OUT / 'candidate.process.log').open('x') as log:
            process = subprocess.run([
                sys.executable, str(EVALUATOR), '--checkpoint', str(CANDIDATE),
                '--checkpoint-sha256', CANDIDATE_SHA, '--runtime-reference', str(REFERENCE),
                '--runtime-reference-sha256', REFERENCE_SHA, '--output-dir', str(OUT / 'candidate'),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            raise RuntimeError('A9 canonical CUDA audit failed; preserve outputs, do not rerun')
        audit_reused_without_inference = False
    audit = json.loads((OUT / 'candidate' / 'aggregate_summary.json').read_text())
    reference, correction = json.loads(REFERENCE.read_text()), json.loads(CORRECTION.read_text())
    metrics, gates = _verify_audit(audit, reference, correction)
    result = {
        'status': 'completed', 'scope': 'canonical_fold0_seed42_resource_screen_only',
        'frozen_baseline_exactly_verified': True, **checkpoint,
        'baseline': reference['runtime_baseline_metrics'], 'candidate': metrics,
        'candidate_minus_baseline': {
            key: metrics[key] - reference['runtime_baseline_metrics'][key] for key in metrics
        },
        'resource_gates_passed': audit['resource_gates_passed'], 'gate_results': gates,
        'replication_review_eligible': bool(audit['resource_gates_passed']),
        'automatic_replication_allowed': False, 'promotion_eligible': False,
        'submission_zip_allowed': False, 'checkpoint_sha256': CANDIDATE_SHA,
        'preflight_sha256': PREFLIGHT_SHA, 'training_summary_sha256': SUMMARY_SHA,
        'training_history_sha256': HISTORY_SHA,
        'audit_summary_sha256': _sha256(OUT / 'candidate' / 'aggregate_summary.json'),
        'training_manifest_sha256': MANIFEST_SHA, 'runtime_reference_sha256': REFERENCE_SHA,
        'evaluation_protocol_sha256': EVALUATION_PROTOCOL_SHA,
        'generic_evaluator_sha256': GENERIC_EVALUATOR_SHA,
        'candidate_evaluator_sha256': EVALUATOR_SHA, 'wrapper_sha256': _sha256(Path(__file__)),
        'candidate_audit_finalized_without_rerun': audit_reused_without_inference,
    }
    _atomic_json(OUT / 'pair_aggregate_summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
