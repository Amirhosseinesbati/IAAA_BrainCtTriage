"""Fixed fold0/seed42/epoch15 CUDA screen with executable inference parity.

First run --baseline-self-test. Candidates must supply that verified baseline
directory AND its result hash. No grids, resume, automatic training or release.
Private per-study records never leave the server; stdout is aggregate only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _metrics, _sha256
from scripts.reconstruct_mls_aligned_cached_screen import POOL_FIELDS, gate
from src.config import WINDOWS
from src.evaluation.splits import load_fold_manifest
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.predict_multitask import aggregate_study_mls, load_multitask_model, predict_reader_slices

CORRECTION = ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/POOLING_COMPARISON_CORRECTION_PROTOCOL_20260904.json'
CORRECTION_SHA = '15ffcfe7772bde61b1cc8c4bdd66f925261aebf50c09a73d63b0621fe560ffee'
CAMPAIGN = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
TRUTH = ROOT / 'reports/eda/deep/deep_series_table.csv'
TRUTH_SHA = '70a3551d9460c73e665cdd3ca6037407f1854152b211e7dfee09394bae149a94'
SOURCE_HASHES = {
    'src/preprocessing/core/dicom_reader.py': 'c8dd7f5ca2bbddfb6dfe9b7005c21c02f16eb1398d53a33929ac0d6793dbfc52',
    'src/strategies/mls_heatmap/predict_multitask.py': 'f56f6b98ea9c6320a0b29358db81e04f09942677dde51e4b464550601f18f11e',
    'src/strategies/mls_heatmap/predict.py': 'e8c86f4b3ec5628b764e3ac38befc6d968b9fa4ae4d3d84d90072440f3384d56',
    'src/strategies/mls_heatmap/utils.py': '96377673b4cbb37da59a32cf67bc152baa4cfff875373ff4d87fdeb4de77239a',
    'src/strategies/mls_heatmap/model.py': '51d6b53572fd0ad720290be802c1cd3a0b4714433c680bf22c8298e501822e0e',
    'scripts/evaluate_mls_three_seed_fold_cuda.py': 'c8c867fbdc39a6206e2992644c9ede4113dd524ba53461dae555148ae3172ddb',
    'scripts/reconstruct_mls_aligned_cached_screen.py': '60d5c1e2da3607bccbfeb4a6482670ec3aa6c46e613fca41ba5edd4869ae1ec2',
    'src/strategies/config_models.py': 'bc4a2263a02f69143d903d4de41dfa1740b782500b8bf8ddb7d4c0b39eeee089',
}
PREPROCESS = {'image_size': 512, 'input_channels': 3, 'use_selector': True, 'selector_head_mode': 'single'}
EXPECTED_WINDOWS = {'brain': {'width': 80, 'level': 40}, 'subdural': {'width': 200, 'level': 80}, 'bone': {'width': 1000, 'level': 400}}


def signature(raw_config, pooling, clamp):
    """Require explicit saved values; do not silently supply inference defaults."""
    required = set(POOL_FIELDS.values()) | set(PREPROCESS)
    if required - raw_config.keys():
        raise ValueError('Checkpoint lacks explicit inference fields')
    actual_pooling = {k: raw_config[v] for k, v in POOL_FIELDS.items()}
    actual_preprocess = {k: raw_config[k] for k in PREPROCESS}
    if actual_pooling != pooling or actual_preprocess != PREPROCESS or clamp != [0, 30]:
        raise ValueError('Inference contract mismatch (pooling/preprocessing/clipping)')
    if {k: WINDOWS[k] for k in EXPECTED_WINDOWS} != EXPECTED_WINDOWS:
        raise ValueError('Imaging windows changed')
    return {'pooling': actual_pooling, 'clamp_mm': list(clamp),
            'preprocessing': actual_preprocess, 'windows': EXPECTED_WINDOWS,
            'decoder': 'spatial_softmax_then_DARK', 'source_sha256': SOURCE_HASHES,
            'precision': 'float32_no_autocast', 'inference_batch_size': 6,
            'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'numpy': np.__version__}}


def require_signature(actual, expected):
    if actual != expected:
        raise ValueError('Baseline/candidate inference signatures differ')


def aggregate(slices, contract):
    if not slices or [s.index for s in slices] != list(range(len(slices))):
        raise ValueError('Incomplete or misordered decoded slices')
    for s in slices:
        values = [s.selector_probability, s.mls_mm, s.heatmap_peak]
        if s.peak_probability is not None: values.append(s.peak_probability)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite decoded values')
        if not 0 <= s.selector_probability <= 1 or (s.peak_probability is not None and not 0 <= s.peak_probability <= 1):
            raise ValueError('Invalid selector probability')
    value = aggregate_study_mls(slices, **contract['pooling'])
    if not math.isfinite(value): raise ValueError('Nonfinite study prediction')
    return float(np.clip(value, *contract['clamp_mm']))


def input_fingerprint(reader):
    uids = [str(getattr(s, 'SOPInstanceUID', '')) for s in reader.slices]
    if not uids or any(not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError('Missing/duplicate SOPInstanceUID')
    if len(reader.dicom_files) != len(uids):
        raise ValueError('Reader dropped raw files')
    content = hashlib.sha256()
    for name in sorted(reader.dicom_files):
        path = Path(name)
        content.update(path.name.encode() + b'\0' + _sha256(path).encode() + b'\n')
    return {'slice_count': len(uids), 'ordered_sop_uid_sha256': hashlib.sha256('\n'.join(uids).encode()).hexdigest(),
            'raw_files_sha256': content.hexdigest()}


def validate_private(rows, expected_ids):
    if len(rows) != len(expected_ids) or {r['study_id'] for r in rows} != set(expected_ids):
        raise ValueError('Private reference coverage is not exact')
    if any(not math.isfinite(r['MLS_mm']) or not math.isfinite(r['gt_MLS_mm']) for r in rows):
        raise ValueError('Nonfinite private reference')
    return {r['study_id']: r for r in rows}


def run(args):
    if args.output_dir.exists(): raise FileExistsError('Refusing output overwrite; no implicit resume')
    if _sha256(CORRECTION) != CORRECTION_SHA: raise ValueError('Correction contract changed')
    spec = json.loads(CORRECTION.read_text())
    for p, digest in [(ROOT / p, h) for p, h in SOURCE_HASHES.items()] + [
        (TRUTH, TRUTH_SHA), (Path(spec['fold_manifest']), spec['fold_manifest_sha256']),
        (Path(spec['baseline_reference_summary']), spec['baseline_reference_summary_sha256']),
        (Path(spec['baseline_reference_private']), spec['baseline_reference_private_sha256'])]:
        if _sha256(p) != digest: raise ValueError('Pinned source/reference mismatch')
    checkpoint = args.checkpoint.resolve()
    if _sha256(checkpoint) != args.checkpoint_sha256: raise ValueError('Checkpoint digest differs from enrollment')
    baseline_sha = spec['inputs'][0]['checkpoint_sha256']
    if args.baseline_self_test and args.checkpoint_sha256 != baseline_sha: raise ValueError('Wrong baseline checkpoint')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    raw = payload['config']
    config = MLSHeatmapConfig.model_validate(raw)
    if payload['epoch'] != 15 or config.fold != 0 or config.seed != 42 or not config.use_competition_folds:
        raise ValueError('Requires heldout fold0/seed42/epoch15 competition checkpoint')
    inference = signature(raw, spec['canonical_pooling'], spec['clamp_mm'])
    del payload
    folds = load_fold_manifest(Path(spec['fold_manifest']))
    heldout = folds.loc[folds.fold == 0]
    if set(heldout.patient_id) & set(folds.loc[folds.fold != 0, 'patient_id']):
        raise ValueError('Patient leakage in immutable folds')
    ids = sorted(heldout.study_id.astype(str))
    if len(ids) != 70: raise ValueError('Wrong fold0 coverage')
    truth_frame = pd.read_csv(TRUTH, dtype={'dicom_series.id': str}).set_index('dicom_series.id')
    if not truth_frame.index.is_unique: raise ValueError('Duplicate official truth')
    truth = {k: float(truth_frame.loc[k, 'MLS_mm']) for k in ids}
    if any(not math.isfinite(v) for v in truth.values()): raise ValueError('Missing/nonfinite truth')
    with Path(spec['baseline_reference_private']).open(newline='') as f:
        old = list(csv.DictReader(f))
    if len(old) != 70 or {r['study_id'] for r in old} != set(ids): raise ValueError('Wrong baseline reference coverage')
    reference = {r['study_id']: float(r['seed42_MLS_mm']) for r in old}
    if not all(math.isfinite(v) for v in reference.values()): raise ValueError('Nonfinite immutable reference')
    for r in old:
        if abs(float(r['gt_MLS_mm']) - truth[r['study_id']]) > 1e-8:
            raise ValueError('Reference/official truth differs')
    verified = None
    if not args.baseline_self_test:
        if not args.verified_baseline_dir or not args.verified_baseline_sha256:
            raise ValueError('Candidate requires checksum-bound verified baseline')
        result_path = args.verified_baseline_dir / 'aggregate_summary.json'
        if _sha256(result_path) != args.verified_baseline_sha256: raise ValueError('Verified baseline result changed')
        b = json.loads(result_path.read_text())
        if b['status'] != 'completed' or not b['baseline_self_test'] or b['checkpoint_sha256'] != baseline_sha or b['source_sha256'] != _sha256(Path(__file__)):
            raise ValueError('Baseline verification has wrong identity/source')
        require_signature(inference, b['inference_signature'])
        private = args.verified_baseline_dir / 'study_predictions_private.json'
        if _sha256(private) != b['private_predictions_sha256']: raise ValueError('Private baseline fingerprint changed')
        verified = validate_private(json.loads(private.read_text()), ids)
        if any(abs(verified[k]['MLS_mm'] - reference[k]) > 1e-5 or verified[k]['gt_MLS_mm'] != truth[k] for k in ids):
            raise ValueError('Verified baseline no longer matches immutable reference')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU model fallback')
    lock = CAMPAIGN / 'gpu_training.lock'
    lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('Concurrent GPU workload')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        status = args.output_dir / 'status.json'
        _atomic_json(status, {'status': 'running', 'completed_studies': 0, 'expected_studies': 70})
        model, loaded_config = load_multitask_model(checkpoint, torch.device('cuda:0'))
        require_signature(signature(loaded_config.model_dump(), spec['canonical_pooling'], spec['clamp_mm']), inference)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        rows = []
        for sid in ids:
            reader = BrainDicomReader(str(ROOT / 'Data/raw/training' / sid)).load_and_sort()
            fingerprint = input_fingerprint(reader)
            if verified and fingerprint != verified[sid]['input_fingerprint']:
                raise ValueError('Raw input or ordered SOP identity differs from baseline')
            slices = predict_reader_slices(reader, model, loaded_config, torch.device('cuda:0'), batch_size=6)
            if len(slices) != fingerprint['slice_count']: raise ValueError('Incomplete CUDA inference')
            prediction = aggregate(slices, inference)
            if args.baseline_self_test and abs(prediction - reference[sid]) > 1e-5:
                raise ValueError('Baseline per-study reproduction failed')
            rows.append({'study_id': sid, 'gt_MLS_mm': truth[sid], 'MLS_mm': prediction,
                         'input_fingerprint': fingerprint, 'slice_predictions': [asdict(s) for s in slices]})
            # Private records kept for resumption investigations, never stdout/MLflow.
            _atomic_json(args.output_dir / 'study_predictions_private.json', rows)
            _atomic_json(status, {'status': 'running', 'completed_studies': len(rows), 'expected_studies': 70})
        validate_private(rows, ids)
        observed = _metrics(np.array([truth[k] for k in ids]), np.array([r['MLS_mm'] for r in rows]))
        baseline_metrics = json.loads(Path(spec['baseline_reference_summary']).read_text())['member_metrics']['seed42']
        if args.baseline_self_test and any(abs(observed[k] - baseline_metrics[k]) > 1e-6 for k in observed):
            raise ValueError('Baseline aggregate reproduction failed')
        gates = gate(observed, spec['gate_bounds_unchanged'], spec['comparison_tolerance'])
        result = {'status': 'completed', 'scope': 'canonical_fold0_seed42_resource_screen_only',
                  'baseline_self_test': args.baseline_self_test, 'checkpoint_sha256': args.checkpoint_sha256,
                  'checkpoint': str(checkpoint), 'fold': 0, 'seed': 42, 'fixed_epoch': 15, 'studies': 70,
                  'compute_policy': 'cuda_only_no_cpu_model_fallback', 'inference_signature': inference,
                  'source_sha256': _sha256(Path(__file__)), 'correction_protocol_sha256': CORRECTION_SHA,
                  'truth_sha256': TRUTH_SHA, 'fold_manifest_sha256': spec['fold_manifest_sha256'],
                  'reference_summary_sha256': spec['baseline_reference_summary_sha256'],
                  'verified_baseline_sha256': args.verified_baseline_sha256,
                  'private_predictions_sha256': _sha256(args.output_dir / 'study_predictions_private.json'),
                  'baseline_metrics': baseline_metrics, 'observed': observed, 'gate_results': gates,
                  'resource_gates_passed': bool(all(gates.values()) and not args.baseline_self_test),
                  'runtime_seconds': time.monotonic() - started, 'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                  'automatic_replication_allowed': False, 'promotion_eligible': False, 'submission_zip_allowed': False}
        _atomic_json(args.output_dir / 'aggregate_summary.json', result)
        _atomic_json(status, {'status': 'completed', 'exit_code': 0, 'completed_studies': 70})
        return result
    except Exception as exc:
        if args.output_dir.exists():
            _atomic_json(args.output_dir / 'status.json', {'status': 'failed', 'exit_code': 1, 'error_type': type(exc).__name__})
        raise
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-self-test', action='store_true')
    parser.add_argument('--verified-baseline-dir', type=Path)
    parser.add_argument('--verified-baseline-sha256')
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({k: result[k] for k in ['status', 'baseline_self_test', 'observed', 'resource_gates_passed', 'promotion_eligible']}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'reason': str(exc)}))
        raise SystemExit(1)


if __name__ == '__main__': main()
