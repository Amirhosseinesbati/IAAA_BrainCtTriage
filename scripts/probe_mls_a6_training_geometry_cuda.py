"""Fixed training-only baseline/A6 geometry postmortem; aggregate outputs only."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_a2_fold0_resource_screen import _atomic_json
from scripts.probe_mls_training_decoders_cuda import EXPECTED_CHECKPOINT, EXPECTED_LABELS, EXPECTED_FOLDS, select_rows, sha256, summarize
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.geometry_decoding import local_softargmax_keypoints
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model
from src.strategies.mls_heatmap.train import differentiable_keypoints_from_heatmaps
from src.strategies.mls_heatmap.train_multitask import configure_training_determinism
from src.strategies.mls_heatmap.utils import decode_heatmap_dark_batch

CAMPAIGN = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
A6_SHA = '88b094341b260d48c90a2a1e12772c5bd5d82ac898e509db7ed7762d0b44aec6'
SAMPLE_SHA = 'b917bc5958d7bac9bf73ef401106524989eee33b52c800d4460e56de48c3d6a1'
RUN_ID = '9b8e9fc5996a42549e3aca5aa40763d7'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status_path = args.output.with_suffix('.status.json')
    if args.output.exists() or status_path.exists():
        raise FileExistsError('Refusing to overwrite diagnostic')
    lock = CAMPAIGN / 'gpu_training.lock'
    acquired = False
    started = time.monotonic()
    try:
        lock.mkdir()
        acquired = True
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('Concurrent GPU workload')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU model fallback')
        _atomic_json(status_path, {'state': 'running'})
        configure_training_determinism('strict')
        labels = ROOT / 'Data/processed/mls_multitask_v2/mls_labels_multitask.csv'
        folds = ROOT / 'config/folds.csv'
        reference_path = CAMPAIGN / 'baseline_train_decoder_probe_20260904.json'
        for path, expected in [(labels, EXPECTED_LABELS), (folds, EXPECTED_FOLDS), (reference_path, '030edeabf30710151e3e5cb6a2b0cb47dd8ac7c3a012d09be56b54462f72d2fb')]:
            if sha256(path) != expected:
                raise ValueError('Input checksum mismatch')
        reference = json.loads(reference_path.read_text())
        for relative, key in [('src/strategies/mls_heatmap/dataset.py', 'dataset_source_sha256'), ('src/strategies/mls_heatmap/predict_multitask.py', 'model_loader_source_sha256'), ('src/strategies/mls_heatmap/train.py', 'softargmax_source_sha256'), ('src/strategies/mls_heatmap/utils.py', 'dark_source_sha256')]:
            if sha256(ROOT / relative) != reference[key]:
                raise ValueError('Diagnostic reference implementation changed')
        train, val = create_mls_dataloaders(str(labels), str(labels.parent / 'images'), img_size=512, heatmap_size=128, heatmap_sigma=3., batch_size=8, augment=False, num_workers=0, fold=0, seed=42, use_competition_folds=True, include_negatives=True, return_selector=True)
        indices, sample_sha = select_rows(train.dataset.data, set(val.dataset.data['patient_id'].astype(str)), 512)
        if sample_sha != SAMPLE_SHA or len(indices) != 128:
            raise ValueError('Frozen training sample changed')
        loader = DataLoader(Subset(train.dataset, indices), batch_size=8, shuffle=False, num_workers=2)
        checkpoints = {
            'baseline': ('mls-vast-exp16-w32-fold0-strict-ensemble-refresh', EXPECTED_CHECKPOINT),
            'a6': ('mls-vast-da-a6-local-geometry-fold0-seed42', A6_SHA),
        }
        results = {}
        for name, (run_name, expected) in checkpoints.items():
            checkpoint = ROOT / 'models/checkpoints/mls_multitask' / run_name / 'mls_multitask_epoch_015.pth'
            if sha256(checkpoint) != expected:
                raise ValueError('Checkpoint checksum mismatch')
            model, config = load_multitask_model(checkpoint, torch.device('cuda:0'))
            if config.fold != 0 or config.seed != 42 or config.image_size != 512 or not config.use_competition_folds:
                raise ValueError('Wrong checkpoint population')
            if name == 'a6' and (config.training_geometry_decoder != 'local_softargmax' or config.local_softargmax_radius != 6):
                raise ValueError('Unexpected A6 decoder')
            buffers = {k: [] for k in ['global', 'local', 'dark', 'truth', 'spacing', 'outside']}
            with torch.inference_mode():
                for images, _targets, masks, truth, spacing, is_target, _study_mls, _ids in loader:
                    if not bool((masks > .5).all()) or not bool((is_target > .5).all()):
                        raise ValueError('Invalid selected annotations')
                    logits, selector = model.forward_multitask(images.cuda())
                    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(selector).all()):
                        raise FloatingPointError('Nonfinite model outputs')
                    global_xy = differentiable_keypoints_from_heatmaps(logits, 512, 1.)
                    local_xy = local_softargmax_keypoints(logits, 512, 1., 6)
                    probabilities = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
                    dark_xy, _peaks = decode_heatmap_dark_batch(probabilities.cpu(), logits.shape[-1], 512)
                    flat_peak = logits.flatten(2).argmax(-1)
                    peak_xy = torch.stack([flat_peak % logits.shape[-1], flat_peak // logits.shape[-1]], dim=-1)
                    outside = ((truth.cuda() / 4 - peak_xy).abs() > 6).any(-1)
                    for key, value in [('global', global_xy.cpu().numpy()), ('local', local_xy.cpu().numpy()), ('dark', dark_xy), ('truth', truth.numpy()), ('spacing', spacing.numpy()), ('outside', outside.cpu().numpy())]:
                        buffers[key].append(value)
            merged = {k: np.concatenate(v) for k,v in buffers.items()}
            global_summary = summarize(merged['global'], merged['dark'], merged['truth'], merged['spacing'])
            local_summary = summarize(merged['local'], merged['dark'], merged['truth'], merged['spacing'])
            results[name] = {'checkpoint_sha256': expected, 'global_vs_dark': global_summary, 'local_vs_dark': local_summary, 'target_outside_local_window_fraction_by_landmark': merged['outside'].mean(0).tolist()}
            if name == 'baseline':
                for decoder in ['softargmax', 'dark']:
                    if abs(global_summary['decoders'][decoder]['slice_mls_mae_mm'] - reference['decoders'][decoder]['slice_mls_mae_mm']) > 1e-4:
                        raise ValueError('Baseline reproduction check failed')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result = {'status': 'completed', 'scope': 'training_only_postmortem_not_validation', 'compute_policy': 'cuda_only_no_cpu_model_fallback', 'sample_count': len(indices), 'represented_studies': int(train.dataset.data.iloc[indices]['patient_id'].nunique()), 'sample_sha256': sample_sha, 'labels_sha256': EXPECTED_LABELS, 'folds_sha256': EXPECTED_FOLDS, 'baseline_reproduction_passed': True, 'validation_images_used': 0, 'models': results, 'source_sha256': sha256(Path(__file__)), 'elapsed_seconds': time.monotonic() - started, 'promotion_eligible': False, 'a6_rejection_unchanged': True}
        _atomic_json(args.output, result)
        try:
            from mlflow.tracking import MlflowClient
            from src.mlops.tracking import configure_tracking_environment
            configure_tracking_environment()
            client = MlflowClient()
            for model_name, values in results.items():
                for mode in ['global_vs_dark', 'local_vs_dark']:
                    summary = values[mode]
                    client.log_metric(RUN_ID, f'postmortem_{model_name}_{mode}_mls_gap_mm', summary['inter_decoder_mls_mean_absolute_difference_mm'])
                    for decoder, stats in summary['decoders'].items():
                        client.log_metric(RUN_ID, f'postmortem_{model_name}_{mode}_{decoder}_mae_mm', stats['slice_mls_mae_mm'])
            client.log_artifact(RUN_ID, str(args.output), 'reports/a6_training_postmortem')
            result['mlflow'] = {'status': 'logged', 'run_id': RUN_ID}
        except Exception as exc:
            result['mlflow'] = {'status': 'deferred', 'error_type': type(exc).__name__}
        _atomic_json(args.output, result)
        _atomic_json(status_path, {'state': 'completed', 'exit_code': 0})
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        _atomic_json(status_path, {'state': 'failed', 'error_type': type(exc).__name__, 'exit_code': 1})
        print(json.dumps({'state': 'failed', 'error_type': type(exc).__name__}))
        return 1
    finally:
        if acquired:
            lock.rmdir()


if __name__ == '__main__':
    raise SystemExit(main())
