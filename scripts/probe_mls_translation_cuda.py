"""Fixed, training-only translation diagnostic. No training or release decisions."""
from __future__ import annotations

import argparse
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
from scripts.probe_mls_training_decoders_cuda import (
    EXPECTED_CHECKPOINT, EXPECTED_LABELS, EXPECTED_FOLDS, select_rows, sha256,
)
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision, precision_flags
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model
from src.strategies.mls_heatmap.utils import decode_heatmap_dark_batch

SAMPLE_SHA = "b917bc5958d7bac9bf73ef401106524989eee33b52c800d4460e56de48c3d6a1"
CAMPAIGN = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
PROTOCOL = ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/TRANSLATION_PROBE_PROTOCOL_20260904.md'
SHIFTS = ((8, 0), (0, 8))


def translate(x, dx, dy):
    """Positive integer shifts, zero fill, never wrap pixels."""
    if not isinstance(dx, int) or not isinstance(dy, int) or dx < 0 or dy < 0:
        raise ValueError('Expected nonnegative integer shifts')
    h, w = x.shape[-2:]
    if dx >= w or dy >= h:
        raise ValueError('Shift exceeds image dimensions')
    out = torch.zeros_like(x)
    out[..., dy:, dx:] = x[..., :h-dy, :w-dx]
    return out


def mls(coords, spacing):
    a, b, c = coords[:, 0], coords[:, 1], coords[:, 2]
    direction = b-a
    length = np.linalg.norm(direction, axis=1)
    if np.any(length < 1e-6):
        raise ValueError('Degenerate midline')
    return np.abs(direction[:, 0]*(a[:, 1]-c[:, 1]) - (a[:, 0]-c[:, 0])*direction[:, 1]) / length * spacing


def aligned_js(base, moved, dx, dy):
    """Compare overlap in heatmap coordinates, renormalizing its mass."""
    h, w = base.shape[-2:]
    p = base[..., :h-dy, :w-dx].flatten(2)
    q = moved[..., dy:, dx:].flatten(2)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    mid = (p+q)*0.5
    return (0.5*(p*(p.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log()) +
                 q*(q.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log()))).sum(-1)


def describe(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite diagnostic values')
    return {'mean': float(x.mean()), 'median': float(np.median(x)),
            'p90': float(np.quantile(x, .9)), 'max': float(x.max())}


def run(output):
    if output.exists() or output.with_suffix('.status.json').exists():
        raise FileExistsError('Refusing to overwrite diagnostic')
    lock = CAMPAIGN/'gpu_training.lock'
    lock.mkdir()
    started = time.monotonic()
    try:
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('Another CUDA workload is active')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU model fallback')
        _atomic_json(output.with_suffix('.status.json'), {'status': 'running'})
        configure_inference_precision()
        checkpoint = ROOT/'models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth'
        labels = ROOT/'Data/processed/mls_multitask_v2/mls_labels_multitask.csv'
        folds = ROOT/'config/folds.csv'
        raw = ROOT/'Data/raw/training_df.pkl'
        for path, expected in ((checkpoint, EXPECTED_CHECKPOINT), (labels, EXPECTED_LABELS), (folds, EXPECTED_FOLDS),
                (raw, '0e00255ce7dcd6963a00db6c4d5a5dfdc5cff5a6fa8aec6ead5cc5da185cfe7d')):
            if sha256(path) != expected:
                raise ValueError('Pinned input changed')
        model, config = load_multitask_model(checkpoint, torch.device('cuda:0'))
        if config.fold != 0 or config.seed != 42 or config.image_size != 512 or not config.use_competition_folds:
            raise ValueError('Unexpected baseline configuration')
        train, val = create_mls_dataloaders(str(labels), str(labels.parent/'images'),
            img_size=512, heatmap_size=128, heatmap_sigma=config.heatmap_sigma,
            batch_size=8, augment=False, num_workers=0, fold=0, seed=42,
            use_competition_folds=True, include_negatives=True, return_selector=True)
        indices, sample_sha = select_rows(train.dataset.data, set(val.dataset.data['patient_id'].astype(str)), 512)
        if sample_sha != SAMPLE_SHA or len(indices) != 128:
            raise ValueError('Frozen training sample changed')
        loader = DataLoader(Subset(train.dataset, indices), batch_size=8, shuffle=False, num_workers=2)
        collected = {s: {'coordinate': [], 'mls': [], 'selector': [], 'js': [],
                         'base_error': [], 'moved_error': [], 'cross3': [], 'cross5': [],
                         'clipped_truth': 0, 'cropped_energy': []} for s in SHIFTS}
        def forward(images):
            if images.device.type != 'cuda':
                raise RuntimeError('CUDA forward only')
            logits, selector = model.forward_multitask(images)
            if not torch.isfinite(logits).all() or not torch.isfinite(selector).all():
                raise ValueError('Nonfinite model output')
            prob = logits.flatten(2).softmax(-1).reshape_as(logits)
            coords, _ = decode_heatmap_dark_batch(prob.cpu(), 128, 512)
            if not np.isfinite(coords).all() or np.any(coords < 0):
                raise ValueError('Invalid DARK output')
            return prob, coords, selector.sigmoid().cpu().numpy()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for images, _, masks, truth, spacing, target, *_ in loader:
                if not (masks > .5).all() or not (target > .5).all():
                    raise ValueError('Invalid positive annotation')
                images = images.cuda()
                spacing = spacing.numpy()
                truth = truth.numpy()
                p0, c0, s0 = forward(images)
                m0, gt = mls(c0, spacing), mls(truth, spacing)
                for dx, dy in SHIFTS:
                    moved = translate(images, dx, dy)
                    p1, c1, s1 = forward(moved)
                    aligned = c1 - np.array([dx, dy])
                    m1 = mls(aligned, spacing)
                    rec = collected[(dx, dy)]
                    rec['coordinate'].extend((np.linalg.norm(aligned-c0, axis=2)*spacing[:, None]).tolist())
                    rec['mls'].extend(np.abs(m1-m0).tolist())
                    rec['selector'].extend(np.abs(s1-s0).tolist())
                    rec['js'].extend(aligned_js(p0, p1, dx//4, dy//4).cpu().numpy().tolist())
                    rec['base_error'].extend(np.abs(m0-gt).tolist())
                    rec['moved_error'].extend(np.abs(m1-gt).tolist())
                    rec['cross3'].extend(((m0 >= 3) != (m1 >= 3)).tolist())
                    rec['cross5'].extend(((m0 >= 5) != (m1 >= 5)).tolist())
                    rec['clipped_truth'] += int(np.any(truth+np.array([dx, dy]) >= 512, axis=(1, 2)).sum())
                    energy = images.abs().sum((1, 2, 3)).clamp_min(1e-12)
                    rec['cropped_energy'].extend(((images.abs().sum((1, 2, 3))-moved.abs().sum((1, 2, 3)))/energy).cpu().tolist())
        summaries = {}
        for shift, rec in collected.items():
            summaries[str(shift)] = {
                'landmark_equivariance_error_mm': describe(rec['coordinate']),
                'absolute_mls_change_mm': describe(rec['mls']),
                'absolute_selector_probability_change': describe(rec['selector']),
                'overlap_heatmap_js_nats': describe(rec['js']),
                'baseline_slice_mae_mm': float(np.mean(rec['base_error'])),
                'translated_slice_mae_mm': float(np.mean(rec['moved_error'])),
                'prediction_crossings_3mm': int(sum(rec['cross3'])),
                'prediction_crossings_5mm': int(sum(rec['cross5'])),
                'clipped_ground_truth_examples': rec['clipped_truth'],
                'cropped_input_absolute_energy_fraction': describe(rec['cropped_energy']),
            }
        signal = any(v['absolute_mls_change_mm']['mean'] > .1 or v['absolute_mls_change_mm']['p90'] > .5 for v in summaries.values())
        result = {'status': 'completed', 'scope': 'training_only_translation_mechanism',
            'sample_count': 128, 'sample_sha256': sample_sha,
            'represented_studies': int(train.dataset.data.iloc[indices]['patient_id'].nunique()),
            'checkpoint_sha256': EXPECTED_CHECKPOINT, 'protocol_sha256': sha256(PROTOCOL),
            'source_sha256': sha256(Path(__file__)), 'precision_flags': precision_flags(),
            'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0),
            'validation_images_used': 0, 'model_updates': 0, 'compute_policy': 'cuda_only_no_cpu_model_fallback',
            'shifts': summaries, 'material_invariance_violation': signal,
            'automatic_training_allowed': False, 'promotion_eligible': False, 'submission_zip_allowed': False,
            'runtime_seconds': time.monotonic()-started, 'peak_vram_gib': torch.cuda.max_memory_allocated()/2**30}
        _atomic_json(output, result)
        _atomic_json(output.with_suffix('.status.json'), {'status': 'completed', 'exit_code': 0})
        print(json.dumps(result))
    except Exception as exc:
        _atomic_json(output.with_suffix('.status.json'), {'status': 'failed', 'error_type': type(exc).__name__})
        raise
    finally:
        lock.rmdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
