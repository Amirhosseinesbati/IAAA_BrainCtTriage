"""A9: preserve the qualified baseline and train only the reference refiner."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision
from scripts.train_mls_a7_paired_cuda import move_batch
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import (
    _atomic_torch_save, _capture_rng_state, _restore_rng_state,
    configure_training_determinism, multitask_loss, seed_training_epoch,
)

BASE = Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK = BASE / 'a9_frozen_baseline_refiner_20260904'
MANIFEST = ROOT / 'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A9_TRAINING_PROTOCOL_20260904.json'
BASELINE = Path('/workspace/IAAA_BrainCtTriage/models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth')
EPOCHS = 10
BATCH_SIZE = 16
STEPS_PER_EPOCH = 169


def loader(config):
    data = ROOT / 'Data/processed/mls_multitask_v2'
    train, val = create_mls_dataloaders(
        str(data / 'mls_labels_multitask.csv'), str(data / 'images'),
        img_size=512, heatmap_size=128, heatmap_sigma=3., batch_size=BATCH_SIZE,
        augment=True, rotation_deg=config.rotation_deg, translation=config.translation,
        intensity_jitter_scale=config.intensity_jitter, augment_prob=config.augment_prob,
        num_workers=2, seed=42, fold=0, use_competition_folds=True,
        include_negatives=True, return_selector=True, balanced_sampling=True,
        sampling_mode=config.sampling_mode, deterministic_workers=True,
    )
    if len(train.dataset) != 2706 or len(train) != STEPS_PER_EPOCH:
        raise ValueError('A9 training population or batch exposure changed')
    if set(train.dataset.data.patient_id) & set(val.dataset.data.patient_id):
        raise ValueError('Patient leakage')
    return train


def setup():
    spec = json.loads(MANIFEST.read_text())
    for relative, digest in spec['source_and_input_sha256'].items():
        if _sha256(ROOT / relative) != digest:
            raise ValueError('Pinned A9 source/input changed: ' + relative)
    if _sha256(BASELINE) != spec['baseline_checkpoint_sha256']:
        raise ValueError('Qualified baseline checkpoint changed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU model fallback')
    configure_inference_precision()
    configure_training_determinism('strict')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    payload = torch.load(BASELINE, map_location='cpu', weights_only=False)
    base_config = MLSHeatmapConfig.model_validate(payload['config'])
    if (base_config.fold, base_config.seed, base_config.selector_head_mode) != (0, 42, 'single'):
        raise ValueError('Unexpected qualified baseline schema')
    if base_config.use_reference_refinement or base_config.use_ordinal_aux_head:
        raise ValueError('Baseline is not the declared plain multitask model')
    config = MLSHeatmapConfig.model_validate({**base_config.model_dump(), 'use_reference_refinement': True})
    model = HRNetHeatmapModel(
        backbone_name=config.backbone, in_channels=config.input_channels,
        num_keypoints=3, pretrained=False, head_dropout=config.head_dropout,
        use_selector=True, selector_head_mode='single', use_ordinal_aux_head=False,
        use_reference_refinement=True,
    ).cuda()
    missing, unexpected = model.load_state_dict(payload['model_state_dict'], strict=False)
    expected = {key for key in model.state_dict() if key.startswith('outer_refinement.')}
    if set(missing) != expected or unexpected:
        raise ValueError('Baseline-to-refiner initialization is not exact')
    for key, value in payload['model_state_dict'].items():
        if not torch.equal(model.state_dict()[key].detach().cpu(), value):
            raise ValueError('Baseline parameter/buffer changed during load')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('outer_refinement.'))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 47617:
        raise ValueError('Unexpected trainable parameter count')
    return spec, config, payload['model_state_dict'], model


def frozen_forward(model, images):
    if images.device.type != 'cuda':
        raise RuntimeError('GPU forward only')
    model.eval()
    model.outer_refinement.train()
    with torch.no_grad():
        features = model.backbone(images)[0]
        coarse = model.head(features)
        selector = model.selector_head(features).squeeze(1)
    refined = model.outer_refinement(features.detach(), coarse.detach())
    return refined, selector.detach(), coarse


def verify_frozen(model, baseline_state):
    state = model.state_dict()
    for key, expected in baseline_state.items():
        if not torch.equal(state[key].detach().cpu(), expected):
            raise ValueError('Frozen baseline state changed: ' + key)


def preflight():
    if WORK.exists():
        raise FileExistsError('A9 work directory already exists')
    spec, config, baseline_state, model = setup()
    train = loader(config)
    batch = move_batch(next(iter(train)))
    images, targets, masks, coords, spacing, target, study_mls, _ = batch
    with torch.no_grad():
        model.eval(); features = model.backbone(images)[0]; coarse = model.head(features)
        initialized = model.outer_refinement(features, coarse)
    if not torch.equal(initialized, coarse):
        raise ValueError('Zero-initialized refiner does not exactly preserve baseline')
    optimizer = AdamW(model.outer_refinement.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    refined, selector, _ = frozen_forward(model, images)
    loss, _ = multitask_loss(refined, selector, targets, masks, coords, spacing,
                             target, study_mls, config)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite A9 preflight loss')
    before = {k: v.detach().clone() for k, v in model.outer_refinement.state_dict().items()}
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.outer_refinement.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    if not any(not torch.equal(before[k], v) for k, v in model.outer_refinement.state_dict().items()):
        raise ValueError('Refiner did not update')
    verify_frozen(model, baseline_state)
    WORK.mkdir()
    result = {'status': 'completed', 'baseline_identity_at_initialization': True,
              'frozen_baseline_unchanged_after_step': True, 'refiner_updated': True,
              'trainable_parameters': 47617, 'batch_size': BATCH_SIZE,
              'cuda_only': True, 'validation_images_used': 0,
              'manifest_sha256': _sha256(MANIFEST),
              'baseline_checkpoint_sha256': spec['baseline_checkpoint_sha256'],
              'promotion_eligible': False}
    _atomic_json(WORK / 'preflight.json', result)
    print(json.dumps(result))


def train(resume):
    spec, config, baseline_state, model = setup()
    if not WORK.exists() or not (WORK / 'preflight.json').exists():
        raise FileNotFoundError('Successful A9 preflight required')
    out = WORK / 'candidate'
    if out.exists() and not resume:
        raise FileExistsError('No implicit overwrite/resume')
    if resume and not (out / 'recovery.pth').exists():
        raise FileNotFoundError('Missing explicit recovery checkpoint')
    out.mkdir(exist_ok=resume)
    train_loader = loader(config)
    optimizer = AdamW(model.outer_refinement.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    scheduler = LambdaLR(optimizer, lambda epoch: .5 * (1 + math.cos(math.pi * min(epoch / EPOCHS, 1))))
    from mlflow.tracking import MlflowClient
    from src.mlops.tracking import configure_tracking_environment
    configure_tracking_environment(); client = MlflowClient()
    history = []; first = 1; started = time.monotonic()
    if resume:
        state = torch.load(out / 'recovery.pth', map_location='cpu', weights_only=False)
        if state['manifest_sha256'] != _sha256(MANIFEST) or state['config'] != config.model_dump():
            raise ValueError('A9 recovery provenance differs')
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict']); scheduler.load_state_dict(state['scheduler_state_dict'])
        _restore_rng_state(state['rng_state']); history = state['history']; first = state['epoch'] + 1
        run_id = state['mlflow_run_id']
        if first > EPOCHS:
            raise ValueError('A9 already completed')
    else:
        experiment = client.get_run('8478b358f7b84f47b41f3b0ca882152d').info.experiment_id
        run_id = client.create_run(experiment, tags={
            'mlflow.runName': 'mls-a9-frozen-baseline-refiner-fold0-seed42',
            'promotion_eligible': 'false', 'compute_policy': 'cuda_only_no_cpu_model_fallback'}).info.run_id
        client.log_artifact(run_id, str(MANIFEST), 'protocol')
        for key, value in {'manifest_sha256': _sha256(MANIFEST),
                           'baseline_checkpoint_sha256': spec['baseline_checkpoint_sha256'],
                           'batch_size': BATCH_SIZE, 'fixed_epoch': EPOCHS,
                           'trainable_parameters': 47617, 'frozen_baseline': True}.items():
            client.log_param(run_id, key, value)
    _atomic_json(out / 'status.json', {'status': 'training', 'pid': os.getpid(), 'mlflow_run_id': run_id})
    try:
        for epoch in range(first, EPOCHS + 1):
            seed_training_epoch(42, epoch); values = []; digest = hashlib.sha256(); epoch_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for batch in train_loader:
                digest.update(batch[0].numpy().tobytes()); digest.update(batch[3].numpy().tobytes())
                images, targets, masks, coords, spacing, target, study_mls, _ = move_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                refined, selector, _ = frozen_forward(model, images)
                loss, _ = multitask_loss(refined, selector, targets, masks, coords, spacing,
                                         target, study_mls, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite A9 training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.outer_refinement.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); values.append(float(loss.detach()))
            if len(values) != STEPS_PER_EPOCH:
                raise ValueError('A9 optimizer exposure changed')
            scheduler.step(); verify_frozen(model, baseline_state)
            row = {'epoch': epoch, 'optimizer_steps': len(values),
                   'input_exposure_sha256': digest.hexdigest(), 'train_loss': float(np.mean(values)),
                   'seconds': time.monotonic() - epoch_start,
                   'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30}
            history.append(row)
            state = {'schema_version': 9, 'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'rng_state': _capture_rng_state(), 'config': config.model_dump(), 'history': history,
                     'manifest_sha256': _sha256(MANIFEST),
                     'baseline_checkpoint_sha256': spec['baseline_checkpoint_sha256'],
                     'mlflow_run_id': run_id, 'checkpoint_selection': 'fixed_epoch10_no_validation_selection'}
            _atomic_torch_save(state, out / 'recovery.pth'); _atomic_json(out / 'training_history.json', history)
            for key in ['train_loss', 'seconds', 'peak_vram_gib']:
                client.log_metric(run_id, key, row[key], step=epoch)
        checkpoint = out / 'mls_multitask_epoch_010.pth'
        _atomic_torch_save({k: v for k, v in state.items() if k not in
                           ['optimizer_state_dict', 'scheduler_state_dict', 'rng_state', 'history']}, checkpoint)
        result = {'status': 'completed', 'epochs_completed': EPOCHS,
                  'optimizer_steps': sum(row['optimizer_steps'] for row in history),
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(checkpoint),
                  'mlflow_run_id': run_id, 'manifest_sha256': _sha256(MANIFEST),
                  'baseline_checkpoint_sha256': spec['baseline_checkpoint_sha256'],
                  'runtime_seconds': time.monotonic() - started, 'validation_images_used': 0,
                  'frozen_baseline_verified': True, 'promotion_eligible': False}
        _atomic_json(out / 'training_summary.json', result)
        for name in ['training_summary.json', 'training_history.json']:
            client.log_artifact(run_id, str(out / name), 'reports')
        client.log_artifact(run_id, str(checkpoint), 'checkpoints'); client.set_terminated(run_id, status='FINISHED')
        _atomic_json(out / 'status.json', {'status': 'completed', 'pid': os.getpid(), 'mlflow_run_id': run_id})
    except Exception as exc:
        _atomic_json(out / 'status.json', {'status': 'failed', 'pid': os.getpid(),
                                          'error_type': type(exc).__name__, 'mlflow_run_id': run_id})
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    lock = BASE / 'gpu_training.lock'; lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('Concurrent GPU workload')
        if shutil.disk_usage(BASE).free < 15 * 2**30:
            raise RuntimeError('Need 15GiB free')
        if args.preflight:
            preflight()
        else:
            train(args.resume)
    finally:
        lock.rmdir()


if __name__ == '__main__':
    main()
