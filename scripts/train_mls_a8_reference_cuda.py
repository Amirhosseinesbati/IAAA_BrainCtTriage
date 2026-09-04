"""Fixed A8 ordinary-view control versus reference-refinement training."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_mls_a7_paired_cuda import loaders,move_batch
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json,_sha256
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision
from scripts.qualify_mls_refinement_runtime import load_qualified_reference,CORRECTION
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import (
    multitask_loss,configure_training_determinism,seed_training_epoch,
    _atomic_torch_save,_capture_rng_state,_restore_rng_state)

BASE=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK=BASE/'a8_reference_refinement_20260904'
MANIFEST=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A8_TRAINING_PROTOCOL_20260904.json'
INIT=BASE/'a7_paired_translation_20260904/initialization.pth'

def setup():
    spec=json.loads(MANIFEST.read_text())
    for name,digest in spec['source_and_input_sha256'].items():
        if _sha256(ROOT/name)!=digest:raise ValueError('Protocol input/source changed')
    if _sha256(INIT)!=spec['initialization_sha256']:raise ValueError('Initialization changed')
    load_qualified_reference(BASE/'reference_refinement_runtime_qualified_20260904.json',spec['runtime_reference_sha256'],CORRECTION)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    configure_inference_precision();configure_training_determinism('strict')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    payload=torch.load(INIT,map_location='cpu',weights_only=False)
    if payload['initialization']!='imagenet_backbone_random_heads_seed42_no_mls_training':raise ValueError('Wrong initialization type')
    config=MLSHeatmapConfig.model_validate(payload['config'])
    if (config.fold,config.seed,config.batch_size,config.epochs)!=(0,42,5,23) or config.use_amp or config.gradient_accumulation_steps!=1:
        raise ValueError('Wrong baseline recipe')
    if config.selector_head_mode!='single' or config.training_geometry_decoder!='global_softargmax':raise ValueError('Wrong geometry/selector')
    if any([config.use_ordinal_aux_head,config.signed_offset_loss_weight,config.study_bag_loss_weight,config.within_study_rank_loss_weight]):raise ValueError('Unexpected auxiliary loss')
    return spec,config,payload['model_state_dict']

def train_arm(spec,config,initial,arm,resume):
    config=MLSHeatmapConfig.model_validate({**config.model_dump(),'use_reference_refinement':arm=='refinement'})
    out=WORK/arm
    if out.exists() and not resume:raise FileExistsError('No implicit overwrite/resume')
    if resume and not (out/'recovery.pth').exists():raise FileNotFoundError('Missing recovery')
    out.mkdir(parents=True,exist_ok=resume)
    started=time.monotonic()
    from mlflow.tracking import MlflowClient
    from src.mlops.tracking import configure_tracking_environment
    configure_tracking_environment();client=MlflowClient()
    model=HRNetHeatmapModel(backbone_name=config.backbone,in_channels=3,pretrained=False,
        head_dropout=config.head_dropout,use_selector=True,use_reference_refinement=config.use_reference_refinement).cuda()
    missing,unexpected=model.load_state_dict(initial,strict=False)
    expected={k for k in model.state_dict() if k.startswith('outer_refinement.')}
    if set(missing)!=expected or unexpected:raise ValueError('Shared initialization not exact')
    if any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items() if k in initial):
        raise ValueError('Shared parameter/buffer mismatch')
    loader=loaders(config,True)
    optimizer=AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=LambdaLR(optimizer,lambda e:(e+1)/2 if e<2 else .5*(1+math.cos(math.pi*min((e-2)/21,1))))
    history=[];first=1
    if resume:
        state=torch.load(out/'recovery.pth',map_location='cpu',weights_only=False)
        if state['manifest_sha256']!=_sha256(MANIFEST) or state['arm']!=arm or state['config']!=config.model_dump():raise ValueError('Recovery differs')
        model.load_state_dict(state['model_state_dict'],strict=True);optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict']);_restore_rng_state(state['rng_state'])
        history=state['history'];first=state['epoch']+1;run_id=state['mlflow_run_id']
        if first>15:raise ValueError('Training already complete')
    else:
        experiment=client.get_run('8478b358f7b84f47b41f3b0ca882152d').info.experiment_id
        run_id=client.create_run(experiment,tags={'mlflow.runName':'mls-a8-'+arm+'-fold0-seed42',
            'arm':arm,'promotion_eligible':'false','compute_policy':'cuda_only_no_cpu_model_fallback'}).info.run_id
        client.log_artifact(run_id,str(MANIFEST),'protocol')
        for k,v in {'manifest_sha256':_sha256(MANIFEST),'initialization_sha256':spec['initialization_sha256'],
            'batch_size':5,'fixed_epoch':15,'scheduler_horizon':23,'use_reference_refinement':config.use_reference_refinement}.items():client.log_param(run_id,k,v)
    _atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),'mlflow_run_id':run_id,'arm':arm})
    try:
        for epoch in range(first,16):
            seed_training_epoch(42,epoch);model.train();values=[];digest=hashlib.sha256();epoch_start=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for batch in loader:
                digest.update(batch[0].numpy().tobytes());digest.update(batch[3].numpy().tobytes())
                image,target_h,masks,coords,spacing,target,study_mls,_=move_batch(batch)
                if image.device.type!='cuda':raise RuntimeError('GPU forward only')
                optimizer.zero_grad(set_to_none=True)
                heatmap,selector=model.forward_multitask(image)
                loss,_=multitask_loss(heatmap,selector,target_h,masks,coords,spacing,target,study_mls,config)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                values.append(float(loss.detach()))
            if len(values)!=541:raise ValueError('Optimizer exposure changed')
            scheduler.step()
            row={'epoch':epoch,'optimizer_steps':len(values),'input_exposure_sha256':digest.hexdigest(),
                'train_loss':float(np.mean(values)),'seconds':time.monotonic()-epoch_start,
                'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30}
            history.append(row)
            state={'schema_version':8,'epoch':epoch,'arm':arm,'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'rng_state':_capture_rng_state(),
                'config':config.model_dump(),'history':history,'manifest_sha256':_sha256(MANIFEST),
                'initialization_sha256':spec['initialization_sha256'],'mlflow_run_id':run_id,
                'checkpoint_selection':'fixed_epoch15_no_validation_selection'}
            _atomic_torch_save(state,out/'recovery.pth');_atomic_json(out/'training_history.json',history)
            for key in ['train_loss','seconds','peak_vram_gib']:client.log_metric(run_id,key,row[key],step=epoch)
        checkpoint=out/'mls_multitask_epoch_015.pth'
        _atomic_torch_save({k:v for k,v in state.items() if k not in ['optimizer_state_dict','scheduler_state_dict','rng_state','history']},checkpoint)
        result={'status':'completed','arm':arm,'epochs_completed':15,'optimizer_steps':sum(r['optimizer_steps'] for r in history),
            'checkpoint':str(checkpoint),'checkpoint_sha256':_sha256(checkpoint),'mlflow_run_id':run_id,
            'manifest_sha256':_sha256(MANIFEST),'initialization_sha256':spec['initialization_sha256'],
            'exposure_sha256_by_epoch':[r['input_exposure_sha256'] for r in history],
            'runtime_seconds':time.monotonic()-started,'validation_images_used':0,'promotion_eligible':False}
        _atomic_json(out/'training_summary.json',result)
        for name in ['training_summary.json','training_history.json']:client.log_artifact(run_id,str(out/name),'reports')
        client.log_artifact(run_id,str(checkpoint),'checkpoints');client.set_terminated(run_id,status='FINISHED')
        _atomic_json(out/'status.json',{'status':'completed','pid':os.getpid(),'mlflow_run_id':run_id})
    except Exception as exc:
        _atomic_json(out/'status.json',{'status':'failed','pid':os.getpid(),'error_type':type(exc).__name__,'mlflow_run_id':run_id})
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control','refinement'])
    parser.add_argument('--resume',action='store_true');parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.validate_only:
        spec,config,initial=setup();print(json.dumps({'status':'protocol_validated','initialization_keys':len(initial)}));return
    if args.arm is None:raise ValueError('Arm required')
    lock=BASE/'gpu_training.lock';lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():raise RuntimeError('Concurrent GPU workload')
        if shutil.disk_usage(BASE).free<15*2**30:raise RuntimeError('Need 15GiB free')
        spec,config,initial=setup();train_arm(spec,config,initial,args.arm,args.resume)
    finally:lock.rmdir()

if __name__=='__main__':main()
