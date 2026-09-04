"""Preregistered paired-view control/consistency training; fixed epoch15 only."""
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
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mls_three_seed_fold_cuda import _atomic_json, _sha256
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision, precision_flags
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.dataset import create_mls_dataloaders
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.train_multitask import (
    multitask_loss, configure_training_determinism, seed_training_epoch,
    _atomic_torch_save, _capture_rng_state, _restore_rng_state,
)
from src.strategies.mls_heatmap.translation_consistency import (
    translate_image, translated_targets, consistency_js, combine_losses,
)

CAMPAIGN=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
WORK=CAMPAIGN/'a7_paired_translation_20260904'
MANIFEST=ROOT/'reports/mls_experiments/mls-deploy-aligned-upgrade-20260902/A7_PAIRED_TRAINING_PROTOCOL_20260904.json'
SHIFTS=((8,0),(-8,0),(0,8),(0,-8))


def setup():
    spec=json.loads(MANIFEST.read_text())
    for relative,digest in spec['source_sha256'].items():
        if _sha256(ROOT/relative)!=digest:raise ValueError('Pinned source changed: '+relative)
    for relative,digest in spec['input_sha256'].items():
        if _sha256(ROOT/relative)!=digest:raise ValueError('Pinned input changed: '+relative)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    configure_inference_precision()
    configure_training_determinism('strict')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    template=yaml.safe_load((ROOT/spec['template']).read_text())['training_config']
    config=MLSHeatmapConfig.model_validate(template)
    if config.fold!=0 or config.seed!=42 or config.batch_size!=5 or config.epochs!=23 or config.use_amp:
        raise ValueError('Unexpected training recipe')
    if config.training_geometry_decoder!='global_softargmax' or config.selector_head_mode!='single':
        raise ValueError('Unexpected model schema')
    return spec,config


def loaders(config, augment):
    data=ROOT/'Data/processed/mls_multitask_v2'
    train,val=create_mls_dataloaders(str(data/'mls_labels_multitask.csv'),str(data/'images'),
        img_size=512,heatmap_size=128,heatmap_sigma=3.,batch_size=5,
        augment=augment,rotation_deg=config.rotation_deg,translation=config.translation,
        intensity_jitter_scale=config.intensity_jitter,augment_prob=config.augment_prob,
        num_workers=2,seed=42,fold=0,use_competition_folds=True,include_negatives=True,
        return_selector=True,balanced_sampling=True,sampling_mode=config.sampling_mode,
        deterministic_workers=True)
    if len(train.dataset)!=2706 or len(train)!=541:raise ValueError('Training population changed')
    if set(train.dataset.data.patient_id)&set(val.dataset.data.patient_id):raise ValueError('Study leakage')
    return train


def make_model(config, pretrained):
    return HRNetHeatmapModel(backbone_name=config.backbone,in_channels=3,num_keypoints=3,
        pretrained=pretrained,head_dropout=config.head_dropout,use_selector=True,
        selector_head_mode='single',use_ordinal_aux_head=False).cuda()


def paired_loss(model, batch, config, shift, weight):
    images,targets,masks,coords,spacing,target,study_mls,_=batch
    if images.device.type!='cuda':raise RuntimeError('CPU model forward forbidden')
    dx,dy=shift
    moved_coords,moved_targets,eligible,positive=translated_targets(coords,masks,target,dx,dy,512,3.)
    heatmaps,selector=model.forward_multitask(torch.cat([images,translate_image(images,dx,dy)],0))
    if not torch.isfinite(heatmaps).all() or not torch.isfinite(selector).all():
        raise FloatingPointError('Nonfinite CUDA forward')
    n=len(images)
    first,_=multitask_loss(heatmaps[:n],selector[:n],targets,masks,coords,spacing,target,study_mls,config)
    if eligible.any():
        second,_=multitask_loss(heatmaps[n:][eligible],selector[n:][eligible],moved_targets[eligible],
            masks[eligible],moved_coords[eligible],spacing[eligible],target[eligible],study_mls[eligible],config)
    else:
        second=first # No second-view eligible labels: retain original supervised scale.
    consistency=consistency_js(heatmaps[:n],heatmaps[n:],positive,dx,dy)
    total=combine_losses(first,second,consistency,weight)
    return total,{'supervised':(first+second)*.5,'consistency':consistency,
                  'eligible':eligible.sum(),'positive':positive.sum()},heatmaps


def move_batch(batch):
    return tuple(item.cuda(non_blocking=True) if torch.is_tensor(item) else item for item in batch)


def preflight(spec,config):
    output=WORK/'preflight.json'
    if WORK.exists():raise FileExistsError('Preflight directory already exists')
    WORK.mkdir()
    _atomic_json(WORK/'preflight_status.json',{'status':'running','pid':os.getpid()})
    started=time.monotonic()
    train=loaders(config,False)
    frame=train.dataset.data
    # Deterministic mixture; training only, no selection by model output.
    rows=list(frame.index[frame.is_target>.5][:3])+list(frame.index[frame.is_target<=.5][:2])
    if len(rows)!=5:raise ValueError('Preflight mixture unavailable')
    model=make_model(config,True)
    initial=WORK/'initialization.pth'
    _atomic_torch_save({'model_state_dict':model.state_dict(),'config':config.model_dump(),
                        'initialization':'imagenet_backbone_random_heads_seed42_no_mls_training'},initial)
    batch=move_batch(next(iter(DataLoader(Subset(train.dataset,rows),batch_size=5))))
    model.train()
    optimizer=AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
    torch.cuda.reset_peak_memory_stats()
    total,parts,heatmaps=paired_loss(model,batch,config,(8,0),10.)
    gsuper=torch.autograd.grad(parts['supervised'],heatmaps,retain_graph=True)[0]
    gcons=torch.autograd.grad(parts['consistency'],heatmaps,retain_graph=True)[0]
    ratio=float(10*gcons.norm()/gsuper.norm().clamp_min(1e-12))
    if not math.isfinite(ratio) or ratio>1:raise RuntimeError('Consistency output gradient dominates preflight')
    total.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Nonfinite gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite optimizer step')
    result={'status':'passed','manifest_sha256':_sha256(MANIFEST),'initialization_sha256':_sha256(initial),
        'source_sha256':spec['source_sha256'],'compute_policy':'cuda_only_no_cpu_model_fallback',
        'base_batch_size':5,'forward_batch_size':10,'optimizer_steps':1,'validation_images_used':0,
        'weighted_consistency_to_supervised_output_gradient_ratio':ratio,
        'eligible_second_view_samples':int(parts['eligible']),'valid_positive_samples':int(parts['positive']),
        'runtime_seconds':time.monotonic()-started,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,
        'precision_flags':precision_flags(),'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'model_updates_are_disposable':True,'promotion_eligible':False}
    _atomic_json(output,result)
    _atomic_json(WORK/'preflight_status.json',{'status':'completed','pid':os.getpid()})
    print(json.dumps(result))


def train_arm(spec,config,arm,resume):
    pre=json.loads((WORK/'preflight.json').read_text())
    if pre['status']!='passed' or pre['manifest_sha256']!=_sha256(MANIFEST):raise ValueError('No matching preflight')
    initial=WORK/'initialization.pth'
    if _sha256(initial)!=pre['initialization_sha256']:raise ValueError('Initialization changed')
    out=WORK/arm
    if out.exists() and not resume:raise FileExistsError('No implicit resume or overwrite')
    if resume and not (out/'recovery.pth').exists():raise FileNotFoundError('No exact recovery checkpoint')
    out.mkdir(exist_ok=resume)
    start=time.monotonic()
    _atomic_json(out/'status.json',{'status':'initializing','pid':os.getpid(),'arm':arm})
    from mlflow.tracking import MlflowClient
    from src.mlops.tracking import configure_tracking_environment
    configure_tracking_environment()
    client=MlflowClient()
    model=make_model(config,False)
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    loader=loaders(config,True)
    optimizer=AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    def rate(epoch):
        return (epoch+1)/2 if epoch<2 else .5*(1+math.cos(math.pi*min((epoch-2)/21,1)))
    scheduler=LambdaLR(optimizer,rate)
    history=[]
    first_epoch=1
    if resume:
        state=torch.load(out/'recovery.pth',map_location='cpu',weights_only=False)
        if state['manifest_sha256']!=_sha256(MANIFEST) or state['arm']!=arm:raise ValueError('Recovery protocol differs')
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict']); scheduler.load_state_dict(state['scheduler_state_dict'])
        _restore_rng_state(state['rng_state'])
        first_epoch=state['epoch']+1; history=state['history']; run_id=state['mlflow_run_id']
        if first_epoch>15:raise ValueError('Training already complete; do not restart')
    else:
        experiment=client.get_run('8478b358f7b84f47b41f3b0ca882152d').info.experiment_id
        run_id=client.create_run(experiment,tags={'mlflow.runName':'mls-a7-paired-'+arm+'-fold0-seed42',
            'promotion_eligible':'false','compute_policy':'cuda_only_no_cpu_model_fallback',
            'scope':'fixed_epoch15_resource_candidate','arm':arm}).info.run_id
        client.log_artifact(run_id,str(MANIFEST),'protocol')
        for k,v in {'manifest_sha256':_sha256(MANIFEST),'initialization_sha256':pre['initialization_sha256'],
                    'base_batch_size':5,'forward_batch_size':10,'fixed_epoch':15,'schedule_horizon':23,
                    'consistency_weight':10 if arm=='consistency' else 0}.items():client.log_param(run_id,k,v)
    _atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),'arm':arm,'mlflow_run_id':run_id})
    try:
        for epoch in range(first_epoch,16):
            seed_training_epoch(42,epoch)
            model.train()
            values=[]
            digest=hashlib.sha256()
            epoch_start=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            weight=(10*min(epoch/3,1)) if arm=='consistency' else 0.
            for step,batch in enumerate(loader):
                # Privacy-preserving audit of equal sample/augmentation exposure.
                digest.update(batch[0].numpy().tobytes())
                digest.update(batch[3].numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                total,parts,_=paired_loss(model,move_batch(batch),config,SHIFTS[(epoch-1+step)%4],weight)
                if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss')
                total.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                values.append([float(total.detach()),float(parts['supervised'].detach()),float(parts['consistency'].detach())])
            scheduler.step()
            means=np.mean(values,axis=0)
            row={'epoch':epoch,'optimizer_steps':len(values),'input_exposure_sha256':digest.hexdigest(),
                 'train_loss':float(means[0]),'supervised_loss':float(means[1]),'consistency_js':float(means[2]),
                 'weight':weight,'seconds':time.monotonic()-epoch_start,
                 'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30}
            history.append(row)
            state={'schema_version':7,'epoch':epoch,'arm':arm,'model_state_dict':model.state_dict(),
                   'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                   'rng_state':_capture_rng_state(),'config':config.model_dump(),'history':history,
                   'manifest_sha256':_sha256(MANIFEST),'initialization_sha256':pre['initialization_sha256'],
                   'mlflow_run_id':run_id,'checkpoint_selection':'fixed_epoch15_no_validation_selection'}
            _atomic_torch_save(state,out/'recovery.pth')
            _atomic_json(out/'training_history.json',history)
            for key in ['train_loss','supervised_loss','consistency_js','seconds','peak_vram_gib']:
                client.log_metric(run_id,key,row[key],step=epoch)
        checkpoint=out/'mls_multitask_epoch_015.pth'
        _atomic_torch_save({k:v for k,v in state.items() if k not in ['optimizer_state_dict','scheduler_state_dict','rng_state','history']},checkpoint)
        final={'status':'completed','arm':arm,'epochs_completed':15,'optimizer_steps':sum(r['optimizer_steps'] for r in history),
               'checkpoint':str(checkpoint),'checkpoint_sha256':_sha256(checkpoint),'mlflow_run_id':run_id,
               'manifest_sha256':_sha256(MANIFEST),'initialization_sha256':pre['initialization_sha256'],
               'exposure_sha256_by_epoch':[r['input_exposure_sha256'] for r in history],
               'runtime_seconds':time.monotonic()-start,'validation_images_used':0,
               'promotion_eligible':False,'submission_zip_allowed':False}
        _atomic_json(out/'training_summary.json',final)
        del model,optimizer,scheduler
        torch.cuda.empty_cache()
        client.log_artifact(run_id,str(out/'training_summary.json'),'reports')
        client.log_artifact(run_id,str(out/'training_history.json'),'reports')
        client.log_artifact(run_id,str(checkpoint),'checkpoints')
        client.set_terminated(run_id,status='FINISHED')
        _atomic_json(out/'status.json',{'status':'completed','pid':os.getpid(),'mlflow_run_id':run_id})
    except Exception as exc:
        _atomic_json(out/'status.json',{'status':'failed','pid':os.getpid(),'error_type':type(exc).__name__,'mlflow_run_id':run_id})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['preflight','train'],required=True)
    parser.add_argument('--arm',choices=['control','consistency'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='train' and not args.arm:raise ValueError('Training arm required')
    lock=CAMPAIGN/'gpu_training.lock'
    lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('Other CUDA workload active')
        if shutil.disk_usage(CAMPAIGN).free<15*2**30:raise RuntimeError('Insufficient free disk')
        spec,config=setup()
        if args.mode=='preflight':preflight(spec,config)
        else:train_arm(spec,config,args.arm,args.resume)
    finally:lock.rmdir()


if __name__=='__main__':main()
