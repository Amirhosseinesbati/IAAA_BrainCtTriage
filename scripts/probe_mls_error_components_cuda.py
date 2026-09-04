"""Fixed training sample: geometric error attribution, never deployable oracles."""
from pathlib import Path
import sys
import json
import hashlib
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_mls_a7_translation_cuda import CHECKPOINTS, state_digest
from scripts.probe_mls_translation_cuda import (
    CAMPAIGN,SAMPLE_SHA,mls,describe,select_rows,sha256,create_mls_dataloaders,
    load_multitask_model,configure_inference_precision,precision_flags,
    decode_heatmap_dark_batch,_atomic_json,EXPECTED_LABELS,EXPECTED_FOLDS)

OUTPUT=CAMPAIGN/'training_error_components_20260904.json'
EXPOSURE='70b85b6d786f019f09e4dd88292a9e617009a79b04b6346b086673b1fed034a5'

def components(pred,truth,spacing):
    delta=truth[:,1]-truth[:,0]
    length=np.linalg.norm(delta,axis=1)
    if (length<1e-6).any(): raise ValueError('Degenerate truth')
    unit=delta/length[:,None]
    normal=np.stack([-unit[:,1],unit[:,0]],axis=1)
    error=(pred-truth)*spacing[:,None,None]
    parallel=(error*unit[:,None,:]).sum(-1)
    perpendicular=(error*normal[:,None,:]).sum(-1)
    ref_oracle=pred.copy(); ref_oracle[:,:2]=truth[:,:2]
    outer_oracle=pred.copy(); outer_oracle[:,2]=truth[:,2]
    target=mls(truth,spacing)
    return {
        'full_mls_abs_error_mm':abs(mls(pred,spacing)-target),
        'truth_reference_only_mls_abs_error_mm':abs(mls(ref_oracle,spacing)-target),
        'truth_outer_only_mls_abs_error_mm':abs(mls(outer_oracle,spacing)-target),
        **{f'point{i+1}_abs_parallel_error_mm':abs(parallel[:,i]) for i in range(3)},
        **{f'point{i+1}_abs_perpendicular_error_mm':abs(perpendicular[:,i]) for i in range(3)},
        'abs_truth_angle_deg':abs((np.degrees(np.arctan2(delta[:,0],delta[:,1]))+90)%180-90),
    }

def main():
    if OUTPUT.exists(): raise FileExistsError('No overwrite')
    pins={
        'scripts/probe_mls_a7_translation_cuda.py':'2c245fd2e50ce33d3fc67f4a632c9ce9125ae6b4f99f551da51a723aa7b0b110',
        'scripts/probe_mls_translation_cuda.py':'02171bdfe04a5e64740d4349749739151cff03399844671c09bba5acdf6cac8f',
        'src/strategies/mls_heatmap/dataset.py':'df0852f12eba9329e3a65787d65e9649922d6b8c2704b2d7d47192ec33c9ac1c',
        'src/strategies/mls_heatmap/utils.py':'96377673b4cbb37da59a32cf67bc152baa4cfff875373ff4d87fdeb4de77239a',
        'Data/processed/mls_multitask_v2/mls_labels_multitask.csv':EXPECTED_LABELS,
        'config/folds.csv':EXPECTED_FOLDS,
        'Data/raw/training_df.pkl':'0e00255ce7dcd6963a00db6c4d5a5dfdc5cff5a6fa8aec6ead5cc5da185cfe7d'}
    for rel,expected in pins.items():
        if sha256(ROOT/rel)!=expected: raise ValueError('Input/source pin changed')
    # Self-test: parallel outer error leaves MLS unchanged; perpendicular error does not.
    t=np.array([[[0.,0.],[0.,10.],[3.,5.]]]); p=t.copy(); p[:,2,1]+=2
    c=components(p,t,np.ones(1))
    if not np.isclose(c['full_mls_abs_error_mm'][0],0): raise ValueError('Parallel test failed')
    p=t.copy(); p[:,2,0]+=2; c=components(p,t,np.ones(1))
    if not np.isclose(c['full_mls_abs_error_mm'][0],2) or not np.isclose(c['truth_outer_only_mls_abs_error_mm'][0],0):
        raise ValueError('Perpendicular test failed')
    lock=CAMPAIGN/'gpu_training.lock'; lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('Concurrent CUDA workload')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        configure_inference_precision()
        labels=ROOT/'Data/processed/mls_multitask_v2/mls_labels_multitask.csv'
        train,val=create_mls_dataloaders(str(labels),str(labels.parent/'images'),img_size=512,
            heatmap_size=128,heatmap_sigma=3.,batch_size=8,augment=False,num_workers=0,fold=0,
            seed=42,use_competition_folds=True,include_negatives=True,return_selector=True)
        indices,sample_sha=select_rows(train.dataset.data,set(val.dataset.data.patient_id.astype(str)),512)
        if sample_sha!=SAMPLE_SHA or len(indices)!=128: raise ValueError('Sample changed')
        loader=DataLoader(Subset(train.dataset,indices),batch_size=8,shuffle=False,num_workers=2)
        results={}
        for arm in ('baseline','consistency'):
            checkpoint,expected=CHECKPOINTS[arm]
            if sha256(checkpoint)!=expected: raise ValueError('Checkpoint changed')
            model,config=load_multitask_model(checkpoint,torch.device('cuda:0')); model.eval()
            before=state_digest(model); exposure=hashlib.sha256(); rows={}
            with torch.inference_mode():
                for images,_,mask,truth,spacing,target,*_ in loader:
                    if not (mask>.5).all() or not (target>.5).all(): raise ValueError('Invalid positive')
                    exposure.update(images.numpy().tobytes()); exposure.update(truth.numpy().tobytes())
                    images=images.cuda()
                    if images.device.type!='cuda': raise RuntimeError('GPU only')
                    logits,selector=model.forward_multitask(images)
                    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite logits')
                    prob=logits.flatten(2).softmax(-1).reshape_as(logits)
                    pred,_=decode_heatmap_dark_batch(prob.cpu(),128,512)
                    if not np.isfinite(pred).all() or (pred<0).any(): raise ValueError('Invalid decoder')
                    for k,v in components(pred,truth.numpy(),spacing.numpy()).items():
                        rows.setdefault(k,[]).extend(v.tolist())
            if exposure.hexdigest()!=EXPOSURE or state_digest(model)!=before:
                raise ValueError('Input or model-state mismatch')
            arrays={k:np.asarray(v) for k,v in rows.items()}
            angle=arrays.pop('abs_truth_angle_deg')
            strata={}
            for name,mask in {'all':np.ones(len(angle),bool),'angle_le_10':angle<=10,'angle_gt_10':angle>10}.items():
                strata[name]={'count':int(mask.sum()),'metrics':{k:describe(v[mask]) for k,v in arrays.items()} if mask.any() else {}}
            results[arm]={'checkpoint_sha256':expected,'model_state_unchanged':True,'strata':strata}
            del model; torch.cuda.empty_cache()
        result={'status':'completed','sample_sha256':sample_sha,'input_exposure_sha256':EXPOSURE,
            'source_sha256':sha256(Path(__file__)),'source_and_data_pins':pins,'precision':precision_flags(),
            'results':results,'model_updates':0,'validation_images_used':0,'promotion_eligible':False,
            'limitations':['Training sample only, not generalization or final triage evidence.',
                'Oracle substitutions are diagnostic only and cannot be deployed.',
                'Errors may cancel; oracle differences are not additive causal attributions.']}
        _atomic_json(OUTPUT,result)
        print(json.dumps(result))
    finally: lock.rmdir()

if __name__=='__main__': main()
