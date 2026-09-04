"""Fixed training-only three-checkpoint mechanism comparison; no model updates."""
from pathlib import Path
import sys
import json
import subprocess
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_mls_translation_cuda import (
    CAMPAIGN, SAMPLE_SHA, SHIFTS, translate, mls, describe, aligned_js,
    select_rows, sha256, create_mls_dataloaders, load_multitask_model,
    configure_inference_precision, precision_flags, decode_heatmap_dark_batch, _atomic_json,
    EXPECTED_CHECKPOINT, EXPECTED_LABELS, EXPECTED_FOLDS,
)
WORK=CAMPAIGN/'a7_paired_translation_20260904'
OUTPUT=WORK/'training_translation_comparison.json'
CHECKPOINTS={
    'baseline':(ROOT/'models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth',EXPECTED_CHECKPOINT),
    'control':(WORK/'control/mls_multitask_epoch_015.pth','a843b370f4e92798586a84f823f06f29aaf100df948b86bd5c5e24dba86b5820'),
    'consistency':(WORK/'consistency/mls_multitask_epoch_015.pth','4ed54985e07e0a7bf6a88f70b924dc144becedbe42cd85a90cd8f1f08826484b'),
}

def state_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    if OUTPUT.exists(): raise FileExistsError('Do not overwrite or rerun')
    for relative,expected in {
        'scripts/probe_mls_translation_cuda.py':'02171bdfe04a5e64740d4349749739151cff03399844671c09bba5acdf6cac8f',
        'scripts/probe_mls_training_decoders_cuda.py':'4080e19453cb5cc34a90d5a76616dda85e5b117337688c12048f2278dc439b97',
        'src/strategies/mls_heatmap/dataset.py':'df0852f12eba9329e3a65787d65e9649922d6b8c2704b2d7d47192ec33c9ac1c',
        'src/strategies/mls_heatmap/predict_multitask.py':'f56f6b98ea9c6320a0b29358db81e04f09942677dde51e4b464550601f18f11e',
        'src/strategies/mls_heatmap/utils.py':'96377673b4cbb37da59a32cf67bc152baa4cfff875373ff4d87fdeb4de77239a',
    }.items():
        if sha256(ROOT/relative)!=expected: raise ValueError('Source pin mismatch')
    labels=ROOT/'Data/processed/mls_multitask_v2/mls_labels_multitask.csv'
    for path,expected in [(labels,EXPECTED_LABELS),(ROOT/'config/folds.csv',EXPECTED_FOLDS),*CHECKPOINTS.values()]:
        if sha256(path)!=expected: raise ValueError('Input checksum mismatch')
    lock=CAMPAIGN/'gpu_training.lock'
    lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('Concurrent GPU workload')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        configure_inference_precision()
        train,val=create_mls_dataloaders(str(labels),str(labels.parent/'images'),img_size=512,
            heatmap_size=128,heatmap_sigma=3.,batch_size=8,augment=False,num_workers=0,fold=0,
            seed=42,use_competition_folds=True,include_negatives=True,return_selector=True)
        indices,sample_sha=select_rows(train.dataset.data,set(val.dataset.data.patient_id.astype(str)),512)
        if sample_sha!=SAMPLE_SHA or len(indices)!=128: raise ValueError('Frozen sample mismatch')
        loader=DataLoader(Subset(train.dataset,indices),batch_size=8,shuffle=False,num_workers=2)
        results={}
        exposure_reference=None
        for arm,(checkpoint,expected) in CHECKPOINTS.items():
            model,config=load_multitask_model(checkpoint,torch.device('cuda:0'))
            model.eval()
            before=state_digest(model)
            exposure=hashlib.sha256()
            rec={s:{k:[] for k in ('change','landmark_change','js','selector_change','original_error','translated_error','original_landmark_error','cross3','cross5')} for s in SHIFTS}
            def forward(images):
                if images.device.type!='cuda': raise RuntimeError('No CPU model inference')
                logits,selector=model.forward_multitask(images)
                if not torch.isfinite(logits).all() or not torch.isfinite(selector).all(): raise ValueError('Nonfinite output')
                p=logits.flatten(2).softmax(-1).reshape_as(logits)
                coords,_=decode_heatmap_dark_batch(p.cpu(),128,512)
                if not np.isfinite(coords).all() or (coords<0).any(): raise ValueError('Invalid decoder output')
                return p,coords,selector.sigmoid().cpu().numpy()
            with torch.inference_mode():
                for images,_,masks,truth,spacing,target,*_ in loader:
                    if not (masks>.5).all() or not (target>.5).all(): raise ValueError('Invalid sample')
                    exposure.update(images.numpy().tobytes()); exposure.update(truth.numpy().tobytes())
                    images=images.cuda(); truth=truth.numpy(); spacing=spacing.numpy()
                    p0,c0,s0=forward(images); m0=mls(c0,spacing); gt=mls(truth,spacing)
                    for dx,dy in SHIFTS:
                        p1,c1,s1=forward(translate(images,dx,dy)); c1=c1-np.array([dx,dy]); m1=mls(c1,spacing)
                        values={'change':np.abs(m1-m0),'landmark_change':np.linalg.norm(c1-c0,axis=2)*spacing[:,None],
                            'js':aligned_js(p0,p1,dx//4,dy//4).cpu().numpy(),'selector_change':np.abs(s1-s0),
                            'original_error':np.abs(m0-gt),'translated_error':np.abs(m1-gt),
                            'original_landmark_error':np.linalg.norm(c0-truth,axis=2)*spacing[:,None],
                            'cross3':(m0>=3)!=(m1>=3),'cross5':(m0>=5)!=(m1>=5)}
                        for k,v in values.items(): rec[(dx,dy)][k].extend(np.asarray(v).reshape(-1).tolist())
            if state_digest(model)!=before: raise ValueError('Model weights or buffers mutated')
            if exposure_reference is None: exposure_reference=exposure.hexdigest()
            if exposure.hexdigest()!=exposure_reference: raise ValueError('Model input exposure mismatch')
            results[arm]={'checkpoint_sha256':expected,'model_state_unchanged':True,'shifts':{
                str(s):{k:(int(sum(v)) if k.startswith('cross') else describe(v)) for k,v in r.items()} for s,r in rec.items()}}
            del model
            torch.cuda.empty_cache()
        result={'status':'completed','sample_count':128,'sample_sha256':sample_sha,'input_exposure_sha256':exposure_reference,
            'source_sha256':sha256(Path(__file__)),'precision_flags':precision_flags(),'torch_version':str(torch.__version__),
            'validation_images_used':0,'model_updates':0,'results':results,'promotion_eligible':False,'automatic_training_allowed':False}
        _atomic_json(OUTPUT,result)
        print(json.dumps(result))
    finally: lock.rmdir()

if __name__=='__main__': main()
