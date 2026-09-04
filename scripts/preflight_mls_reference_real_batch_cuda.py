"""Training-only real batch5 feasibility with fixed untrained base initialization."""
from pathlib import Path
import sys
import json
import hashlib
import subprocess
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_mls_a7_paired_cuda import loaders,move_batch
from scripts.probe_mls_training_decoders_cuda import select_rows,sha256,EXPECTED_LABELS,EXPECTED_FOLDS
from scripts.probe_mls_translation_cuda import SAMPLE_SHA
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.reference_refinement import predicted_reference_fields
from src.strategies.mls_heatmap.train_multitask import multitask_loss,configure_training_determinism
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision

BASE=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
INIT=BASE/'a7_paired_translation_20260904/initialization.pth'
OUT=BASE/'reference_real_batch_preflight_20260904.json'

def main():
    if OUT.exists():raise FileExistsError('Existing evidence')
    for path,expected in [(INIT,'99996c103d672bbbdc3f589a38b6555e4f54217769c03fef5922ea6d7e15367a'),
        (ROOT/'Data/processed/mls_multitask_v2/mls_labels_multitask.csv',EXPECTED_LABELS),
        (ROOT/'config/folds.csv',EXPECTED_FOLDS),
        (ROOT/'Data/raw/training_df.pkl','0e00255ce7dcd6963a00db6c4d5a5dfdc5cff5a6fa8aec6ead5cc5da185cfe7d')]:
        if sha256(path)!=expected:raise ValueError('Pinned input mismatch')
    lock=BASE/'gpu_training.lock';lock.mkdir()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('Concurrent GPU job')
        configure_inference_precision();configure_training_determinism('strict');torch.manual_seed(42)
        initial=torch.load(INIT,map_location='cpu',weights_only=False)
        if initial['initialization']!='imagenet_backbone_random_heads_seed42_no_mls_training':
            raise ValueError('Not an untrained initialization')
        config=MLSHeatmapConfig.model_validate({**initial['config'],'use_reference_refinement':True})
        train=loaders(config,False);frame=train.dataset.data
        indices,sample_sha=select_rows(frame,set(),512)
        if sample_sha!=SAMPLE_SHA:raise ValueError('Training sample changed')
        model=HRNetHeatmapModel(backbone_name=config.backbone,pretrained=False,use_selector=True,
            head_dropout=config.head_dropout,use_reference_refinement=True).cuda()
        missing,unexpected=model.load_state_dict(initial['model_state_dict'],strict=False)
        expected_missing={k for k in model.state_dict() if k.startswith('outer_refinement.')}
        if set(missing)!=expected_missing or unexpected:raise ValueError('Base initialization mismatch')
        del initial
        model.eval();valid_count=0;seen=0
        with torch.no_grad():
            for batch in DataLoader(Subset(train.dataset,indices),batch_size=8,num_workers=2):
                image=batch[0].cuda()
                feature=model.backbone(image)[0];coarse=model.head(feature)
                fields,valid=predicted_reference_fields(coarse)
                if not torch.isfinite(coarse).all() or not torch.isfinite(fields).all():raise ValueError('Nonfinite conditioning')
                valid_count+=int(valid.sum());seen+=len(image)
        # Deterministic mixed batch by metadata only, not by model output.
        rows=list(frame.index[frame.is_target>.5][:3])+list(frame.index[frame.is_target<=.5][:2])
        batch=move_batch(next(iter(DataLoader(Subset(train.dataset,rows),batch_size=5))))
        image,target_heatmaps,masks,coords,spacing,is_target,study_mls,_=batch
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
        torch.cuda.reset_peak_memory_stats()
        active=[]
        def observe(module,inputs):active.append(int(predicted_reference_fields(inputs[1])[1].sum()))
        hook=model.outer_refinement.register_forward_pre_hook(observe)
        heatmaps,selector=model.forward_multitask(image);hook.remove()
        total,_=multitask_loss(heatmaps,selector,target_heatmaps,masks,coords,spacing,is_target,study_mls,config)
        total.backward()
        if not torch.isfinite(total) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite actual training loss/gradients')
        before=model.outer_refinement.refine[-1].weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        changed=not torch.equal(before,model.outer_refinement.refine[-1].weight)
        if not changed:raise ValueError('Refinement did not receive effective update')
        result={'status':'completed','sample_sha256':sample_sha,'fixed_training_positive_slices':seen,
            'initial_eval_valid_reference_count':valid_count,'initial_eval_fallback_count':seen-valid_count,
            'mixed_batch_size':5,'mixed_batch_positives':3,'mixed_batch_valid_reference_count':active[0],
            'real_training_batch_optimizer_steps':1,'finite_gradients':True,'refiner_updated':changed,
            'peak_allocated_gib_with_adam':torch.cuda.max_memory_allocated()/1024**3,
            'trained_checkpoint_saved':False,'validation_images_used':0,'promotion_eligible':False,
            'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
                ROOT/'src/strategies/mls_heatmap/reference_refinement.py',ROOT/'src/strategies/mls_heatmap/model.py',
                ROOT/'src/strategies/mls_heatmap/train_multitask.py',ROOT/'src/strategies/config_models.py']},
            'limitations':['One mixed unaugmented batch, not efficacy or full-training stability.',
                'Fallback counts describe untrained ImageNet/random-head initialization only.',
                'No new checkpoint retained; future training must reset to declared paired initialization.']}
        with OUT.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result))
    finally:lock.rmdir()

if __name__=='__main__':main()
