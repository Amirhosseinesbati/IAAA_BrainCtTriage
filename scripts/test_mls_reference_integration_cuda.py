"""CUDA integration: historical parity, checkpoint roundtrip and actual loss."""
from pathlib import Path
import sys
import json
import hashlib
import importlib.util
import subprocess
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.predict_multitask import load_multitask_model
from src.strategies.mls_heatmap.predict import _load_heatmap_model
from src.strategies.mls_heatmap.train_multitask import multitask_loss
from src.strategies.mls_heatmap.utils import generate_gaussian_heatmap
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision

BASE=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
BACKUP=BASE/'pre_reference_integration_source_20260904'
OUT=BASE/'reference_integration_cuda_20260904.json'
CKPT=ROOT/'models/checkpoints/mls_multitask/mls-vast-exp16-w32-fold0-strict-ensemble-refresh/mls_multitask_epoch_015.pth'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists(): raise FileExistsError('Existing result')
    if sha(CKPT)!='c242732048179eb8c7765fc9554dd3aa89d3392626e6a16c995a50615f14a062':
        raise ValueError('Historical checkpoint changed')
    oldpath=BACKUP/'src/strategies/mls_heatmap/model.py'
    if sha(oldpath)!='51d6b53572fd0ad720290be802c1cd3a0b4714433c680bf22c8298e501822e0e':
        raise ValueError('Historical model source changed')
    lock=BASE/'gpu_training.lock';lock.mkdir()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('Concurrent GPU workload')
        configure_inference_precision();torch.manual_seed(42)
        passed=[]
        def check(name,condition):
            if not condition: raise AssertionError(name)
            passed.append(name)
        model,config=load_multitask_model(CKPT,torch.device('cuda'))
        check('legacy_config_defaults_off',not config.use_reference_refinement and model.outer_refinement is None)
        spec=importlib.util.spec_from_file_location('legacy_mls_model',oldpath)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        old=legacy.HRNetHeatmapModel(backbone_name=config.backbone,in_channels=config.input_channels,
            pretrained=False,use_selector=True,head_dropout=config.head_dropout,selector_head_mode=config.selector_head_mode,
            use_ordinal_aux_head=config.use_ordinal_aux_head).cuda().eval()
        old.load_state_dict(model.state_dict(),strict=True)
        x=torch.randn(2,3,512,512,device='cuda')
        with torch.no_grad():
            a=old.forward_multitask(x);b=model.forward_multitask(x)
        check('legacy_source_exact_cuda_outputs',all(torch.equal(u,v) for u,v in zip(a,b)))
        del old,a,b,model
        config=MLSHeatmapConfig.model_validate({**config.model_dump(),'use_reference_refinement':True})
        model=HRNetHeatmapModel(backbone_name=config.backbone,pretrained=False,use_selector=True,
            head_dropout=config.head_dropout,selector_head_mode=config.selector_head_mode,use_reference_refinement=True).cuda()
        model.eval()
        with torch.no_grad():
            h,s=model.forward_multitask(x);he,se,o=model.forward_multitask_extended(x);hf=model(x)
        check('all_forward_contracts_match',torch.equal(h,he) and torch.equal(s,se) and torch.equal(h,hf) and o is None)
        del h,s,he,se,hf
        model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        xy=torch.tensor([[[250.,50.],[250.,450.],[260.,220.]]]*2,device='cuda')
        hm,mask=generate_gaussian_heatmap([(250.,50.),(250.,450.),(260.,220.)],512,128,3.,device=torch.device('cuda'))
        torch.cuda.reset_peak_memory_stats()
        h,s=model.forward_multitask(x)
        loss,_=multitask_loss(h,s,hm[None].expand(2,-1,-1,-1),mask[None].expand(2,-1),xy,
            torch.full((2,),.5,device='cuda'),torch.ones(2,device='cuda'),torch.full((2,),5.,device='cuda'),config)
        loss.backward()
        check('actual_multitask_loss_finite_gradients',torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        before=model.outer_refinement.refine[-1].weight.detach().clone();optimizer.step()
        check('actual_loss_updates_refiner',not torch.equal(before,model.outer_refinement.refine[-1].weight))
        peak=torch.cuda.max_memory_allocated()/1024**3
        model.eval()
        with torch.no_grad(): expected=model.forward_multitask(x)
        with tempfile.TemporaryDirectory(prefix='reference_checkpoint_test_',dir=BASE) as directory:
            path=Path(directory)/'synthetic.pth'
            torch.save({'config':config.model_dump(),'model_state_dict':model.state_dict()},path)
            restored,rc=load_multitask_model(path,torch.device('cuda'))
            with torch.no_grad(): actual=restored.forward_multitask(x)
            check('refined_checkpoint_exact_roundtrip',rc.use_reference_refinement and all(torch.equal(u,v) for u,v in zip(expected,actual)))
            try: _load_heatmap_model(str(path),config,torch.device('cuda'))
            except ValueError: passed.append('legacy_loader_rejects_refined_checkpoint')
            else: raise AssertionError('Legacy loader silently dropped refinement')
            del restored
            damaged={k:v for k,v in model.state_dict().items() if not k.startswith('outer_refinement.')}
            torch.save({'config':config.model_dump(),'model_state_dict':damaged},path)
            try: load_multitask_model(path,torch.device('cuda'))
            except RuntimeError: passed.append('missing_refiner_weights_rejected')
            else: raise AssertionError('Missing refiner weights accepted')
        result={'status':'completed','tests_passed':passed,'batch_size':2,'synthetic_optimizer_steps':1,
            'patient_images_used':0,'peak_allocated_gib_including_adam_state':peak,
            'historical_checkpoint_sha256':sha(CKPT),'promotion_eligible':False,
            'limitations':['Synthetic integration only; no efficacy claim.','Batch5 actual-loss memory and real-data fallback frequency remain untested.','This does not requalify historical source-pinned evaluators.']}
        with OUT.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result))
    finally:lock.rmdir()

if __name__=='__main__':main()
