"""Synthetic CUDA-only prototype tests, no patient data or trained checkpoints."""
from pathlib import Path
import sys
import hashlib
import json
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.strategies.mls_heatmap.reference_refinement import (
    predicted_reference_fields,ReferenceConditionedOuterHead,ReferenceConditionedMLSPrototype)
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from scripts.evaluate_mls_canonical_resource_cuda import configure_inference_precision

BASE=Path('/workspace/iaaa_artifacts/mls_deploy_aligned_20260902')
OUT=BASE/'reference_refinement_preflight_20260904.json'

def main():
    if OUT.exists(): raise FileExistsError('Preflight already recorded')
    lock=BASE/'gpu_training.lock'; lock.mkdir()
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('Concurrent GPU job')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA only, no fallback')
        configure_inference_precision(); torch.manual_seed(42)
        tests=[]
        def check(name, condition):
            if not condition: raise AssertionError(name)
            tests.append(name)
        def peaks(a,b):
            z=torch.full((1,3,32,32),-10.,device='cuda')
            z[0,0,a[1],a[0]]=10; z[0,1,b[1],b[0]]=10
            return z
        z=peaks((8,4),(8,24)); fields,valid=predicted_reference_fields(z)
        scale=(32*32*2)**.5
        check('vertical_parallel_coordinate',torch.allclose(fields[0,0,14,11],torch.tensor(10/scale,device='cuda')))
        check('vertical_perpendicular_coordinate',torch.allclose(fields[0,1,14,11],torch.tensor(-3/scale,device='cuda')))
        shifted,_=predicted_reference_fields(peaks((10,7),(10,27)))
        check('translation_equivariant_coordinate_fields',torch.allclose(fields[:,:,:-3,:-2],shifted[:,:,3:,2:],atol=1e-7))
        # Rotate both endpoints and query 90 degrees on square heatmap.
        rotated,_=predicted_reference_fields(peaks((27,8),(7,8)))
        check('rotation_coordinate_roundtrip',torch.allclose(fields[0,:,14,11],rotated[0,:,11,17],atol=1e-7))
        flat=torch.zeros_like(z); f,v=predicted_reference_fields(flat)
        check('degenerate_reference_finite_zero',not v.any() and torch.isfinite(f).all() and (f==0).all())
        head=ReferenceConditionedOuterHead(8).cuda()
        feat=torch.randn(1,8,32,32,device='cuda',requires_grad=True)
        check('zero_initialization_exact_identity',torch.equal(head(feat,z),z))
        with torch.no_grad(): head.refine[-1].weight.fill_(.01); head.refine[-1].bias.fill_(.1)
        output=head(feat,z)
        check('reference_channels_unchanged',torch.equal(output[:,:2],z[:,:2]))
        check('nonzero_refinement_active',not torch.equal(output[:,2],z[:,2]))
        check('degenerate_reference_exact_fallback',torch.equal(head(feat,flat),flat))
        output[:,2].square().mean().backward()
        check('feature_gradient_finite_nonzero',feat.grad is not None and torch.isfinite(feat.grad).all() and feat.grad.abs().sum()>0)
        check('head_gradients_finite',all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        del head,feat,output
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base=HRNetHeatmapModel(backbone_name='hrnet_w32',pretrained=False,use_selector=True).cuda()
        model=ReferenceConditionedMLSPrototype(base).cuda().eval()
        x=torch.randn(5,3,512,512,device='cuda')
        with torch.no_grad():
            expected_h,expected_s=base.forward_multitask(x)
            actual_h,actual_s=model.forward_multitask(x)
        check('full_model_initial_heatmap_parity',torch.equal(expected_h,actual_h))
        check('full_model_initial_selector_parity',torch.equal(expected_s,actual_s))
        del expected_h,expected_s,actual_h,actual_s
        model.train()
        heatmap,selector=model.forward_multitask(x)
        target=torch.randn_like(heatmap)
        loss=(heatmap-target).square().mean()+selector.square().mean()
        loss.backward()
        check('batch5_full_forward_backward_finite',torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        check('full_backbone_received_gradient',any(p.grad is not None and p.grad.abs().sum()>0 for p in base.backbone.parameters()))
        check('final_refiner_received_gradient',model.outer_refinement.refine[-1].weight.grad is not None and model.outer_refinement.refine[-1].weight.grad.abs().sum()>0)
        result={'status':'completed','tests_passed':tests,'device':torch.cuda.get_device_name(),
            'batch_size':5,'image_size':512,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'extra_parameters':sum(p.numel() for p in model.outer_refinement.parameters()),
            'pretrained_weights_loaded':False,'patient_images_used':0,'optimizer_steps':0,
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
                'src/strategies/mls_heatmap/reference_refinement.py','scripts/preflight_mls_reference_refinement_cuda.py']},
            'promotion_eligible':False,'training_launch_authorized_by_result':False,
            'limitations':['Synthetic one-batch feasibility, not convergence, efficacy or full trainer compatibility.',
                'Zero final conv delays upstream refinement gradients until final conv moves.',
                'Discrete endpoint conditioning has no coordinate gradient; coarse head remains supervised.']}
        with OUT.open('x') as f: json.dump(result,f,indent=2)
        print(json.dumps(result))
    finally: lock.rmdir()

if __name__=='__main__':main()
