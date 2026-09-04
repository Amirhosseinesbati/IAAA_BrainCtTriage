import copy
import unittest

from scripts.qualify_mls_runtime_reference import BASELINE_SHA, EXPECTED_FLAGS, conservative_bounds, verify_pair


def fixture():
    a={'status':'failed_baseline_reproduction','baseline_self_test':True,'checkpoint_sha256':BASELINE_SHA,
       'fold':0,'seed':42,'fixed_epoch':15,'studies':1,'compute_policy':'cuda_only_no_cpu_model_fallback',
       'inference_signature':{'precision_flags':EXPECTED_FLAGS},'execution_id':'a','process_id':101,
       'source_sha256':'source','hardware_signature':'same-gpu','truth_sha256':'truth',
       'fold_manifest_sha256':'folds','reference_summary_sha256':'ref'}
    b=copy.deepcopy(a);b.update(execution_id='b',process_id=102)
    rows=[{'study_id':'one','MLS_mm':2.,'gt_MLS_mm':2.1,'input_fingerprint':{'raw':'x','sop':'y'},
           'slice_predictions':[{'index':0,'mls_mm':2.}]}]
    return a,b,rows,copy.deepcopy(rows),{'one':2.001},copy.deepcopy(rows)


class RuntimeReferenceTests(unittest.TestCase):
    def test_cross_runtime_failure_does_not_erase_exact_same_runtime_proof(self):
        args=fixture();self.assertTrue(verify_pair(*args))
        self.assertEqual(args[0]['status'],'failed_baseline_reproduction')

    def test_two_independent_executions_required(self):
        for key in ['execution_id','process_id']:
            args=fixture();args[1][key]=args[0][key]
            with self.subTest(key=key),self.assertRaises(ValueError):verify_pair(*args)

    def test_all_decoded_outputs_and_inputs_must_repeat_exactly(self):
        for mutate in [lambda r:r.update(MLS_mm=2.000000001),
                       lambda r:r['slice_predictions'][0].update(mls_mm=2.000000001),
                       lambda r:r['input_fingerprint'].update(raw='different')]:
            args=fixture();mutate(args[3][0])
            with self.assertRaises(ValueError):verify_pair(*args)

    def test_old_boundary_decisions_and_input_anchor_are_preserved(self):
        args=fixture();args[4]['one']=3.001
        with self.assertRaises(ValueError):verify_pair(*args)
        args=fixture();args[5][0]['input_fingerprint']['sop']='changed'
        with self.assertRaises(ValueError):verify_pair(*args)

    def test_config_hardware_checkpoint_coverage_are_bound(self):
        for key,value in [('hardware_signature','new-gpu'),('source_sha256','other'),('checkpoint_sha256','bad'),('studies',2)]:
            args=fixture();args[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_pair(*args)
        args=fixture();args[1]['inference_signature']['precision_flags']['cudnn_allow_tf32']=True
        with self.assertRaises(ValueError):verify_pair(*args)

    def test_bounds_never_relax_and_preserve_point01_objective_gain(self):
        old={'mae_mm_lte':1.5,'f1_3mm_gte':.8,'selection_objective_lte':1.9}
        current={'mae_mm':1.4,'f1_3mm':.79,'selection_objective':1.88}
        result=conservative_bounds(old,current)
        self.assertEqual(result['mae_mm_lte'],1.4);self.assertEqual(result['f1_3mm_gte'],.8)
        self.assertAlmostEqual(result['selection_objective_lte'],1.87)
        worse={'mae_mm':1.6,'f1_3mm':.82,'selection_objective':2.}
        result=conservative_bounds(old,worse)
        self.assertEqual(result,{'mae_mm_lte':1.5,'f1_3mm_gte':.82,'selection_objective_lte':1.9})

    def test_nonfinite_metric_and_prediction_refused(self):
        with self.assertRaises(ValueError):conservative_bounds({'mae_mm_lte':1.},{'mae_mm':float('nan')})
        args=fixture();args[4]['one']=float('inf')
        with self.assertRaises(ValueError):verify_pair(*args)


if __name__=='__main__':unittest.main()
