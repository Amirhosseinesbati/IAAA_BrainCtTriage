import copy
import unittest
from scripts.evaluate_mls_a8_pair import verify_configs,verify_training,compare_audits,PROTOCOL_SHA,EVALUATOR_SHA,REFERENCE_SHA
from scripts.reconstruct_mls_aligned_cached_screen import gate


def fixtures():
    metrics={'mae_mm':1.,'boundary_f1':.8,'f1_3mm':.8,'f1_5mm':.8,'selection_objective':1.4}
    bounds={'mae_mm_lte':1.5,'boundary_f1_gte':.75,'f1_3mm_gte':.75,'f1_5mm_gte':.75,'selection_objective_lte':1.8}
    ref={'prospective_gate_bounds':bounds,'inference_signature':{'fixed':True},'hardware_signature':'gpu','runtime_baseline_metrics':metrics}
    audits={}
    for arm,mae in [('control',1.2),('refinement',1.)]:
        obs={**metrics,'mae_mm':mae,'selection_objective':mae+.4}
        audits[arm]={'status':'completed','fold':0,'seed':42,'fixed_epoch':15,'studies':70,
            'reference_refinement_enabled':arm=='refinement','checkpoint_sha256':arm,'baseline_self_test':False,'runtime_reference_sha256':REFERENCE_SHA,
            'source_sha256':EVALUATOR_SHA,'inference_signature':{'fixed':True},'hardware_signature':'gpu',
            'effective_gate_bounds':bounds,'runtime_baseline_metrics':metrics,'observed':obs,
            'gate_results':gate(obs,bounds,1e-8),'resource_gates_passed':True,
            'automatic_replication_allowed':False,'promotion_eligible':False,'submission_zip_allowed':False}
    return audits,ref,{'control':'control','refinement':'refinement'}


class PairAuditTest(unittest.TestCase):
    def test_only_declared_variant_may_differ(self):
        a={'use_reference_refinement':False,'batch_size':5}
        b={'use_reference_refinement':True,'batch_size':5}
        self.assertTrue(verify_configs([a,b]))
        for change in [{'use_reference_refinement':False},{'batch_size':10}]:
            with self.assertRaises(ValueError):verify_configs([a,{**b,**change}])

    def test_wrong_audited_variant_rejected(self):
        audits,ref,sha=fixtures()
        audits['refinement']['reference_refinement_enabled']=False
        with self.assertRaises(ValueError):compare_audits(audits,ref,sha)

    def test_gate_fail_not_replicated(self):
        audits,ref,sha=fixtures()
        audits['refinement']['observed']['f1_3mm']=.5
        audits['refinement']['gate_results']=gate(audits['refinement']['observed'],ref['prospective_gate_bounds'],1e-8)
        audits['refinement']['resource_gates_passed']=False
        self.assertFalse(compare_audits(audits,ref,sha)['refinement_replication_review_eligible'])

    def test_matched_history_and_mismatch(self):
        history=[{'epoch':e,'optimizer_steps':541,'input_exposure_sha256':str(e).zfill(64)} for e in range(1,16)]
        histories={a:copy.deepcopy(history) for a in ['control','refinement']}
        summaries={a:{'status':'completed','arm':a,'epochs_completed':15,'optimizer_steps':8115,
            'manifest_sha256':PROTOCOL_SHA,'initialization_sha256':'initial',
            'exposure_sha256_by_epoch':[r['input_exposure_sha256'] for r in history]} for a in histories}
        self.assertTrue(verify_training(summaries,histories,'initial'))
        histories['refinement'][0]['input_exposure_sha256']='x'*64
        summaries['refinement']['exposure_sha256_by_epoch'][0]='x'*64
        with self.assertRaises(ValueError):verify_training(summaries,histories,'initial')

    def test_pass_is_not_release(self):
        result=compare_audits(*fixtures())
        self.assertTrue(result['refinement_replication_review_eligible'])
        self.assertFalse(result['promotion_eligible'])
        self.assertFalse(result['automatic_replication_allowed'])

    def test_identical_model_not_refinement_benefit(self):
        audits,ref,sha=fixtures()
        audits['refinement']['observed']=copy.deepcopy(audits['control']['observed'])
        self.assertFalse(compare_audits(audits,ref,sha)['refinement_replication_review_eligible'])

    def test_false_pass_rejected(self):
        audits,ref,sha=fixtures()
        audits['refinement']['observed']['f1_3mm']=.5
        with self.assertRaises(ValueError):compare_audits(audits,ref,sha)

    def test_runtime_checkpoint_coverage_mismatch_rejected(self):
        for key,value in [('checkpoint_sha256','wrong'),('studies',69),('hardware_signature','other'),
                          ('runtime_reference_sha256','other'),('promotion_eligible',True)]:
            audits,ref,sha=fixtures();audits['refinement'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):compare_audits(audits,ref,sha)

    def test_objective_cannot_be_fabricated(self):
        audits,ref,sha=fixtures();audits['refinement']['observed']['selection_objective']=.1
        with self.assertRaises(ValueError):compare_audits(audits,ref,sha)

    def test_nonfinite_rejected(self):
        audits,ref,sha=fixtures();audits['refinement']['observed']['mae_mm']=float('nan')
        with self.assertRaises(ValueError):compare_audits(audits,ref,sha)


if __name__=='__main__':unittest.main()


