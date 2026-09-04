import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.reconstruct_mls_aligned_cached_screen import aligned, gate, predictions, read_cache
from src.strategies.mls_heatmap.predict_multitask import SliceMLSPrediction


class AlignedCachedScreenTests(unittest.TestCase):
    def test_profile_difference_is_not_hidden(self):
        cache={'s':(0.,[SliceMLSPrediction(i,p,v,1.) for i,(p,v) in enumerate([(.9,1.),(.4,2.),(.8,9.)])])}
        legacy=predictions(cache,['s'],{'selector_threshold':.5,'top_k':3,'aggregation':'p90'},None)
        actual=predictions(cache,['s'],{'selector_threshold':.6,'aggregation':'relative_component','min_active_slices':3,'probability_weighted':True},[0,30])
        self.assertAlmostEqual(float(legacy[0]),7.6)
        self.assertAlmostEqual(float(actual[0]),.1)

    def test_clipping_follows_three_seed_evaluator(self):
        c={'s':(0.,[SliceMLSPrediction(0,1.,40.,1.)])}
        self.assertEqual(float(predictions(c,['s'],{},[0,30])[0]),30.)

    def test_alignment_rejects_missing_study_truth_and_slice(self):
        s=SliceMLSPrediction(0,.9,1.,1.)
        reference={'s':(2.,[s])}
        for invalid in ({}, {'s':(3.,[s])}, {'s':(2.,[])}) :
            with self.assertRaises(ValueError): aligned(invalid,reference)

    def test_rounding_tolerance_and_real_failure(self):
        self.assertTrue(gate({'mae_mm':1.4709586392130174},{'mae_mm_lte':1.4709586392},1e-8)['mae_mm_lte'])
        self.assertFalse(gate({'mae_mm':1.48},{'mae_mm_lte':1.4709586392},1e-8)['mae_mm_lte'])
        self.assertFalse(gate({'f1_3mm':.7},{'f1_3mm_gte':.8},1e-8)['f1_3mm_gte'])

    def test_cache_rejects_duplicate_keys_nonfinite_and_bad_probability(self):
        item={'index':0,'selector_probability':.9,'mls_mm':1.,'heatmap_peak':1.}
        for items,repeat in [([item,item],False),([dict(item,mls_mm=float('nan'))],False),([dict(item,selector_probability=2.)],False),([item],True)]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'cache.csv'
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['study_id','gt_MLS_mm','error','slice_predictions_json'])
                    w.writeheader()
                    row={'study_id':'s','gt_MLS_mm':'1','error':'','slice_predictions_json':json.dumps(items)}
                    w.writerow(row)
                    if repeat:w.writerow(row)
                with self.assertRaises(ValueError):read_cache(p)


if __name__=='__main__':unittest.main()
