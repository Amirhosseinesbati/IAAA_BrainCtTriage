import unittest
from scripts.diagnose_mls_a6_factor_swap import mix
from src.strategies.mls_heatmap.predict_multitask import SliceMLSPrediction


class FactorSwapTests(unittest.TestCase):
    def test_both_probabilities_follow_selector_geometry_is_preserved(self):
        g={'s':(1.,[SliceMLSPrediction(0,.1,12.,.8,.2)])}
        s={'s':(1.,[SliceMLSPrediction(0,.9,2.,.3,.7)])}
        item=mix(g,s)['s'][1][0]
        self.assertEqual(item,SliceMLSPrediction(0,.9,12.,.8,.7))
        self.assertEqual(mix(g,g),g)
        self.assertEqual(g['s'][1][0].selector_probability,.1)

    def test_missing_peak_probability_is_preserved(self):
        g={'s':(1.,[SliceMLSPrediction(0,.1,12.,.8,.2)])}
        s={'s':(1.,[SliceMLSPrediction(0,.9,2.,.3)])}
        self.assertIsNone(mix(g,s)['s'][1][0].peak_probability)

    def test_rejects_unmatched_slice_and_truth(self):
        g={'s':(1.,[SliceMLSPrediction(0,.1,12.,.8)])}
        for s in [{'s':(2.,g['s'][1])},{'s':(1.,[])},{}]:
            with self.assertRaises(ValueError):mix(g,s)


if __name__=='__main__':unittest.main()
