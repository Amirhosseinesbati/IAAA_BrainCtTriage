"""Pure contract tests; run on target server. No CPU model execution."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.evaluate_mls_canonical_resource_cuda import (
    POOL_FIELDS, PREPROCESS, aggregate, input_fingerprint,
    require_signature, signature, validate_private, migrate_known_baseline, LEGACY_BASELINE_SHA,
)
from scripts.evaluate_mls_three_seed_fold_cuda import _aggregate
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.predict_multitask import SliceMLSPrediction


POOL = {'selector_threshold': .6, 'top_k': 5, 'aggregation': 'relative_component',
        'relative_ratio': .3, 'aggregation_quantile': .75, 'probability_weighted': True,
        'anchor_window_radius': 3, 'min_active_slices': 3, 'heatmap_guard_ratio': 0.,
        'negative_value': .1}


def raw():
    return {**PREPROCESS, **{v: POOL[k] for k, v in POOL_FIELDS.items()}}


class CanonicalResourceTests(unittest.TestCase):
    def test_legacy_migration_is_bound_to_exact_baseline_hash_and_field(self):
        original = raw(); del original['selector_head_mode']
        repaired = migrate_known_baseline(original, LEGACY_BASELINE_SHA)
        signature(repaired, POOL, [0, 30])
        self.assertNotIn('selector_head_mode', original)
        with self.assertRaises(ValueError): signature(migrate_known_baseline(original, '0'*64), POOL, [0, 30])
        del original['top_k_slices']
        with self.assertRaises(ValueError): signature(migrate_known_baseline(original, LEGACY_BASELINE_SHA), POOL, [0, 30])
        changed = raw(); changed['selector_head_mode'] = 'dual'
        with self.assertRaises(ValueError): signature(migrate_known_baseline(changed, LEGACY_BASELINE_SHA), POOL, [0, 30])

    def test_every_saved_pooling_field_must_match(self):
        for key in POOL_FIELDS.values():
            with self.subTest(key=key):
                changed = raw()
                value = changed[key]
                changed[key] = not value if isinstance(value, bool) else ('p90' if isinstance(value, str) else value + 1)
                with self.assertRaises(ValueError): signature(changed, POOL, [0, 30])

    def test_every_inference_field_must_be_explicit(self):
        for key in raw():
            changed = raw(); del changed[key]
            with self.subTest(key=key), self.assertRaises(ValueError): signature(changed, POOL, [0, 30])

    def test_changed_preprocess_and_missing_or_changed_clamp_are_rejected(self):
        for key, value in [('image_size', 256), ('input_channels', 1), ('use_selector', False), ('selector_head_mode', 'dual')]:
            changed = raw(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): signature(changed, POOL, [0, 30])
        for clamp in [None, [0, 40], [-1, 30], [0]]:
            with self.subTest(clamp=clamp), self.assertRaises(ValueError): signature(raw(), POOL, clamp)

    def test_legacy_candidate_profile_cannot_be_accepted(self):
        changed = raw(); changed.update(selector_threshold=.5, top_k_slices=3, aggregation='p90')
        with self.assertRaises(ValueError): signature(changed, POOL, [0, 30])

    def test_no_change_is_equal_but_sources_clamp_and_runtime_are_bound(self):
        reference = signature(raw(), POOL, [0, 30])
        require_signature(reference, copy.deepcopy(reference))
        for key in ['source_sha256', 'runtime', 'clamp_mm', 'decoder']:
            changed = copy.deepcopy(reference); changed[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError): require_signature(changed, reference)

    def test_aggregation_matches_existing_three_seed_implementation(self):
        config = MLSHeatmapConfig.model_validate(raw())
        contract = signature(raw(), POOL, [0, 30])
        for values in [[(.9, 1.), (.4, 2.), (.8, 9.)], [(1., 40.)]*4, [(1., 2.)]*4]:
            slices = [SliceMLSPrediction(i, p, v, 1.) for i, (p, v) in enumerate(values)]
            self.assertEqual(aggregate(slices, contract), _aggregate(slices, config))
        self.assertEqual(aggregate([SliceMLSPrediction(i, 1., 40., 1.) for i in range(4)], contract), 30.)

    def test_rejects_nonfinite_missing_and_misordered_slices(self):
        contract = signature(raw(), POOL, [0, 30])
        for rows in [[], [SliceMLSPrediction(1, 1., 2., 1.)],
                     [SliceMLSPrediction(0, 1., float('nan'), 1.)],
                     [SliceMLSPrediction(0, 2., 2., 1.)]]:
            with self.assertRaises(ValueError): aggregate(rows, contract)

    def test_private_exact_coverage_and_finiteness(self):
        row = {'study_id': 'a', 'gt_MLS_mm': 1., 'MLS_mm': 1.}
        self.assertEqual(validate_private([row], ['a']), {'a': row})
        for rows in [[], [row, row], [dict(row, study_id='b')], [dict(row, MLS_mm=float('nan'))]]:
            with self.assertRaises(ValueError): validate_private(rows, ['a'])

    def test_raw_file_changes_and_uid_order_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.dcm'; p.write_bytes(b'one')
            q = Path(d) / 'b.dcm'; q.write_bytes(b'two')
            reader = SimpleNamespace(dicom_files=[str(p), str(q)], slices=[SimpleNamespace(SOPInstanceUID='1'), SimpleNamespace(SOPInstanceUID='2')])
            first = input_fingerprint(reader)
            reader.slices.reverse()
            second = input_fingerprint(reader)
            self.assertNotEqual(first['ordered_sop_uid_sha256'], second['ordered_sop_uid_sha256'])
            self.assertEqual(first['raw_files_sha256'], second['raw_files_sha256'])
            p.write_bytes(b'changed')
            self.assertNotEqual(second['raw_files_sha256'], input_fingerprint(reader)['raw_files_sha256'])
            reader.slices.append(reader.slices[0])
            with self.assertRaises(ValueError): input_fingerprint(reader)


if __name__ == '__main__': unittest.main()
