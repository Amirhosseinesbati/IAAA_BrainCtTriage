from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from scripts.compare_ich_2p5d_segmentation_oof import _metric_vector
from scripts.analyze_ich_slice_context import analyze_context_manifest, contiguous_runs
from scripts.crossfit_ich_temporal_volume_residual_head import (
    crossfit_volume_promotion_decision,
    inner_validation_fold,
    paired_patient_volume_bootstrap,
    parse_fold_checkpoints,
)
from scripts.diagnose_ich_multitask_gradient_conflict import (
    _suggested_auxiliary_weight_fields,
)
from scripts.evaluate_ich_2p5d_segmentation_checkpoint import checkpoint_config
from scripts.evaluate_ich_temporal_residual_outer import (
    OUTER_GATE,
    temporal_outer_decision,
)
from scripts.screen_ich_horizontal_flip_tta import tta_screen_decision
from scripts.screen_ich_sequence_pooling import (
    POOLERS,
    PRIMARY_METHOD,
    evaluation_summary,
    pool_slice_scores,
)
from scripts.screen_ich_sequence_meta_head import (
    GATE,
    crossfit_sequence_meta_head,
    feature_columns,
    promotion_decision,
    sequence_feature_frame,
)
from scripts.train_ich_temporal_residual_head import (
    GATE as TEMPORAL_GATE,
    temporal_promotion_decision,
)
from scripts.train_ich_temporal_volume_residual_head import (
    VOLUME_GATE,
    _checkpoint_score as temporal_volume_checkpoint_score,
    temporal_volume_promotion_decision,
)
from src.strategies.ich_2p5d.cache import OUTPUT_LABELS, resize_label_slice
from src.strategies.ich_2p5d.segmentation_data import (
    ICHAdjacentSegmentationDataset,
    ivh_center_target,
    oof_hard_negative_row_mask,
    segmentation_foreground_weights,
    split_segmentation_slices,
    subtype_aware_sampling_weights,
)
from src.strategies.ich_2p5d.segmentation_evaluation import (
    summarize_segmentation_predictions,
)
from src.strategies.ich_2p5d.segmentation_loss import (
    ICH25DSegmentationLoss,
    positive_sah_tversky_loss,
)
from src.strategies.ich_2p5d.segmentation_model import (
    FiveSliceContextInputAdapter,
    HorizontalSymmetryInputAdapter,
    SahBackgroundExpansionAdapter,
)
from src.strategies.ich_2p5d.temporal_head import (
    TemporalResidualHead,
    temporal_classification_loss,
)
from src.strategies.ich_2p5d.temporal_volume_head import (
    ICHSequenceVolumeDataset,
    SUBTYPE_LABELS,
    TRIAGE_VOLUME_THRESHOLDS_ML,
    TemporalVolumeResidualHead,
    forward_frozen_segmentation_components,
    temporal_volume_loss,
    volume_summary,
)
from src.strategies.ich_2p5d.segmentation_train import (
    ICH25DSegmentationTrainConfig,
    _flatten_summary_metrics,
    _predict_probabilities,
    _should_evaluate_outer,
    _should_stop_after_epoch,
    checkpoint_selection_score,
    configure_trainable_parameters,
    log_safe_mlflow_artifacts,
    load_initial_segmentation_checkpoint,
    set_segmentation_training_mode,
    validate_initial_checkpoint_provenance,
)


class ICH25DSegmentationTests(unittest.TestCase):
    @staticmethod
    def _tiny_smp_model() -> torch.nn.Module:
        class Encoder(torch.nn.Module):
            def forward(self, images):
                return [images, images[:, :1]]

        class Decoder(torch.nn.Module):
            def forward(self, features):
                return features[-1]

        class TinySmp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.decoder = Decoder()
                self.segmentation_head = torch.nn.Conv2d(1, 6, kernel_size=1)
                self.classification_head = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(1, len(OUTPUT_LABELS)),
                )

            def forward(self, images):
                features = self.encoder(images)
                decoded = self.decoder(features)
                return (
                    self.segmentation_head(decoded),
                    self.classification_head(features[-1]),
                )

        return TinySmp()

    def test_gradient_diagnostic_keeps_auxiliary_weight_suggestions_distinct(self):
        physical = _suggested_auxiliary_weight_fields(
            prefix="physical_volume",
            target_ratios=(0.10,),
            segmentation_norm=10.0,
            auxiliary_norm=2.0,
        )
        diffuse = _suggested_auxiliary_weight_fields(
            prefix="diffuse_tversky",
            target_ratios=(0.05,),
            segmentation_norm=10.0,
            auxiliary_norm=5.0,
        )

        self.assertEqual(physical["suggested_physical_volume_weight_10pct"], 0.5)
        self.assertEqual(diffuse["suggested_diffuse_tversky_weight_05pct"], 0.1)
        self.assertNotEqual(next(iter(physical)), next(iter(diffuse)))

    def test_mlflow_logging_excludes_row_level_medical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in (
                "best.pth",
                "resolved_config.json",
                "history.csv",
                "best_calibration_summary.json",
                "outer_summary.json",
                "run_summary.json",
                "best_calibration_slice_predictions.csv",
                "best_calibration_study_predictions.csv",
                "outer_slice_predictions.csv",
                "outer_study_predictions.csv",
            ):
                (output / name).write_text("test", encoding="utf-8")

            with mock.patch(
                "src.strategies.ich_2p5d.segmentation_train.mlflow.log_artifact"
            ) as log_artifact:
                logged = log_safe_mlflow_artifacts(output)

        called_names = {
            Path(call.args[0]).name for call in log_artifact.call_args_list
        }
        self.assertEqual(called_names, set(logged))
        self.assertNotIn("best_calibration_slice_predictions.csv", called_names)
        self.assertNotIn("best_calibration_study_predictions.csv", called_names)
        self.assertNotIn("outer_slice_predictions.csv", called_names)
        self.assertNotIn("outer_study_predictions.csv", called_names)

    def test_crossfit_checkpoint_mapping_requires_every_fold_once(self):
        mapping = parse_fold_checkpoints(
            tuple(f"{fold}=fold{fold}.pth" for fold in range(5))
        )
        self.assertEqual(set(mapping), set(range(5)))
        self.assertEqual(mapping[2], Path("fold2.pth"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_fold_checkpoints(
                ("0=a.pth", "0=b.pth", "1=c.pth", "2=d.pth", "3=e.pth")
            )
        with self.assertRaisesRegex(ValueError, "folds 0..4"):
            parse_fold_checkpoints(tuple(f"{fold}=x.pth" for fold in range(4)))

    def test_crossfit_inner_validation_policy_is_fixed_and_disjoint(self):
        expected = {0: 3, 1: 3, 2: 3, 3: 4, 4: 3}
        self.assertEqual(
            {fold: inner_validation_fold(fold) for fold in range(5)}, expected
        )
        self.assertTrue(
            all(fold != inner for fold, inner in expected.items())
        )

    def test_patient_bootstrap_detects_strict_volume_mae_improvement(self):
        truth = np.zeros((6, len(SUBTYPE_LABELS)), dtype=np.float64)
        truth[1:, 1] = np.arange(1.0, 6.0)
        baseline = truth.copy()
        baseline[1:, 1] += 2.0
        candidate = truth.copy()
        patients = np.asarray(["p0", "p1", "p2", "p3", "p4", "p5"])
        result = paired_patient_volume_bootstrap(
            truth,
            baseline,
            candidate,
            patients,
            samples=200,
            seed=42,
        )
        mae = result["metrics"]["total_volume_mae_ml"]
        self.assertEqual(mae["bootstrap_probability_candidate_better"], 1.0)
        self.assertLess(mae["delta_ci95"][1], 0.0)

    def test_crossfit_promotion_adds_bootstrap_to_volume_gates(self):
        delta = {
            "total_volume_mae_ml": -0.5,
            "absolute_total_volume_bias_ml": -0.5,
            "normal_false_positive_rate_at_0_1ml": 0.0,
            "presence_f1_at_0_1ml": 0.0,
            "presence_sensitivity_at_0_1ml": 0.0,
            "critical_trigger_macro_f1": 0.0,
            "subtype_mae_ml": {label: 0.0 for label in SUBTYPE_LABELS},
        }
        bootstrap = {
            "metrics": {
                "total_volume_mae_ml": {
                    "bootstrap_probability_candidate_better": 0.95,
                    "delta_ci95": [-1.0, 0.0],
                }
            }
        }
        self.assertTrue(
            crossfit_volume_promotion_decision(delta, bootstrap)[
                "promotion_allowed"
            ]
        )
        bootstrap["metrics"]["total_volume_mae_ml"]["delta_ci95"][1] = 0.001
        self.assertFalse(
            crossfit_volume_promotion_decision(delta, bootstrap)[
                "promotion_allowed"
            ]
        )

    def test_temporal_volume_dataset_preserves_one_patient_per_study(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "volume_cache.npz"
            np.savez_compressed(
                cache_path,
                embeddings=np.zeros((3, 4), dtype=np.float16),
                base_logits=np.zeros((3, len(OUTPUT_LABELS)), dtype=np.float32),
                base_slice_volumes=np.zeros(
                    (3, len(SUBTYPE_LABELS)), dtype=np.float32
                ),
                target_slice_volumes=np.zeros(
                    (3, len(SUBTYPE_LABELS)), dtype=np.float32
                ),
                spatial_known=np.ones(3, dtype=np.float32),
                study_id=np.asarray(["s1", "s1", "s2"]),
                patient_id=np.asarray(["p1", "p1", "p2"]),
                slice_index=np.asarray([0, 1, 0], dtype=np.int32),
            )
            truth = pd.DataFrame(
                {
                    "study_id": ["s1", "s2"],
                    **{
                        f"gt_{key}": [0.0, 0.0]
                        for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH")
                    },
                }
            )
            dataset = ICHSequenceVolumeDataset(cache_path, truth)
            self.assertEqual(dataset.study_ids, ["s1", "s2"])
            self.assertEqual(dataset.patient_ids, ["p1", "p2"])
            self.assertEqual(dataset[0]["patient_id"], "p1")

    def test_frozen_segmentation_forward_uses_list_decoder_contract(self):
        class Encoder(torch.nn.Module):
            def forward(self, images):
                return [images, images + 1.0]

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received_list = False

            def forward(self, features):
                self.received_list = isinstance(features, list)
                return features[-1]

        class ClassificationHead(torch.nn.Module):
            def forward(self, features):
                return features.mean(dim=(-2, -1))

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.decoder = Decoder()
                self.segmentation_head = torch.nn.Identity()
                self.classification_head = ClassificationHead()

        base = Base()
        features, masks, classes = forward_frozen_segmentation_components(
            base, torch.zeros((2, 6, 4, 4))
        )
        self.assertTrue(base.decoder.received_list)
        self.assertEqual(len(features), 2)
        self.assertEqual(masks.shape, (2, 6, 4, 4))
        self.assertEqual(classes.shape, (2, 6))

    def test_temporal_volume_head_is_exact_identity_at_initialization(self):
        model = TemporalVolumeResidualHead(
            12, projection_dim=8, hidden_dim=4, dropout=0.5
        ).train()
        features = torch.randn((3, 7, 12))
        base_logits = torch.randn((3, 7, len(OUTPUT_LABELS)))
        base_volumes = torch.rand((3, 7, len(SUBTYPE_LABELS)))
        base_volumes[0, 0] = 0.0
        lengths = torch.tensor([7, 5, 2])
        output = model(features, base_logits, base_volumes, lengths)
        torch.testing.assert_close(output, base_volumes, rtol=0.0, atol=0.0)

    def test_temporal_volume_head_can_recover_a_zero_base_volume(self):
        model = TemporalVolumeResidualHead(
            4, projection_dim=4, hidden_dim=2, dropout=0.0
        ).eval()
        with torch.no_grad():
            model.residual.bias.fill_(1.0)
        output = model(
            torch.zeros((1, 2, 4)),
            torch.zeros((1, 2, len(OUTPUT_LABELS))),
            torch.zeros((1, 2, len(SUBTYPE_LABELS))),
            torch.tensor([2]),
        )
        self.assertTrue(torch.all(output > 0))

    def test_temporal_volume_head_can_lock_zero_support(self):
        model = TemporalVolumeResidualHead(
            4,
            projection_dim=4,
            hidden_dim=2,
            dropout=0.0,
            preserve_zero_support=True,
        ).eval()
        with torch.no_grad():
            model.residual.bias.fill_(1.0)
        base_volumes = torch.zeros((1, 2, len(SUBTYPE_LABELS)))
        base_volumes[0, 1, 0] = 2.0
        output = model(
            torch.zeros((1, 2, 4)),
            torch.zeros((1, 2, len(OUTPUT_LABELS))),
            base_volumes,
            torch.tensor([2]),
        )
        torch.testing.assert_close(
            output[base_volumes == 0],
            torch.zeros_like(output[base_volumes == 0]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(float(output[0, 1, 0].detach()), 2.0)

    def test_temporal_volume_support_lock_is_exact_identity_at_initialization(self):
        model = TemporalVolumeResidualHead(
            12,
            projection_dim=8,
            hidden_dim=4,
            dropout=0.5,
            preserve_zero_support=True,
        ).train()
        features = torch.randn((3, 7, 12))
        base_logits = torch.randn((3, 7, len(OUTPUT_LABELS)))
        base_volumes = torch.rand((3, 7, len(SUBTYPE_LABELS)))
        base_volumes[0, 0] = 0.0
        output = model(
            features, base_logits, base_volumes, torch.tensor([7, 5, 2])
        )
        torch.testing.assert_close(output, base_volumes, rtol=0.0, atol=0.0)

    def test_temporal_volume_loss_is_finite_and_differentiable(self):
        candidate = torch.full(
            (2, 3, len(SUBTYPE_LABELS)), 0.25, requires_grad=True
        )
        target = torch.zeros_like(candidate)
        target[0, 0, 0] = 1.0
        target[1, 1, 1] = 2.0
        known = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.float32)
        study_target = target.detach().sum(dim=1)
        components = temporal_volume_loss(
            candidate,
            target,
            known,
            study_target,
            torch.tensor([1, 2]),
            slice_pos_weight=torch.ones(len(SUBTYPE_LABELS)),
            study_pos_weight=torch.ones(len(SUBTYPE_LABELS)),
        )
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        components["loss"].backward()
        self.assertIsNotNone(candidate.grad)
        self.assertTrue(torch.isfinite(candidate.grad).all())

    def test_temporal_volume_loss_rejects_non_sequence_shape(self):
        with self.assertRaisesRegex(ValueError, "Candidate slice volumes"):
            temporal_volume_loss(
                torch.zeros((2, len(SUBTYPE_LABELS))),
                torch.zeros((2, len(SUBTYPE_LABELS))),
                torch.ones(2),
                torch.zeros((2, len(SUBTYPE_LABELS))),
                torch.ones(2, dtype=torch.long),
                slice_pos_weight=torch.ones(len(SUBTYPE_LABELS)),
                study_pos_weight=torch.ones(len(SUBTYPE_LABELS)),
            )

    def test_temporal_volume_summary_covers_all_official_volume_boundaries(self):
        truth = np.zeros((7, len(SUBTYPE_LABELS)), dtype=np.float64)
        truth[1, SUBTYPE_LABELS.index("EDH")] = TRIAGE_VOLUME_THRESHOLDS_ML[
            "edh_critical"
        ]
        truth[2, SUBTYPE_LABELS.index("SDH")] = TRIAGE_VOLUME_THRESHOLDS_ML[
            "sdh_critical"
        ]
        truth[3, SUBTYPE_LABELS.index("IPH")] = TRIAGE_VOLUME_THRESHOLDS_ML[
            "iph_critical"
        ]
        truth[4, SUBTYPE_LABELS.index("SAH")] = TRIAGE_VOLUME_THRESHOLDS_ML[
            "total_fracture_combo"
        ]
        truth[5, SUBTYPE_LABELS.index("SAH")] = TRIAGE_VOLUME_THRESHOLDS_ML[
            "total_mls_combo"
        ]
        truth[6, SUBTYPE_LABELS.index("SAH")] = TRIAGE_VOLUME_THRESHOLDS_ML[
            "total_critical"
        ]
        summary = volume_summary(truth, truth.copy())
        self.assertEqual(summary["normal_false_positive_rate_at_0_1ml"], 0.0)
        self.assertEqual(summary["presence_f1_at_0_1ml"], 1.0)
        self.assertEqual(summary["critical_trigger_macro_f1"], 1.0)
        self.assertEqual(
            set(summary["triage_volume_trigger_f1"]),
            set(TRIAGE_VOLUME_THRESHOLDS_ML),
        )
        self.assertTrue(
            all(value == 1.0 for value in summary["triage_volume_trigger_f1"].values())
        )

    def test_temporal_volume_checkpoint_selection_enforces_fpr_and_f1_safety(self):
        baseline = {
            "total_volume_mae_ml": 10.0,
            "normal_false_positive_rate_at_0_1ml": 0.20,
            "presence_f1_at_0_1ml": 0.80,
        }
        safe = {
            **baseline,
            "total_volume_mae_ml": 9.0,
            "normal_false_positive_rate_at_0_1ml": 0.22,
            "presence_f1_at_0_1ml": 0.79,
        }
        self.assertEqual(temporal_volume_checkpoint_score(safe, baseline), -9.0)
        self.assertIsNone(
            temporal_volume_checkpoint_score(
                {**safe, "normal_false_positive_rate_at_0_1ml": 0.221}, baseline
            )
        )
        self.assertIsNone(
            temporal_volume_checkpoint_score(
                {**safe, "presence_f1_at_0_1ml": 0.789}, baseline
            )
        )

    def test_temporal_volume_promotion_uses_every_preregistered_gate(self):
        passing = {
            "total_volume_mae_ml": -VOLUME_GATE[
                "minimum_total_mae_improvement_ml"
            ],
            "absolute_total_volume_bias_ml": -VOLUME_GATE[
                "minimum_absolute_bias_improvement_ml"
            ],
            "normal_false_positive_rate_at_0_1ml": VOLUME_GATE[
                "maximum_fpr_delta"
            ],
            "presence_f1_at_0_1ml": VOLUME_GATE[
                "minimum_presence_f1_delta"
            ],
            "presence_sensitivity_at_0_1ml": 0.0,
            "critical_trigger_macro_f1": VOLUME_GATE[
                "minimum_critical_trigger_macro_f1_delta"
            ],
            "subtype_mae_ml": {
                label: VOLUME_GATE["maximum_subtype_mae_increase_ml"]
                for label in SUBTYPE_LABELS
            },
        }
        self.assertTrue(
            temporal_volume_promotion_decision(passing)["promotion_allowed"]
        )
        failing_variants = (
            {**passing, "total_volume_mae_ml": -0.499},
            {**passing, "absolute_total_volume_bias_ml": -0.499},
            {**passing, "normal_false_positive_rate_at_0_1ml": 0.0201},
            {**passing, "presence_f1_at_0_1ml": -0.0101},
            {**passing, "critical_trigger_macro_f1": -0.0201},
            {
                **passing,
                "subtype_mae_ml": {
                    **passing["subtype_mae_ml"],
                    SUBTYPE_LABELS[0]: 0.5001,
                },
            },
        )
        for variant in failing_variants:
            with self.subTest(variant=variant):
                self.assertFalse(
                    temporal_volume_promotion_decision(variant)[
                        "promotion_allowed"
                    ]
                )

    def test_temporal_outer_gate_requires_independent_replication(self):
        delta = {
            "selection_proxy": OUTER_GATE["minimum_selection_proxy_delta"],
            "macro_subtype_auc": OUTER_GATE[
                "minimum_macro_subtype_auc_delta"
            ],
            "any_ich_auc": OUTER_GATE["minimum_any_ich_auc_delta"],
            "subtype_auc": {
                label: OUTER_GATE["minimum_subtype_auc_delta"]
                for label in OUTPUT_LABELS[1:]
            },
        }
        self.assertTrue(temporal_outer_decision(delta)["expansion_allowed"])
        delta["selection_proxy"] -= 1e-5
        self.assertFalse(temporal_outer_decision(delta)["expansion_allowed"])

    def test_temporal_residual_head_is_exact_identity_at_initialization(self):
        model = TemporalResidualHead(
            12, projection_dim=8, hidden_dim=4, dropout=0.0
        ).eval()
        features = torch.randn((3, 7, 12))
        base_logits = torch.randn((3, 7, len(OUTPUT_LABELS)))
        lengths = torch.tensor([7, 5, 2])
        output = model(features, base_logits, lengths)
        torch.testing.assert_close(output, base_logits, rtol=0.0, atol=0.0)

    def test_temporal_loss_averages_slice_supervision_per_study(self):
        logits = torch.zeros((2, 3, len(OUTPUT_LABELS)), requires_grad=True)
        targets = torch.zeros_like(logits)
        known = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float32)
        study_targets = torch.zeros((2, len(OUTPUT_LABELS)))
        components = temporal_classification_loss(
            logits,
            targets,
            known,
            study_targets,
            torch.tensor([1, 3]),
            slice_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            study_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            study_loss_weight=0.5,
            focal_gamma=1.0,
        )
        self.assertTrue(torch.isfinite(components["loss"]))
        self.assertAlmostEqual(
            float(components["slice"].detach()),
            0.5 * np.log(2.0),
            places=6,
        )
        components["loss"].backward()
        self.assertIsNotNone(logits.grad)

    def test_temporal_promotion_uses_all_preregistered_safety_gates(self):
        delta = {
            "selection_proxy": TEMPORAL_GATE["minimum_selection_proxy_delta"],
            "macro_subtype_auc": TEMPORAL_GATE[
                "minimum_macro_subtype_auc_delta"
            ],
            "any_ich_auc": TEMPORAL_GATE["minimum_any_ich_auc_delta"],
            "subtype_auc": {
                label: TEMPORAL_GATE["minimum_subtype_auc_delta"]
                for label in OUTPUT_LABELS[1:]
            },
        }
        self.assertTrue(
            temporal_promotion_decision(delta)["promotion_allowed"]
        )
        delta["subtype_auc"]["SAH"] -= 1e-5
        self.assertFalse(
            temporal_promotion_decision(delta)["promotion_allowed"]
        )

    def test_classification_head_only_freezes_every_spatial_parameter(self):
        class TinySegmentationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 4, kernel_size=1),
                    torch.nn.BatchNorm2d(4),
                )
                self.decoder = torch.nn.Conv2d(4, 6, kernel_size=1)
                self.classification_head = torch.nn.Sequential(
                    torch.nn.Dropout(0.2), torch.nn.Linear(4, 6)
                )

        model = TinySegmentationModel()
        parameters = configure_trainable_parameters(
            model,
            freeze_base_model=False,
            classification_head_only=True,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in parameters),
            sum(
                parameter.numel()
                for parameter in model.classification_head.parameters()
            ),
        )
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.decoder.parameters()))
        self.assertTrue(
            all(p.requires_grad for p in model.classification_head.parameters())
        )
        set_segmentation_training_mode(
            model,
            freeze_base_model=False,
            classification_head_only=True,
        )
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.decoder.training)
        self.assertTrue(model.classification_head.training)

    def test_sequence_meta_features_are_fixed_threshold_free_statistics(self):
        rows = []
        for index, score in enumerate((0.1, 0.4, 0.9)):
            rows.append({
                "study_id": "s",
                "slice_index": index,
                "outer_fold": 2,
                **{f"prob_{label}": score for label in OUTPUT_LABELS},
            })
        features = sequence_feature_frame(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(len(feature_columns("any_ich")), 8)
        self.assertAlmostEqual(features["feature_any_ich_sequence_mean"], 1.4 / 3)
        self.assertAlmostEqual(
            features["feature_any_ich_sequence_std"],
            np.std(np.asarray([0.1, 0.4, 0.9]), ddof=0),
        )
        self.assertAlmostEqual(features["feature_any_ich_log_slice_count"], np.log1p(3))

    def test_sequence_meta_head_crossfits_every_fold_and_label(self):
        rows = []
        for fold in range(5):
            for offset, positive in enumerate((0, 0, 1, 1)):
                row = {
                    "study_id": f"f{fold}-{offset}",
                    "outer_fold": fold,
                    "truth_any_ich": positive,
                }
                for label in OUTPUT_LABELS:
                    score = 0.8 + 0.01 * fold if positive else 0.2 - 0.01 * fold
                    for column in feature_columns(label):
                        row[column] = score
                    if label != "any_ich":
                        row[f"truth_{label}"] = positive
                rows.append(row)
        scored, fold_models, final_models = crossfit_sequence_meta_head(
            pd.DataFrame(rows)
        )
        self.assertEqual(len(fold_models), 5)
        self.assertEqual(set(final_models), set(OUTPUT_LABELS))
        self.assertFalse(
            scored[[f"meta_score_{label}" for label in OUTPUT_LABELS]]
            .isna()
            .any()
            .any()
        )
        for payload in fold_models:
            self.assertNotIn(payload["heldout_fold"], payload["development_folds"])
            self.assertEqual(payload["development_studies"], 16)

    def test_sequence_meta_promotion_requires_every_preregistered_gate(self):
        delta = {
            "selection_proxy": GATE["minimum_selection_proxy_delta"],
            "macro_subtype_auc": GATE["minimum_macro_subtype_auc_delta"],
            "any_ich_auc": GATE["minimum_any_ich_auc_delta"],
            "subtype_auc": {
                label: GATE["minimum_subtype_auc_delta"]
                for label in OUTPUT_LABELS[1:]
            },
        }
        foldwise = {
            str(fold): {"delta": {"selection_proxy": 0.001 if fold < 3 else -0.001}}
            for fold in range(5)
        }
        bootstrap = {"deltas": {
            "selection_proxy": {"probability_positive": 0.90},
            "macro_subtype_auc": {"probability_positive": 0.90},
        }}
        self.assertTrue(
            promotion_decision(delta, foldwise, bootstrap)["promotion_allowed"]
        )
        delta["selection_proxy"] = 0.00199
        self.assertFalse(
            promotion_decision(delta, foldwise, bootstrap)["promotion_allowed"]
        )

    def test_sequence_pooling_rewards_adjacent_support_without_erasing_max(self):
        self.assertEqual(
            evaluation_summary({
                "selection_score": 0.1,
                "rescored_summary": {"selection_score": 0.2},
            })["selection_score"],
            0.2,
        )
        values = np.asarray([0.9, 0.1, 0.2], dtype=np.float64)
        self.assertAlmostEqual(POOLERS["max"](values), 0.9)
        self.assertAlmostEqual(POOLERS["top_two_mean"](values), 0.55)
        self.assertAlmostEqual(
            POOLERS["adjacent_pair_mean_max"](values), 0.5
        )
        self.assertAlmostEqual(
            POOLERS["adjacent_triple_mean_max"](values), 0.4
        )
        self.assertAlmostEqual(POOLERS[PRIMARY_METHOD](values), 0.7)

        rows = []
        for index, score in ((2, 0.2), (0, 0.9), (1, 0.1)):
            rows.append({
                "study_id": "s",
                "slice_index": index,
                "outer_fold": 3,
                **{f"prob_{label}": score for label in OUTPUT_LABELS},
            })
        pooled = pool_slice_scores(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(int(pooled["outer_fold"]), 3)
        self.assertAlmostEqual(
            pooled[f"score_any_ich_{PRIMARY_METHOD}"], 0.7
        )

    def test_context_audit_counts_contiguous_and_isolated_runs(self):
        np.testing.assert_array_equal(contiguous_runs(np.array([0, 1, 1, 0, 1])), [2, 1])
        rows = []
        for index, positive in enumerate((0, 1, 1, 0, 1)):
            rows.append({
                "study_id": "s",
                "slice_index": index,
                "slice_count": 5,
                "slice_spacing_mm": 5.0,
                "slice_thickness_mm": 5.0,
                "classification_known": 1,
                **{
                    label: int(label == "SAH" and positive)
                    for label in OUTPUT_LABELS[1:]
                },
            })
        result = analyze_context_manifest(pd.DataFrame(rows))
        sah = result["subtypes"]["SAH"]
        self.assertEqual(sah["positive_slices"], 3)
        self.assertEqual(sah["runs"], 2)
        self.assertAlmostEqual(sah["isolated_positive_slice_fraction"], 1 / 3)

    def test_zero_initialized_symmetry_adapter_is_exact_identity(self):
        class Echo(torch.nn.Module):
            def forward(self, images):
                return images, images.mean(dim=(-2, -1))

        model = HorizontalSymmetryInputAdapter(Echo(), input_channels=9)
        images = torch.randn((2, 9, 5, 7))
        masks, classes = model(images)
        torch.testing.assert_close(masks, images, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            classes, images.mean(dim=(-2, -1)), rtol=0.0, atol=0.0
        )

    def test_frozen_symmetry_adapter_exposes_only_162_parameters(self):
        base = torch.nn.Conv2d(9, 6, kernel_size=1)
        model = HorizontalSymmetryInputAdapter(base, input_channels=9)
        parameters = configure_trainable_parameters(
            model, freeze_base_model=True
        )
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 162)
        set_segmentation_training_mode(model, freeze_base_model=True)
        self.assertFalse(model.base_model.training)
        self.assertTrue(model.symmetry_residual.training)

    def test_zero_initialized_five_slice_adapter_selects_legacy_middle_context(self):
        class Echo(torch.nn.Module):
            def forward(self, images):
                return images, images.mean(dim=(-2, -1))

        model = FiveSliceContextInputAdapter(Echo())
        images = torch.randn((2, 15, 5, 7))
        masks, classes = model(images)
        expected = images[:, 3:12]
        torch.testing.assert_close(masks, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            classes, expected.mean(dim=(-2, -1)), rtol=0.0, atol=0.0
        )

    def test_frozen_five_slice_adapter_exposes_only_1215_parameters(self):
        model = FiveSliceContextInputAdapter(
            torch.nn.Conv2d(9, 6, kernel_size=1)
        )
        parameters = configure_trainable_parameters(
            model, freeze_base_model=True
        )
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 1215)
        set_segmentation_training_mode(model, freeze_base_model=True)
        self.assertFalse(model.base_model.training)
        self.assertTrue(model.context_residual.training)

    def test_zero_initialized_sah_residual_adapter_is_exact_identity(self):
        base = self._tiny_smp_model()
        model = SahBackgroundExpansionAdapter(base, hidden_channels=4)
        images = torch.randn((2, 9, 5, 7))
        expected_masks, expected_classes = base(images)
        masks, classes = model(images)
        torch.testing.assert_close(masks, expected_masks, rtol=0.0, atol=0.0)
        torch.testing.assert_close(classes, expected_classes, rtol=0.0, atol=0.0)

    def test_sah_residual_can_only_expand_incumbent_background(self):
        base = self._tiny_smp_model()
        with torch.no_grad():
            base.segmentation_head.weight.zero_()
            base.segmentation_head.bias.fill_(-2.0)
            base.segmentation_head.bias[0] = 2.0
            base.segmentation_head.weight[1, 0, 0, 0] = 5.0
            base.segmentation_head.weight[5, 0, 0, 0] = -5.0
        model = SahBackgroundExpansionAdapter(
            base,
            hidden_channels=4,
            maximum_logit_residual=8.0,
        )
        final = model.sah_residual_head[-1]
        self.assertIsInstance(final, torch.nn.Conv2d)
        with torch.no_grad():
            final.bias.fill_(1.0)
        images = torch.zeros((1, 9, 1, 3))
        images[:, 0, 0] = torch.tensor([0.0, 1.0, -1.0])
        baseline = base(images)[0].argmax(dim=1)
        candidate = model(images)[0].argmax(dim=1)
        self.assertEqual(baseline.tolist(), [[[0, 1, 5]]])
        self.assertEqual(candidate.tolist(), [[[5, 1, 5]]])

    def test_frozen_sah_residual_exposes_only_its_small_head(self):
        model = SahBackgroundExpansionAdapter(
            self._tiny_smp_model(), hidden_channels=4
        )
        parameters = configure_trainable_parameters(
            model, freeze_base_model=True
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in parameters),
            sum(parameter.numel() for parameter in model.sah_residual_head.parameters()),
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_model.parameters()))
        set_segmentation_training_mode(model, freeze_base_model=True)
        self.assertFalse(model.base_model.training)
        self.assertTrue(model.sah_residual_head.training)

    def test_legacy_checkpoint_can_zero_expand_into_symmetry_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            base = torch.nn.Conv2d(9, 6, kernel_size=1)
            checkpoint = Path(directory) / "legacy.pth"
            payload = {
                "state_dict": base.state_dict(),
                "config": {
                    "architecture": "unetplusplus",
                    "encoder_name": "efficientnet-b2",
                    "outer_fold": 2,
                    "calibration_fold": 1,
                },
                "output_labels": OUTPUT_LABELS,
                "segmentation_classes": 6,
                "input_channels": 9,
            }
            torch.save(payload, checkpoint)
            target = HorizontalSymmetryInputAdapter(
                torch.nn.Conv2d(9, 6, kernel_size=1), input_channels=9
            )
            config = ICH25DSegmentationTrainConfig(
                run_name="symmetry",
                output_dir="symmetry",
                outer_fold=2,
                calibration_fold=1,
                initial_checkpoint=str(checkpoint),
                horizontal_symmetry_adapter=True,
                freeze_base_model=True,
            )
            load_initial_segmentation_checkpoint(target, checkpoint, config)
            for expected, observed in zip(
                base.parameters(), target.base_model.parameters(), strict=True
            ):
                torch.testing.assert_close(expected, observed)
            self.assertEqual(
                float(target.symmetry_residual.weight.detach().abs().sum()), 0.0
            )

    def test_legacy_checkpoint_can_zero_expand_into_five_slice_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            base = torch.nn.Conv2d(9, 6, kernel_size=1)
            checkpoint = Path(directory) / "legacy.pth"
            payload = {
                "state_dict": base.state_dict(),
                "config": {
                    "architecture": "unetplusplus",
                    "encoder_name": "efficientnet-b2",
                    "outer_fold": 2,
                    "calibration_fold": 1,
                },
                "output_labels": OUTPUT_LABELS,
                "segmentation_classes": 6,
                "input_channels": 9,
            }
            torch.save(payload, checkpoint)
            target = FiveSliceContextInputAdapter(
                torch.nn.Conv2d(9, 6, kernel_size=1)
            )
            config = ICH25DSegmentationTrainConfig(
                run_name="five-slice",
                output_dir="five-slice",
                outer_fold=2,
                calibration_fold=1,
                initial_checkpoint=str(checkpoint),
                five_slice_context_adapter=True,
                slice_context_radius=2,
                freeze_base_model=True,
            )
            load_initial_segmentation_checkpoint(target, checkpoint, config)
            for expected, observed in zip(
                base.parameters(), target.base_model.parameters(), strict=True
            ):
                torch.testing.assert_close(expected, observed)
            self.assertEqual(
                float(target.context_residual.weight.detach().abs().sum()), 0.0
            )

    def test_legacy_checkpoint_can_zero_expand_into_sah_residual_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self._tiny_smp_model()
            checkpoint = Path(directory) / "legacy.pth"
            payload = {
                "state_dict": base.state_dict(),
                "config": {
                    "architecture": "unetplusplus",
                    "encoder_name": "efficientnet-b2",
                    "outer_fold": 2,
                    "calibration_fold": 1,
                },
                "output_labels": OUTPUT_LABELS,
                "segmentation_classes": 6,
                "input_channels": 9,
            }
            torch.save(payload, checkpoint)
            target = SahBackgroundExpansionAdapter(
                self._tiny_smp_model(), hidden_channels=4
            )
            config = ICH25DSegmentationTrainConfig(
                run_name="sah-residual",
                output_dir="sah-residual",
                outer_fold=2,
                calibration_fold=1,
                initial_checkpoint=str(checkpoint),
                sah_residual_adapter=True,
                sah_residual_hidden_channels=4,
                freeze_base_model=True,
            )
            load_initial_segmentation_checkpoint(target, checkpoint, config)
            for expected, observed in zip(
                base.parameters(), target.base_model.parameters(), strict=True
            ):
                torch.testing.assert_close(expected, observed)
            final = target.sah_residual_head[-1]
            self.assertEqual(float(final.weight.detach().abs().sum()), 0.0)
            self.assertEqual(float(final.bias.detach().abs().sum()), 0.0)

    def test_horizontal_flip_tta_restores_spatial_axis_and_averages_probabilities(self):
        class TwoViewModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images):
                self.calls += 1
                mask_logits = torch.zeros((1, 6, 2, 3))
                if self.calls == 1:
                    mask_logits[:, 1, :, 0] = 4.0
                    class_logits = torch.zeros((1, 6))
                else:
                    mask_logits[:, 1, :, -1] = 4.0
                    class_logits = torch.full((1, 6), 2.0)
                return mask_logits, class_logits

        model = TwoViewModel()
        masks, classes = _predict_probabilities(
            model, torch.zeros((1, 9, 2, 3)), horizontal_flip_tta=True
        )
        self.assertEqual(model.calls, 2)
        self.assertTrue(
            torch.equal(
                masks.argmax(dim=1),
                torch.tensor([[[1, 0, 0], [1, 0, 0]]]),
            )
        )
        expected = 0.5 * (
            torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))
        )
        torch.testing.assert_close(classes, torch.full((1, 6), expected))

    def test_tta_screen_requires_checkpoint_gain_and_primary_nonregression(self):
        baseline = {
            "selection_score": 0.66,
            "mean_foreground_dice": 0.45,
            "any_ich_study_auc": 0.92,
            "macro_subtype_study_auc": 0.89,
            "normal_false_positive_rate_at_0_1ml": 0.20,
            "presence_f1_at_0_1ml": 0.88,
            "total_volume_mae_ml": 10.0,
        }
        candidate = {
            **baseline,
            "selection_score": 0.665,
            "mean_foreground_dice": 0.455,
            "normal_false_positive_rate_at_0_1ml": 0.18,
            "total_volume_mae_ml": 9.9,
        }
        self.assertEqual(
            tta_screen_decision(baseline, candidate)["decision"],
            "advance_to_oof",
        )
        rejected = {**candidate, "normal_false_positive_rate_at_0_1ml": 0.22}
        self.assertEqual(
            tta_screen_decision(baseline, rejected)["decision"],
            "reject_before_outer",
        )

    def test_ivh_center_target_equalizes_component_center_area(self):
        mask = torch.zeros((15, 15), dtype=torch.long)
        mask[2, 2] = 1
        mask[9:12, 9:12] = 1
        centers = ivh_center_target(mask, square_size=3)
        self.assertEqual(float(centers.sum()), 18.0)
        self.assertEqual(float(centers[2, 2]), 1.0)
        self.assertEqual(float(centers[10, 10]), 1.0)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            ivh_center_target(mask, square_size=4)

    def test_smoke_run_never_evaluates_outer_fold(self):
        full = ICH25DSegmentationTrainConfig(run_name="full", output_dir="full")
        smoke = ICH25DSegmentationTrainConfig(
            run_name="smoke", output_dir="smoke", max_train_steps=2
        )
        calibration_screen = ICH25DSegmentationTrainConfig(
            run_name="screen",
            output_dir="screen",
            evaluate_outer=False,
        )
        self.assertTrue(_should_evaluate_outer(full))
        self.assertFalse(_should_evaluate_outer(smoke))
        self.assertFalse(_should_evaluate_outer(calibration_screen))
        self.assertFalse(_should_stop_after_epoch(full))
        self.assertTrue(_should_stop_after_epoch(smoke))
        self.assertFalse(_should_stop_after_epoch(calibration_screen))

    def test_fpr_penalized_checkpoint_score_uses_preregistered_tradeoff(self):
        summary = {
            "selection_score": 0.61,
            "normal_false_positive_rate_at_0_1ml": 0.36,
            "total_volume_mae_ml": 10.0,
            "total_volume_bias_ml": -4.0,
        }
        self.assertAlmostEqual(
            checkpoint_selection_score(summary, "fpr_penalized"), 0.574
        )
        self.assertAlmostEqual(
            checkpoint_selection_score(summary, "fpr_volume_penalized"),
            0.520,
        )
        self.assertEqual(checkpoint_selection_score(summary, "legacy"), 0.61)
        with self.assertRaisesRegex(ValueError, "checkpoint_selection_strategy"):
            checkpoint_selection_score(summary, "unknown")

    def test_warm_start_requires_same_model_and_held_out_folds(self):
        config = ICH25DSegmentationTrainConfig(
            run_name="warm", output_dir="warm", outer_fold=2, calibration_fold=1
        )
        payload = {
            "config": {
                "architecture": "unetplusplus",
                "encoder_name": "efficientnet-b2",
                "outer_fold": 2,
                "calibration_fold": 1,
            },
            "output_labels": OUTPUT_LABELS,
            "segmentation_classes": 6,
            "input_channels": 9,
        }
        validate_initial_checkpoint_provenance(payload, config)
        payload["config"]["outer_fold"] = 3
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_initial_checkpoint_provenance(payload, config)

    def test_foreground_weights_emphasize_rare_slice_labels(self):
        frame = pd.DataFrame({
            "IVH": [1] * 2 + [0] * 8,
            "IPH": [1] * 10,
            "SDH": [1] * 5 + [0] * 5,
            "EDH": [1] + [0] * 9,
            "SAH": [1] * 4 + [0] * 6,
        })
        weights = segmentation_foreground_weights(frame, power=1.0, maximum=8.0)
        np.testing.assert_allclose(weights.numpy(), [5.0, 1.0, 2.0, 8.0, 2.5])

    def test_foreground_weights_can_use_only_supervised_mask_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "labels.npy"
            labels = np.zeros((3, 4, 4), dtype=np.uint8)
            labels[0].reshape(-1)[:8] = 1
            labels[0].reshape(-1)[8:12] = 2
            labels[0].reshape(-1)[12:14] = 3
            labels[0].reshape(-1)[14] = 4
            labels[0].reshape(-1)[15] = 5
            labels[1] = 5
            labels[2] = 2
            np.save(label_path, labels)
            frame = pd.DataFrame({
                "label_cache_path": [str(label_path)] * 3,
                "slice_index": [0, 1, 2],
                "segmentation_known": [1, 1, 0],
            })
            weights = segmentation_foreground_weights(
                frame, power=1.0, maximum=8.0, basis="pixel"
            )
            np.testing.assert_allclose(
                weights.numpy(), [2.125, 4.25, 8.0, 8.0, 1.0]
            )

    def test_foreground_weight_basis_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "basis"):
            segmentation_foreground_weights(
                pd.DataFrame(), power=1.0, maximum=8.0, basis="voxel"
            )

    def test_study_balanced_sampler_preserves_positive_mass_and_equalizes_studies(self):
        rows = []
        for study_id, subtype, slices in (
            ("ivh-small", "IVH", 1),
            ("ivh-large", "IVH", 4),
            ("iph", "IPH", 1),
            ("sdh", "SDH", 1),
            ("edh", "EDH", 1),
            ("sah", "SAH", 1),
        ):
            for _ in range(slices):
                rows.append({
                    "study_id": study_id,
                    **{label: int(label == subtype) for label in OUTPUT_LABELS[1:]},
                })
        rows.append({
            "study_id": "normal",
            **{label: 0 for label in OUTPUT_LABELS[1:]},
        })
        frame = pd.DataFrame(rows)
        original = subtype_aware_sampling_weights(frame, study_balance_power=0.0)
        balanced = subtype_aware_sampling_weights(frame, study_balance_power=1.0)
        positive = frame[list(OUTPUT_LABELS[1:])].any(axis=1).to_numpy()

        self.assertAlmostEqual(
            float(original[positive].sum()), float(balanced[positive].sum())
        )
        small_mass = float(balanced[frame["study_id"] == "ivh-small"].sum())
        large_mass = float(balanced[frame["study_id"] == "ivh-large"].sum())
        self.assertAlmostEqual(small_mass, large_mass)
        self.assertLess(
            float(original[frame["study_id"] == "ivh-small"].sum()),
            float(original[frame["study_id"] == "ivh-large"].sum()),
        )
        self.assertEqual(float(balanced[~positive].item()), 1.0)

    def test_study_balanced_sampler_power_is_validated(self):
        frame = pd.DataFrame({
            "study_id": ["a"],
            **{label: [1] for label in OUTPUT_LABELS[1:]},
        })
        with self.assertRaisesRegex(ValueError, "study-balance power"):
            subtype_aware_sampling_weights(frame, study_balance_power=1.1)

    def test_oof_hard_negative_sampler_preserves_class_mass(self):
        rows = []
        for index, subtype in enumerate(OUTPUT_LABELS[1:]):
            rows.append({
                "study_id": f"positive-{subtype}",
                "patient_id": f"p-{subtype}",
                "slice_index": 0,
                "fold": index % 5,
                **{label: int(label == subtype) for label in OUTPUT_LABELS[1:]},
            })
        rows.extend([
            {
                "study_id": "hard",
                "patient_id": "p-hard",
                "slice_index": 3,
                "fold": 2,
                **{label: 0 for label in OUTPUT_LABELS[1:]},
            },
            {
                "study_id": "easy",
                "patient_id": "p-easy",
                "slice_index": 1,
                "fold": 3,
                **{label: 0 for label in OUTPUT_LABELS[1:]},
            },
        ])
        frame = pd.DataFrame(rows)
        hard = pd.DataFrame({
            "study_id": ["hard"],
            "patient_id": ["p-hard"],
            "slice_index": [3],
            "source_outer_fold": [2],
            "ground_truth_any_ich": [0],
            "predicted_foreground_pixels": [42],
        })
        original = subtype_aware_sampling_weights(frame)
        reweighted = subtype_aware_sampling_weights(
            frame,
            hard_negative_slices=hard,
            hard_negative_multiplier=3.0,
        )
        positive = frame[list(OUTPUT_LABELS[1:])].any(axis=1).to_numpy()
        negative = ~positive
        hard_row = frame["study_id"].eq("hard").to_numpy()
        easy_row = frame["study_id"].eq("easy").to_numpy()
        self.assertAlmostEqual(
            float(original[positive].sum()), float(reweighted[positive].sum())
        )
        self.assertAlmostEqual(
            float(original[negative].sum()), float(reweighted[negative].sum())
        )
        self.assertGreater(float(reweighted[hard_row].item()), 1.0)
        self.assertLess(float(reweighted[easy_row].item()), 1.0)
        self.assertAlmostEqual(
            float(reweighted[hard_row].item() / reweighted[easy_row].item()), 3.0
        )

    def test_oof_hard_negative_provenance_rejects_fold_mismatch(self):
        frame = pd.DataFrame({
            "study_id": ["hard"],
            "patient_id": ["p"],
            "slice_index": [1],
            "fold": [2],
            **{label: [0] for label in OUTPUT_LABELS[1:]},
        })
        hard = pd.DataFrame({
            "study_id": ["hard"],
            "patient_id": ["p"],
            "slice_index": [1],
            "source_outer_fold": [3],
            "ground_truth_any_ich": [0],
            "predicted_foreground_pixels": [1],
        })
        with self.assertRaisesRegex(ValueError, "source fold"):
            oof_hard_negative_row_mask(frame, hard)

    def test_label_resize_preserves_categorical_values(self):
        label = np.asarray([[0, 3], [5, 1]], dtype=np.uint8)
        resized = resize_label_slice(label, 8)
        self.assertEqual(resized.shape, (8, 8))
        self.assertEqual(set(np.unique(resized)), {0, 1, 3, 5})

    def test_split_keeps_classification_only_rows_but_evaluates_all_slices(self):
        rows = []
        for fold in range(5):
            for known in (0, 1):
                rows.append({
                    "study_id": f"{fold}-{known}",
                    "patient_id": f"p{fold}",
                    "fold": fold,
                    "slice_index": known,
                    "known": known,
                    "classification_known": 1,
                    "segmentation_known": known,
                    "metadata_missing": 0,
                })
        train, calibration, outer = split_segmentation_slices(
            pd.DataFrame(rows), outer_fold=0, calibration_fold=1
        )
        self.assertEqual(set(train["segmentation_known"]), {0, 1})
        self.assertTrue((train["classification_known"] == 1).all())
        self.assertEqual(set(calibration["known"]), {0, 1})
        self.assertEqual(set(outer["known"]), {0, 1})
        self.assertEqual(set(train["fold"]), {2, 3, 4})

    def test_dataset_returns_registered_center_mask_and_physical_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            label_path = root / "label.npy"
            image = np.zeros((3, 3, 8, 8), dtype=np.uint8)
            image[1] = 128
            label = np.zeros((3, 8, 8), dtype=np.uint8)
            label[1, 2:4, 3:6] = 3
            np.save(image_path, image)
            np.save(label_path, label)
            row = {
                "study_id": "a",
                "patient_id": "p",
                "slice_index": 1,
                "known": 1,
                "classification_known": 1,
                "segmentation_known": 1,
                "metadata_missing": 0,
                "cache_path": str(image_path),
                "label_cache_path": str(label_path),
                "resized_voxel_volume_ml": 0.002,
                **{name: int(name in {"any_ich", "SDH"}) for name in OUTPUT_LABELS},
            }
            item = ICHAdjacentSegmentationDataset(pd.DataFrame([row]))[0]
            self.assertEqual(tuple(item["image"].shape), (9, 8, 8))
            self.assertEqual(int((item["mask"] == 3).sum()), 6)
            self.assertAlmostEqual(float(item["voxel_volume_ml"]), 0.002)

    def test_dataset_five_slice_context_edge_padding_is_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.npy"
            label_path = root / "label.npy"
            image = np.zeros((3, 3, 4, 4), dtype=np.uint8)
            image[1] = 64
            image[2] = 128
            np.save(image_path, image)
            np.save(label_path, np.zeros((3, 4, 4), dtype=np.uint8))
            row = {
                "study_id": "edge",
                "patient_id": "p",
                "slice_index": 0,
                "classification_known": 1,
                "segmentation_known": 1,
                "cache_path": str(image_path),
                "label_cache_path": str(label_path),
                "resized_voxel_volume_ml": 0.002,
                **{name: 0 for name in OUTPUT_LABELS},
            }
            item = ICHAdjacentSegmentationDataset(
                pd.DataFrame([row]), context_radius=2
            )[0]
            observed = item["image"]
            self.assertEqual(tuple(observed.shape), (15, 4, 4))
            torch.testing.assert_close(observed[0:3], observed[3:6])
            torch.testing.assert_close(observed[3:6], observed[6:9])
            self.assertFalse(torch.equal(observed[6:9], observed[9:12]))
            self.assertFalse(torch.equal(observed[9:12], observed[12:15]))

    def test_multitask_loss_backpropagates(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS))
        )
        mask_logits = torch.zeros((2, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((2, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((2, 8, 8), dtype=torch.long)
        masks[0, 2:4, 2:4] = 2
        targets = torch.zeros_like(class_logits)
        targets[0, :3] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            torch.ones(2),
        )
        components["loss"].backward()
        self.assertGreater(float(mask_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(class_logits.grad.abs().sum()), 0.0)

    def test_known_empty_mask_gets_non_focal_foreground_penalty(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            empty_foreground_weight=0.05,
        )
        mask_logits = torch.zeros((2, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((2, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((2, 8, 8), dtype=torch.long)
        masks[1, 2:4, 2:4] = 2
        targets = torch.zeros_like(class_logits)
        targets[1, :3] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.ones(2),
        )
        self.assertAlmostEqual(
            float(components["empty_foreground"].detach()),
            float(np.log(6.0)),
            places=5,
        )
        components["loss"].backward()
        self.assertLess(float(mask_logits.grad[0, 0].mean()), 0.0)
        self.assertGreater(float(mask_logits.grad[0, 1:].mean()), 0.0)

    def test_ivh_center_loss_adds_equal_area_recall_gradient(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            ivh_center_loss_weight=0.1,
        )
        mask_logits = torch.zeros((1, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        masks[0, 3, 3] = 1
        centers = torch.zeros_like(masks, dtype=torch.float32)
        centers[0, 2:5, 2:5] = 1.0
        targets = torch.zeros_like(class_logits)
        targets[0, :2] = 1.0
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.ones(1),
            ivh_center_targets=centers,
        )
        self.assertAlmostEqual(
            float(components["ivh_center"].detach()),
            float(np.log(6.0)),
            places=5,
        )
        components["loss"].backward()
        self.assertLess(float(mask_logits.grad[0, 1, 3, 3]), 0.0)

    def test_ivh_center_loss_requires_targets_only_when_enabled(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            ivh_center_loss_weight=0.1,
        )
        with self.assertRaisesRegex(ValueError, "ivh_center_targets"):
            loss_fn.components(
                torch.zeros((1, 6, 4, 4)),
                torch.zeros((1, len(OUTPUT_LABELS))),
                torch.zeros((1, 4, 4), dtype=torch.long),
                torch.zeros((1, len(OUTPUT_LABELS))),
                segmentation_known=torch.ones(1),
            )

    def test_physical_volume_loss_backpropagates_toward_target_burden(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            physical_volume_loss_weight=0.05,
        )
        mask_logits = torch.full((1, 6, 4, 4), -4.0, requires_grad=True)
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        masks[0, 1:3, 1:3] = 2
        with torch.no_grad():
            mask_logits[:, 0] = 4.0
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            torch.zeros_like(class_logits),
            segmentation_known=torch.ones(1),
            voxel_volume_ml=torch.tensor([0.5]),
        )
        self.assertGreater(float(components["physical_volume"].detach()), 0.0)
        components["physical_volume"].backward()
        self.assertLess(float(mask_logits.grad[0, 2, 1:3, 1:3].mean()), 0.0)

    def test_physical_volume_loss_ignores_classification_only_rows(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            physical_volume_loss_weight=0.05,
        )
        mask_logits = torch.zeros((1, 6, 4, 4), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        components = loss_fn.components(
            mask_logits,
            class_logits,
            torch.zeros((1, 4, 4), dtype=torch.long),
            torch.zeros_like(class_logits),
            segmentation_known=torch.zeros(1),
            classification_known=torch.ones(1),
            voxel_volume_ml=torch.tensor([0.5]),
        )
        components["loss"].backward()
        self.assertEqual(float(components["physical_volume"].detach()), 0.0)
        self.assertTrue(mask_logits.grad is None or not mask_logits.grad.any())

    def test_physical_volume_loss_requires_valid_voxel_sizes_when_enabled(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            physical_volume_loss_weight=0.05,
        )
        arguments = (
            torch.zeros((1, 6, 4, 4)),
            torch.zeros((1, len(OUTPUT_LABELS))),
            torch.zeros((1, 4, 4), dtype=torch.long),
            torch.zeros((1, len(OUTPUT_LABELS))),
            torch.ones(1),
        )
        with self.assertRaisesRegex(ValueError, "voxel_volume_ml"):
            loss_fn.components(*arguments)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            loss_fn.components(*arguments, voxel_volume_ml=torch.tensor([0.0]))

    def test_physical_volume_loss_is_invariant_to_pixel_volume_representation(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            physical_volume_loss_weight=0.05,
        )
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)))
        target = torch.zeros_like(class_logits)

        logits_a = torch.full((1, 6, 1, 4), -8.0)
        logits_a[:, 0] = 8.0
        logits_a[:, 2, :, :2] = 16.0
        masks_a = torch.tensor([[[2, 2, 0, 0]]])
        loss_a = loss_fn.components(
            logits_a,
            class_logits,
            masks_a,
            target,
            torch.ones(1),
            voxel_volume_ml=torch.tensor([1.0]),
        )["physical_volume"]

        logits_b = torch.full((1, 6, 1, 8), -8.0)
        logits_b[:, 0] = 8.0
        logits_b[:, 2, :, :4] = 16.0
        masks_b = torch.tensor([[[2, 2, 2, 2, 0, 0, 0, 0]]])
        loss_b = loss_fn.components(
            logits_b,
            class_logits,
            masks_b,
            target,
            torch.ones(1),
            voxel_volume_ml=torch.tensor([0.5]),
        )["physical_volume"]
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_positive_diffuse_tversky_recovers_sdh_false_negative(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            diffuse_tversky_loss_weight=0.1,
        )
        mask_logits = torch.full((1, 6, 4, 4), -4.0, requires_grad=True)
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        masks[0, 1:3, 1:3] = 3
        with torch.no_grad():
            mask_logits[:, 0] = 4.0
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            torch.zeros_like(class_logits),
            segmentation_known=torch.ones(1),
        )
        self.assertGreater(float(components["diffuse_tversky"].detach()), 0.0)
        components["diffuse_tversky"].backward()
        self.assertLess(float(mask_logits.grad[0, 3, 1:3, 1:3].mean()), 0.0)

    def test_positive_sah_tversky_has_no_sdh_objective(self):
        mask_logits = torch.zeros((1, 6, 4, 4), requires_grad=True)
        sdh_masks = torch.zeros((1, 4, 4), dtype=torch.long)
        sdh_masks[0, 1:3, 1:3] = 3
        loss = positive_sah_tversky_loss(
            mask_logits,
            sdh_masks,
            torch.ones(1),
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(mask_logits.grad is None or not mask_logits.grad.any())

    def test_positive_sah_tversky_recovers_sah_false_negative(self):
        mask_logits = torch.full((1, 6, 4, 4), -4.0, requires_grad=True)
        sah_masks = torch.zeros((1, 4, 4), dtype=torch.long)
        sah_masks[0, 1:3, 1:3] = 5
        with torch.no_grad():
            mask_logits[:, 0] = 4.0
        loss = positive_sah_tversky_loss(
            mask_logits,
            sah_masks,
            torch.ones(1),
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertLess(float(mask_logits.grad[0, 5, 1:3, 1:3].mean()), 0.0)

    def test_positive_diffuse_tversky_excludes_empty_and_non_diffuse_rows(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            diffuse_tversky_loss_weight=0.1,
        )
        mask_logits = torch.zeros((2, 6, 4, 4), requires_grad=True)
        masks = torch.zeros((2, 4, 4), dtype=torch.long)
        masks[1, 1:3, 1:3] = 2
        class_logits = torch.zeros((2, len(OUTPUT_LABELS)), requires_grad=True)
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            torch.zeros_like(class_logits),
            segmentation_known=torch.ones(2),
        )
        components["loss"].backward()
        self.assertEqual(float(components["diffuse_tversky"].detach()), 0.0)

    def test_positive_diffuse_tversky_ignores_classification_only_rows(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            diffuse_tversky_loss_weight=0.1,
        )
        mask_logits = torch.zeros((1, 6, 4, 4), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        components = loss_fn.components(
            mask_logits,
            class_logits,
            torch.full((1, 4, 4), 3, dtype=torch.long),
            torch.zeros_like(class_logits),
            segmentation_known=torch.zeros(1),
            classification_known=torch.ones(1),
        )
        components["loss"].backward()
        self.assertEqual(float(components["diffuse_tversky"].detach()), 0.0)
        self.assertTrue(mask_logits.grad is None or not mask_logits.grad.any())

    def test_positive_only_batch_has_no_empty_foreground_penalty(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            empty_foreground_weight=0.05,
        )
        mask_logits = torch.zeros((1, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        masks[0, 2:4, 2:4] = 2
        targets = torch.zeros_like(class_logits)
        targets[0, :3] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.ones(1),
        )
        self.assertEqual(float(components["empty_foreground"].detach()), 0.0)

    def test_empty_foreground_top_fraction_focuses_sparse_hard_pixels(self):
        common = {
            "classification_pos_weight": torch.ones(len(OUTPUT_LABELS)),
            "empty_foreground_weight": 0.05,
        }
        average_loss = ICH25DSegmentationLoss(
            **common, empty_foreground_top_fraction=1.0
        )
        hard_loss = ICH25DSegmentationLoss(
            **common, empty_foreground_top_fraction=1.0 / 64.0
        )
        mask_logits = torch.full((1, 6, 8, 8), -4.0)
        mask_logits[:, 0] = 4.0
        mask_logits[:, 1, 0, 0] = 8.0
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)))
        targets = torch.zeros_like(class_logits)
        arguments = (
            mask_logits,
            class_logits,
            masks,
            targets,
            torch.ones(1),
        )
        average = average_loss.components(*arguments)["empty_foreground"]
        hard = hard_loss.components(*arguments)["empty_foreground"]
        self.assertGreater(float(hard), float(average) * 20.0)

    def test_empty_foreground_top_fraction_is_validated(self):
        with self.assertRaisesRegex(ValueError, "top_fraction"):
            ICH25DSegmentationLoss(
                classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
                empty_foreground_top_fraction=0.0,
            )

    def test_classification_only_row_has_no_segmentation_gradient(self):
        loss_fn = ICH25DSegmentationLoss(
            classification_pos_weight=torch.ones(len(OUTPUT_LABELS)),
            empty_foreground_weight=0.05,
        )
        mask_logits = torch.zeros((1, 6, 8, 8), requires_grad=True)
        class_logits = torch.zeros((1, len(OUTPUT_LABELS)), requires_grad=True)
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        targets = torch.zeros_like(class_logits)
        targets[0, :2] = 1
        components = loss_fn.components(
            mask_logits,
            class_logits,
            masks,
            targets,
            segmentation_known=torch.zeros(1),
            classification_known=torch.ones(1),
        )
        components["loss"].backward()
        self.assertEqual(float(components["segmentation"].detach()), 0.0)
        self.assertTrue(mask_logits.grad is None or not mask_logits.grad.any())
        self.assertGreater(float(class_logits.grad.abs().sum()), 0.0)

    def test_evaluation_uses_only_ich_truth_and_physical_pixels(self):
        rows = []
        for study_id, has_ich in (("normal", False), ("positive", True)):
            row = {
                "study_id": study_id,
                "known": 1,
                "voxel_volume_ml": 0.01,
                "prob_any_ich": 0.9 if has_ich else 0.1,
            }
            for label in OUTPUT_LABELS[1:]:
                is_iph = has_ich and label == "IPH"
                row[f"prob_{label}"] = 0.9 if is_iph else 0.1
                row[f"pred_pixels_{label}"] = 100 if is_iph else 0
                row[f"intersection_{label}"] = 100 if is_iph else 0
                row[f"predicted_known_pixels_{label}"] = 100 if is_iph else 0
                row[f"observed_known_pixels_{label}"] = 100 if is_iph else 0
            rows.append(row)
        truth_rows = []
        for study_id, has_ich in (("normal", False), ("positive", True)):
            truth_row = {"study_id": study_id}
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                truth_row[f"gt_{key}"] = 1.0 if has_ich and key == "V_IPH" else 0.0
            truth_rows.append(truth_row)
        studies, summary = summarize_segmentation_predictions(
            pd.DataFrame(rows), pd.DataFrame(truth_rows)
        )
        self.assertEqual(summary["evaluation_scope"], "ich_only_no_mls_no_fracture_no_triage")
        self.assertEqual(summary["presence_f1_at_0_1ml"], 1.0)
        self.assertAlmostEqual(
            float(studies.loc[studies["study_id"] == "positive", "pred_V_IPH"].iloc[0]),
            1.0,
        )

    def test_evaluation_reports_small_ivh_quality_without_changing_selection(self):
        specifications = (
            ("small-hit", 1.0, 1.0, 10, 10),
            ("small-miss", 2.0, 0.0, 10, 0),
            ("medium", 5.0, 4.0, 10, 8),
            ("large", 12.0, 9.0, 10, 9),
            ("normal", 0.0, 0.0, 0, 0),
        )
        prediction_rows = []
        truth_rows = []
        for (
            study_id,
            true_ivh,
            predicted_ivh,
            observed_pixels,
            intersection,
        ) in specifications:
            row = {
                "study_id": study_id,
                "known": 1,
                "voxel_volume_ml": 1.0,
                "prob_any_ich": 0.9 if predicted_ivh > 0 else 0.1,
            }
            for label in OUTPUT_LABELS[1:]:
                is_ivh = label == "IVH"
                row[f"prob_{label}"] = 0.9 if is_ivh and predicted_ivh > 0 else 0.1
                row[f"pred_pixels_{label}"] = int(predicted_ivh) if is_ivh else 0
                row[f"intersection_{label}"] = intersection if is_ivh else 0
                row[f"predicted_known_pixels_{label}"] = (
                    int(predicted_ivh) if is_ivh else 0
                )
                row[f"observed_known_pixels_{label}"] = (
                    observed_pixels if is_ivh else 0
                )
            prediction_rows.append(row)
            truth_row = {"study_id": study_id}
            for key in ("V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH"):
                truth_row[f"gt_{key}"] = true_ivh if key == "V_IVH" else 0.0
            truth_rows.append(truth_row)

        _, summary = summarize_segmentation_predictions(
            pd.DataFrame(prediction_rows), pd.DataFrame(truth_rows)
        )
        small = summary["subtypes"]["IVH"]["volume_strata"]["small_le_2ml"]
        medium = summary["subtypes"]["IVH"]["volume_strata"][
            "medium_gt_2_le_10ml"
        ]
        large = summary["subtypes"]["IVH"]["volume_strata"]["large_gt_10ml"]
        self.assertEqual(small["positive_studies"], 2)
        self.assertAlmostEqual(small["presence_sensitivity_at_0_1ml"], 0.5)
        self.assertAlmostEqual(small["mae_ml"], 1.0)
        self.assertAlmostEqual(small["dice_known_pixels"], 20.0 / 21.0)
        self.assertEqual(medium["positive_studies"], 1)
        self.assertEqual(large["positive_studies"], 1)
        self.assertAlmostEqual(
            summary["selection_score"],
            0.55 * summary["mean_foreground_dice"]
            + 0.30 * float(summary["any_ich_study_auc"] or 0.0)
            + 0.15 * summary["macro_subtype_study_auc"],
        )
        flattened = _flatten_summary_metrics("calibration", summary)
        self.assertEqual(
            flattened["calibration_ivh_small_le_2ml_positive_studies"], 2.0
        )
        self.assertAlmostEqual(
            flattened[
                "calibration_ivh_small_le_2ml_presence_sensitivity_at_0_1ml"
            ],
            0.5,
        )
        self.assertNotIn("volume_strata", summary)

    def test_oof_metric_vector_reconstructs_perfect_ich_metrics(self):
        rows = []
        for study_id, has_ich in (("normal", False), ("positive", True)):
            row = {
                "study_id": study_id,
                "score_any_ich": 0.9 if has_ich else 0.1,
            }
            for volume_key, label in {
                "V_IVH": "IVH",
                "V_IPH": "IPH",
                "V_SDH": "SDH",
                "V_EDH": "EDH",
                "V_SAH": "SAH",
            }.items():
                is_iph = has_ich and label == "IPH"
                row[f"intersection_{label}"] = 100 if is_iph else 0
                row[f"predicted_known_pixels_{label}"] = 100 if is_iph else 0
                row[f"observed_known_pixels_{label}"] = 100 if is_iph else 0
                row[f"score_{label}"] = 0.9 if is_iph else 0.1
                row[f"gt_{volume_key}"] = 1.0 if is_iph else 0.0
                row[f"pred_{volume_key}"] = 1.0 if is_iph else 0.0
            rows.append(row)
        metrics = _metric_vector(pd.DataFrame(rows), np.ones(2))
        self.assertEqual(metrics["selection_score"], 1.0)
        self.assertEqual(metrics["mean_foreground_dice"], 1.0)
        self.assertEqual(metrics["presence_f1_at_0_1ml"], 1.0)
        self.assertEqual(metrics["normal_false_positive_rate_at_0_1ml"], 0.0)
        self.assertEqual(metrics["total_volume_mae_ml"], 0.0)

    def test_recovery_evaluator_requires_patient_safe_checkpoint_config(self):
        config = {
            "architecture": "unetplusplus",
            "encoder_name": "efficientnet-b2",
            "outer_fold": 0,
            "calibration_fold": 1,
            "batch_size": 16,
            "workers": 4,
            "dropout": 0.2,
            "seed": 42,
        }
        self.assertEqual(checkpoint_config({"config": config}), config)
        with self.assertRaisesRegex(ValueError, "must differ"):
            checkpoint_config({"config": {**config, "calibration_fold": 0}})


if __name__ == "__main__":
    unittest.main()
