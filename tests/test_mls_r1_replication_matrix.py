"""CPU-only contract tests for the R1R continuation matrix helpers."""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scripts.materialize_mls_r1_replication_matrix import (
    ARMS,
    AUDIT_SEEDS,
    _canonical_sha256,
    _field_differences,
    _model_signature,
)
from scripts.validate_mls_r1_replication_matrix import _validate_config_family
from src.strategies.config_models import MLSHeatmapConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "config" / "experiments" / "mls-vast-deploy-aligned-baseline-template.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R1ReplicationMatrixTests(unittest.TestCase):
    def test_model_signature_ignores_only_fold_and_seed(self) -> None:
        control = {"fold": 1, "seed": 42, "batch_size": 5, "horizontal_flip_prob": 0.0}
        replica = {"fold": 1, "seed": 2026, "batch_size": 5, "horizontal_flip_prob": 0.0}
        self.assertEqual(_model_signature(control), _model_signature(replica))
        self.assertEqual(_field_differences(control, replica), ["seed"])

    def test_config_family_accepts_exact_paired_seed_replication(self) -> None:
        contract = self._contract_with_yaml_members()
        collected = _validate_config_family(contract)
        self.assertEqual(set(collected), set(ARMS))
        self.assertEqual(set(collected["control"]), set(AUDIT_SEEDS))

    def test_config_family_rejects_hidden_recipe_change(self) -> None:
        contract = self._contract_with_yaml_members(candidate_seed3407_batch_size=6)
        with self.assertRaisesRegex(ValueError, "differs from its arm signature"):
            _validate_config_family(contract)

    def _contract_with_yaml_members(
        self, *, candidate_seed3407_batch_size: int | None = None,
    ) -> dict:
        # Keep the temporary directory alive for the duration of this test by
        # attaching it to the instance. unittest calls tearDown afterward.
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
        arms: dict[str, dict] = {}
        members: dict[str, dict] = {}
        for arm, arm_spec in ARMS.items():
            arm_members: dict[str, dict] = {}
            reference_signature = None
            for seed in AUDIT_SEEDS:
                payload = copy.deepcopy(template)
                config = payload["training_config"]
                config.update({
                    "fold": 1,
                    "seed": seed,
                    "dataset_variant": "multitask_2p5d_v1",
                    "input_channels": 3,
                    "horizontal_flip_prob": arm_spec["horizontal_flip_prob"],
                    "resume_checkpoint": None,
                })
                if arm == "candidate" and seed == 3407 and candidate_seed3407_batch_size is not None:
                    config["batch_size"] = candidate_seed3407_batch_size
                config = MLSHeatmapConfig.model_validate(config).model_dump(mode="json")
                payload["training_config"] = config
                path = directory / f"{arm}-{seed}.yaml"
                path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
                signature = _model_signature(config)
                if reference_signature is None:
                    reference_signature = signature
                member = {
                    "member_kind": "inherited_r1_seed42" if seed == 42 else "planned_r1r_replica",
                    "seed": seed,
                    "config_path": str(path),
                    "config_sha256": _sha256(path),
                }
                if seed == 42:
                    member.update({"checkpoint_path": str(directory / "inherited.pth"), "checkpoint_sha256": "a" * 64})
                else:
                    member["expected_checkpoint_path"] = str(directory / f"epoch15-{arm}-{seed}.pth")
                arm_members[f"seed{seed}"] = member
            arms[arm] = {
                "horizontal_flip_prob": arm_spec["horizontal_flip_prob"],
                "model_config_signature": reference_signature,
                "model_config_signature_sha256": _canonical_sha256(reference_signature),
            }
            members[arm] = arm_members
        return {"arms": arms, "members": members}


if __name__ == "__main__":
    unittest.main()
