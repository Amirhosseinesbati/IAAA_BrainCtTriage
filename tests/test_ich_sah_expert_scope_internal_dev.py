from __future__ import annotations

import torch

from scripts.diagnose_ich_sah_expert_scope_internal_dev import (
    _scope_gate,
)


def _metric(ap: float, auc: float) -> dict[str, float | int]:
    return {
        "positive_pixels": 1000,
        "negative_pixels": 100_000,
        "prevalence": 0.01,
        "average_precision": ap,
        "roc_auc": auc,
        "precision_at_recall_0_10": 0.1,
        "precision_at_recall_0_25": 0.05,
    }


def test_scope_gate_requires_primary_and_near_field_gain() -> None:
    metrics = {
        "background_or_iph": {
            "expert_raw": _metric(0.08, 0.90),
            "incumbent_raw": _metric(0.06, 0.87),
        },
        "near_incumbent_foreground": {
            "expert_raw": _metric(0.16, 0.90),
            "incumbent_raw": _metric(0.12, 0.87),
        },
    }
    assert _scope_gate(metrics)["all_passed"]
    metrics["near_incumbent_foreground"]["expert_raw"] = _metric(0.125, 0.90)
    assert not _scope_gate(metrics)["all_passed"]


def test_scope_probe_import_does_not_change_torch_grad_mode() -> None:
    assert torch.is_grad_enabled()
