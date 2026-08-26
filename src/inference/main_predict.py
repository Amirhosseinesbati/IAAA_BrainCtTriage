"""Local inference facade that deliberately reuses the submission pipeline."""

from __future__ import annotations

from pathlib import Path

from submission.model import load_models, predict as predict_intermediates
from src.evaluation.triage import triage_from_intermediates

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_all_models(device: str = "auto") -> dict:
    """Load exactly the models and preprocessing shipped to the leaderboard."""
    return load_models(str(PROJECT_ROOT / "submission" / "models"), device=device)


def predict(study_dir: str, models: dict) -> int:
    """Return the official triage class for one study; errors are not hidden."""
    values = predict_intermediates(study_dir, models=models)
    return triage_from_intermediates(values)


def predict_with_intermediates(study_dir: str, models: dict) -> tuple[int, dict[str, float]]:
    values = predict_intermediates(study_dir, models=models)
    return triage_from_intermediates(values), values
