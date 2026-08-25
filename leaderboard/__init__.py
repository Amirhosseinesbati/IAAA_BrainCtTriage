"""
leaderboard — Personal leaderboard for IAAA 2026 Brain CT Triage Challenge.

Evaluates trained models (placed in submission/models/) against ground truth
from Data/metadata/training_df.csv.

Modules
-------
evaluate             Main entry point — evaluates the full triage pipeline
                     (ICH → fracture → MLS → triage class) and computes
                     Quadratic Weighted Kappa (QWK), the official metric.

ground_truth         Loads and aggregates slice-level CSV to study-level labels.

scorer               QWK computation + supplementary metrics + export utilities.

task_fracture         🦴 Task-specific: skull fracture binary detection
                     (AUC-ROC, accuracy, precision, recall, F1, optimal threshold).

task_mls              📏 Task-specific: midline shift regression + threshold
                     classification (MAE, RMSE, R², Bland-Altman, ≥3mm/≥5mm).
                     ``--run-inference`` auto-detects the ``mls_heatmap``
                     strategy model placed in ``submission/models/mls_heatmap/``.

task_hemorrhage       🩸 Task-specific: ICH detection + volume estimation
                     (per-type & AnyICH: AUC-ROC, F1; per-type & total volume:
                     MAE, RMSE, R²). Supports --compare-all across strategies.

Usage
-----
    # Full triage (official competition metric)
    python -m leaderboard.evaluate

    # Task-specific leaderboards (CSV mode — from existing results.csv)
    python -m leaderboard.task_fracture
    python -m leaderboard.task_mls
    python -m leaderboard.task_hemorrhage

    # Task-specific leaderboards (inference mode — runs models directly)
    python -m leaderboard.task_fracture --run-inference
    python -m leaderboard.task_mls --run-inference          # MLS-only, auto-detects heatmap
    python -m leaderboard.task_hemorrhage --run-inference --compare-all
"""


def normalize_study_id(value) -> str:
    """Normalize a study identifier to a stable string key.

    Pandas reads an integer-looking CSV column back as float by default
    (e.g. ``1011`` → ``1011.0``), so ``str(value)`` produces ``"1011.0"``
    while the ground-truth loader produces ``"1011"`` — causing study
    matching to fail with zero overlap. This helper canonicalizes both
    to ``"1011"``.

    Args:
        value: Raw study id from a dataframe cell (int, float, str, ...).

    Returns:
        Stable string key.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

