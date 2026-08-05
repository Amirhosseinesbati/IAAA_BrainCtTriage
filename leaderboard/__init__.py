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
    python -m leaderboard.task_mls --run-inference
    python -m leaderboard.task_hemorrhage --run-inference --compare-all
"""
