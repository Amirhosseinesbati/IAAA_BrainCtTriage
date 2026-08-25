"""
submission.py — Main submission entry point for the IAAA 2026 Brain CT Triage
leaderboard.

Implements the leaderboard-required interface:

    python submission.py \\
        --data-dir /path/to/data/dir \\
        --predictions-file-path /path/to/submission.csv

The script discovers every study under ``data_dir`` (one sub-directory per
study, each containing ``*.dcm`` files), runs the bundled models
(``model.py``), applies the official triage function (``triage.py``) and
writes a CSV with columns ``id`` (study identifier) and ``prediction``
(triage class 0 / 1 / 2).

The models are loaded **once** at the start and cached. If a single study
fails during inference it is logged and predicted as 0 (non-urgent) so the
run always completes and produces a valid CSV.
"""

import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd

# Self-contained modules shipped inside the submission zip.
from model import INFERENCE_CONFIG, load_models, predict as model_predict
from triage import triage_from_intermediates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Globally cached models (loaded once at startup)
# ---------------------------------------------------------------------------
_models = None


def _get_models() -> dict:
    """Load models on first call, then return the cached instance."""
    global _models
    if _models is None:
        logger.info("Loading models from '%s' [device: %s] ...",
                    _MODELS_DIR, _DEVICE)
        _models = load_models(
            models_dir=_MODELS_DIR,
            device=_DEVICE,
            mls_min_peak=_MLS_MIN_PEAK,
            mls_top_k=_MLS_TOP_K,
            mls_batch_size=_MLS_BATCH_SIZE,
            mls_aggregation=_MLS_AGGREGATION,
        )
        logger.info("Models loaded successfully.")
    return _models


# ---------------------------------------------------------------------------
# Required API functions
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> pd.DataFrame:
    """Discover all studies under ``data_dir``.

    Expected layout (competition)::

        data_dir/
        ├── study_001/
        │   ├── slice_001.dcm
        │   ├── slice_002.dcm
        │   └── ...
        ├── study_002/
        │   └── ...
        └── ...

    If ``data_dir`` contains ``*.dcm`` files directly, the whole directory is
    treated as a single study.

    Returns:
        A DataFrame with columns ``id`` (study identifier) and ``study_dir``
        (full path to the study directory), indexed by ``id``.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"data_dir does not exist: {data_dir}")

    study_dirs = sorted(
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    records = []
    for sd in study_dirs:
        records.append({"id": sd.name, "study_dir": str(sd.resolve())})

    # Fallback: data_dir itself is a single study (contains .dcm files).
    if not records:
        if list(data_path.glob("*.dcm")):
            records.append({
                "id": data_path.name,
                "study_dir": str(data_path.resolve()),
            })

    if not records:
        raise FileNotFoundError(
            f"No study directories (or .dcm files) found under {data_dir}"
        )

    result = pd.DataFrame(records).set_index("id")
    logger.info("Found %d studies under %s", len(result), data_dir)
    return result


def predict() -> np.ndarray:
    """Run inference on all studies and return triage predictions.

    For every study: ``model.predict(study_dir)`` → 7 intermediate imaging
    primitives → ``triage_from_intermediates()`` → triage class 0/1/2.

    Per-study failures are logged and predicted as 0 so the run always
    completes with a valid CSV (no silent random baseline).

    Returns:
        1-D numpy array of integer triage classes (0, 1, or 2).
    """
    studies = _STUDIES
    models = _get_models()

    predictions = []
    failed: list[str] = []

    for study_id, row in studies.iterrows():
        logger.info("Processing study %s ...", study_id)
        try:
            intermediates = model_predict(row["study_dir"], models=models)
            predictions.append(int(triage_from_intermediates(intermediates)))
        except Exception as exc:  # noqa: BLE001
            failed.append(str(study_id))
            logger.error("  ✗ Study %s failed (%s). Predicted 0.", study_id, exc)
            predictions.append(0)

    if failed:
        logger.warning(
            "%d/%d studies failed and were predicted as 0 (non-urgent).",
            len(failed), len(studies),
        )

    return np.array(predictions, dtype=int)


def save_predictions(predictions: np.ndarray, output_path: str) -> None:
    """Write predictions to a CSV file.

    Args:
        predictions: 1-D array of triage classes (0, 1, 2).
        output_path: Destination CSV path.
    """
    result = pd.DataFrame({
        "id": _STUDIES.index.values,
        "prediction": predictions,
    })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    click.echo(f"Saved {len(result)} predictions to {output_path}")


# ---------------------------------------------------------------------------
# Module-level state (set by main, consumed by predict / save_predictions)
# ---------------------------------------------------------------------------
_STUDIES: pd.DataFrame = None
_MODELS_DIR = "models"
_DEVICE = "auto"
_MLS_MIN_PEAK = float(INFERENCE_CONFIG["mls"]["min_peak"])
_MLS_TOP_K = INFERENCE_CONFIG["mls"]["top_k"]
_MLS_BATCH_SIZE = int(INFERENCE_CONFIG["mls"]["batch_size"])
_MLS_AGGREGATION = INFERENCE_CONFIG["mls"]["aggregation"]


@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing one sub-directory per study (each with *.dcm files).",
)
@click.option(
    "--predictions-file-path",
    required=True,
    type=click.Path(),
    help="Path to write the output predictions CSV.",
)
@click.option(
    "--models-dir",
    default="models",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing trained model weights.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cuda", "cpu"]),
    help="Device to run inference on (auto = cuda if available).",
)
@click.option(
    "--mls-min-peak",
    default=float(INFERENCE_CONFIG["mls"]["min_peak"]),
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum heatmap peak (all 3 MLS keypoints) to trust a slice.",
)
@click.option(
    "--mls-top-k",
    default=None,
    type=click.INT,
    help="If set, keep the top-K most confident slices instead of --mls-min-peak.",
)
@click.option(
    "--mls-aggregation",
    default=INFERENCE_CONFIG["mls"]["aggregation"],
    show_default=True,
    type=click.Choice(["max", "p90"]),
    help="How to aggregate per-slice MLS values.",
)
@click.option(
    "--mls-batch-size",
    default=int(INFERENCE_CONFIG["mls"]["batch_size"]),
    show_default=True,
    type=click.IntRange(1, 128),
    help="Slices per heatmap forward pass.",
)
def main(
    data_dir: str,
    predictions_file_path: str,
    models_dir: str,
    device: str,
    mls_min_peak: float,
    mls_top_k: int,
    mls_aggregation: str,
    mls_batch_size: int,
):
    """IAAA 2026 Brain CT Triage — leaderboard submission entry point."""
    global _STUDIES, _MODELS_DIR, _DEVICE
    global _MLS_MIN_PEAK, _MLS_TOP_K, _MLS_AGGREGATION, _MLS_BATCH_SIZE

    click.echo("=" * 60)
    click.echo("  IAAA 2026 Brain CT Triage — Submission")
    click.echo("=" * 60)

    _MODELS_DIR = models_dir
    _DEVICE = device
    _MLS_MIN_PEAK = mls_min_peak
    _MLS_TOP_K = mls_top_k
    _MLS_AGGREGATION = mls_aggregation
    _MLS_BATCH_SIZE = mls_batch_size

    # 1. Discover studies
    _STUDIES = load_data(data_dir)
    click.echo(f"  Studies found: {len(_STUDIES)}")

    # 2. Run inference
    click.echo("  Running inference ...")
    predictions = predict()

    # 3. Save output
    save_predictions(predictions, predictions_file_path)

    # Summary
    unique, counts = np.unique(predictions, return_counts=True)
    distribution = dict(zip(unique, counts))
    click.echo(f"  Prediction distribution: {distribution}")
    click.echo("Done.")


if __name__ == "__main__":
    main()
