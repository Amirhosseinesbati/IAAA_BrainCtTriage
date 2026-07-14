"""
submission.py — Main submission entry point for IAAA 2026 Brain CT Triage.

This script implements the competition-required interface:

    python submission.py \\
        --data-dir /path/to/test/data \\
        --predictions-file-path /path/to/output.csv

It loads the trained models, runs inference on every study in ``data-dir``,
applies the official triage function, and writes a CSV with columns
``id`` (study identifier) and ``prediction`` (triage class 0/1/2).

Replace the placeholder logic in :func:`predict` with your actual model
pipeline once you have trained your models.
"""

import os
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

# Import the competition model API and triage function.
# Both are designed to be self-contained inside the submission zip.
from model import load_models, predict as model_predict
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


def _get_models(models_dir: str = "models", device: str = "cuda"):
    """Load models on first call, then return cached instance."""
    global _models
    if _models is None:
        logger.info("Loading models from '%s' ...", models_dir)
        _models = load_models(models_dir=models_dir, device=device)
        logger.info("Models loaded successfully.")
    return _models


# ---------------------------------------------------------------------------
# Required API functions
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> pd.DataFrame:
    """Discover all study directories under *data_dir*.

    Expected layout::

        data_dir/
        ├── study_001/
        │   ├── slice_001.dcm
        │   ├── slice_002.dcm
        │   └── ...
        ├── study_002/
        │   └── ...
        └── ...

    Args:
        data_dir: Path to the root directory containing one sub-directory
                  per study (each containing ``*.dcm`` files).

    Returns:
        A DataFrame with a single column ``"study_dir"`` containing the
        full path to each study directory.  The index is the study id
        (the directory basename).
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"data_dir does not exist: {data_dir}")

    study_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir()]
    )
    if not study_dirs:
        raise FileNotFoundError(
            f"No study sub-directories found under {data_dir}"
        )

    records = []
    for sd in study_dirs:
        # Ignore hidden directories and non-digit names (adjust if needed)
        if sd.name.startswith("."):
            continue
        records.append({"id": sd.name, "study_dir": str(sd.resolve())})

    result = pd.DataFrame(records).set_index("id")
    logger.info("Found %d studies under %s", len(result), data_dir)
    return result


def predict() -> np.ndarray:
    """Run inference on all studies and return triage predictions.

    **Replace this placeholder with your actual model logic.**

    The current implementation:
        1. Loads study information from the calling context (saved in module
           state by :func:`main`).
        2. Iterates over each study directory.
        3. Calls ``model.predict(study_dir)`` to obtain the 7 intermediate
           imaging primitives.
        4. Applies ``triage_from_intermediates()`` to get the triage class.
        5. Falls back to a random baseline if the model is not yet available.

    Returns:
        A 1-D numpy array of integer triage classes (0, 1, or 2) with
        length equal to the number of studies in ``load_data()``.
    """
    # -------- BEGIN CUSTOM MODEL LOGIC ------------------------------------
    #
    # You can replace everything below with your own inference pipeline.
    # The only requirement is that you return an np.ndarray of ints
    # where each element is 0 (non-urgent), 1 (urgent), or 2 (critical).

    studies = _STUDIES  # set by main() before calling predict()

    # Attempt to use the real models.  If loading fails (e.g. no weights
    # present yet), fall back to a random baseline so the script can still
    # be tested for structural correctness.
    try:
        models = _get_models()
        predictions = []
        for study_id, row in studies.iterrows():
            logger.info("Processing study %s ...", study_id)
            intermediates = model_predict(row["study_dir"], models=models)
            triage_class = triage_from_intermediates(intermediates)
            predictions.append(triage_class)
        return np.array(predictions, dtype=int)

    except Exception as exc:
        logger.warning(
            "Model inference failed (%s). Falling back to random baseline.",
            exc,
        )
        rng = np.random.default_rng(seed=42)
        return rng.integers(0, 2, size=len(studies))

    # -------- END CUSTOM MODEL LOGIC --------------------------------------


def save_predictions(predictions: np.ndarray, output_path: str) -> None:
    """Write predictions to a CSV file.

    Args:
        predictions: 1-D array of triage classes (0, 1, 2).
        output_path: Destination CSV path.
    """
    studies = _STUDIES
    result = pd.DataFrame({
        "id": studies.index.values,
        "prediction": predictions,
    })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    click.echo(f"Saved {len(result)} predictions to {output_path}")


# ---------------------------------------------------------------------------
# Module-level state (set by main, consumed by predict / save_predictions)
# ---------------------------------------------------------------------------
_STUDIES: pd.DataFrame = None


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
    default="cuda",
    show_default=True,
    type=click.Choice(["cuda", "cpu"]),
    help="Device to run inference on.",
)
def main(data_dir: str, predictions_file_path: str,
         models_dir: str, device: str):
    """IAAA 2026 Brain CT Triage — submission entry point.

    Loads models, runs inference on every study in DATA_DIR, applies the
    official triage function, and writes a CSV with columns ``id`` and
    ``prediction``.
    """
    global _STUDIES

    click.echo("=" * 60)
    click.echo("  IAAA 2026 Brain CT Triage — Submission")
    click.echo("=" * 60)

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
