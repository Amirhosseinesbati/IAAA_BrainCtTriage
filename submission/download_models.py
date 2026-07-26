"""
download_models.py — Download trained models from MLflow Model Registry.

After you have manually registered your best models in the MLflow Model
Registry (via DagsHub UI or CLI), run this script to download the model
artifacts and place them in the correct locations under ``submission/models/``.

Prerequisites:
    - You have registered models in MLflow with names like:
        * ``ich_nnunet``       (ICH segmentation, nnU-Net)
        * ``fracture_yolo``    (fracture detection, YOLO)
        * ``mls_slice_selector`` (MLS slice selector)
        * ``mls_keypoint``     (MLS keypoint detector)
    - MLflow tracking URI is accessible (local ``file:`` or remote DagsHub).

Usage examples:

    # Download from DagsHub (default)
    python download_models.py --model-name ich_nnunet --version 1

    # Download latest version
    python download_models.py --model-name fracture_yolo --version latest

    # Download from a local MLflow server
    python download_models.py \\
        --tracking-uri file:///path/to/logs/mlflow_runs \\
        --model-name ich_nnunet --version 1

    # Download all four models at once
    python download_models.py --all
"""

import os
import shutil
import logging
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: model name  ->  target sub-directory in submission/models/
# ---------------------------------------------------------------------------
MODEL_TARGETS = {
    "ich_nnunet":        "nnunet",
    "fracture_yolo":     "yolo",
    "mls_slice_selector": "mls",
    "mls_keypoint":       "mls",
    "mls_heatmap":        "mls_heatmap",
}

SUBMISSION_MODELS_DIR = Path(__file__).resolve().parent / "models"


def _ensure_target_dir(target_subdir: str) -> Path:
    """Create the target sub-directory if it does not exist."""
    target = SUBMISSION_MODELS_DIR / target_subdir
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_single_model(
    tracking_uri: str,
    model_name: str,
    version: str,
    target_subdir: str,
    overwrite: bool = False,
) -> Path:
    """Download a single model from MLflow Registry.

    Args:
        tracking_uri: MLflow tracking URI.
        model_name: Name of the registered model.
        version: Model version (e.g. ``"1"``, ``"latest"``).
        target_subdir: Sub-directory under ``models/`` to place files into.
        overwrite: If ``True``, remove existing files first.

    Returns:
        Path to the target directory.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    # Resolve "latest" to an actual version string
    if version.lower() == "latest":
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            raise RuntimeError(
                f"No versions found for model '{model_name}'. "
                "Make sure the model is registered in the MLflow Model Registry."
            )
        # Pick the newest version (highest version number)
        latest = max(latest_versions, key=lambda v: int(v.version))
        version = latest.version
        click.echo(f"Resolved '{model_name}:latest' → version {version}")

    target_dir = _ensure_target_dir(target_subdir)
    if overwrite and target_dir.exists():
        # Remove old checkpoint files but keep .gitkeep
        for f in target_dir.glob("*"):
            if f.name != ".gitkeep":
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

    click.echo(
        f"Downloading model '{model_name}' version {version} "
        f"to {target_dir} ..."
    )

    # Download artifacts from the Model Registry
    model_uri = f"models:/{model_name}/{version}"
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=None,
        dst_path=str(target_dir),
        tracking_uri=tracking_uri,
        model_uri=model_uri,
    )

    click.echo(f"  ✓ Downloaded to: {local_path}")
    return Path(local_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(
    help="Download trained models from MLflow Model Registry into submission/models/."
)
@click.option(
    "--tracking-uri",
    default=None,
    help=(
        "MLflow tracking URI.  If not set, reads from the environment "
        "variable MLFLOW_TRACKING_URI.  Example for DagsHub: "
        "https://dagshub.com/youruser/yourrepo.mlflow"
    ),
)
@click.option(
    "--model-name",
    default=None,
    help=(
        "Name of the registered model in MLflow Model Registry. "
        "Ignored when --all is used."
    ),
)
@click.option(
    "--version",
    default="latest",
    show_default=True,
    help="Model version number or 'latest'.",
)
@click.option(
    "--all",
    "download_all",
    is_flag=True,
    default=False,
    help="Download all four models (ich_nnunet, fracture_yolo, mls_slice_selector, mls_keypoint).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Remove existing model files before downloading.",
)
def main(tracking_uri: str, model_name: str, version: str,
         download_all: bool, overwrite: bool):
    """Download trained models from MLflow Model Registry."""
    # Resolve tracking URI
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        click.echo(
            "Error: --tracking-uri is not set and MLFLOW_TRACKING_URI "
            "environment variable is not defined.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"MLflow tracking URI: {tracking_uri}")
    click.echo(f"Target directory:    {SUBMISSION_MODELS_DIR}")

    if download_all:
        models_to_download = list(MODEL_TARGETS.items())
    elif model_name:
        if model_name not in MODEL_TARGETS:
            click.echo(
                f"Error: Unknown model name '{model_name}'. "
                f"Available models: {', '.join(MODEL_TARGETS.keys())}",
                err=True,
            )
            raise SystemExit(1)
        models_to_download = [(model_name, MODEL_TARGETS[model_name])]
    else:
        click.echo(
            "Error: Specify either --model-name or --all.",
            err=True,
        )
        raise SystemExit(1)

    for m_name, m_target in models_to_download:
        try:
            download_single_model(
                tracking_uri=tracking_uri,
                model_name=m_name,
                version=version,
                target_subdir=m_target,
                overwrite=overwrite,
            )
        except Exception as exc:
            logger.error("Failed to download '%s': %s", m_name, exc)
            raise

    click.echo("\n✓ All models downloaded successfully.")
    click.echo(f"  → {SUBMISSION_MODELS_DIR}")


if __name__ == "__main__":
    main()
