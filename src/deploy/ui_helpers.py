"""Pure helpers used by the Streamlit experiment control center."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from src.deploy.experiment import ExperimentManifest, HardwareSpec, RuntimeSpec
from src.config import config_section


def parse_tags(payload: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Tag line {line_number} must use key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or not value:
            raise ValueError(f"Invalid tag at line {line_number}")
        tags[key] = value
    return tags


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.").lower()
    return slug or "experiment"


def build_manifest(
    *,
    task: str,
    strategy: str,
    run_name: str,
    notes: str,
    tags: Mapping[str, Any],
    training_config: Mapping[str, Any],
    gpu_profile: str,
    disk_gb: int,
    max_price_per_hour: float,
    min_reliability: float,
    git_branch: str,
    prepare_data: bool,
    auto_destroy: bool,
) -> ExperimentManifest:
    return ExperimentManifest(
        task=task,
        strategy=strategy,
        run_name=run_name,
        notes=notes,
        tags=dict(tags),
        training_config=dict(training_config),
        hardware=HardwareSpec(
            gpu_profile=gpu_profile, disk_gb=disk_gb,
            max_price_per_hour=max_price_per_hour, min_reliability=min_reliability,
        ),
        runtime=RuntimeSpec(
            git_branch=git_branch, prepare_data=prepare_data, auto_destroy=auto_destroy,
        ),
    )


def save_manifest(manifest: ExperimentManifest, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{slugify(manifest.run_name)}.yaml"
    if target.exists():
        index = 1
        while (directory / f"{target.stem}-{index}.yaml").exists():
            index += 1
        target = directory / f"{target.stem}-{index}.yaml"
    target.write_text(manifest.to_yaml(), encoding="utf-8")
    return target


def expand_fold_suite(manifest: ExperimentManifest) -> list[ExperimentManifest]:
    """Create one validated manifest per immutable competition fold."""
    if manifest.task == "triage_calibration":
        raise ValueError("Calibration consumes all OOF folds and cannot be expanded")
    n_folds = int(config_section("competition", "evaluation", "n_folds"))
    stem = re.sub(r"(?:[-_. ]*fold[-_. ]*\d+)$", "", manifest.run_name, flags=re.IGNORECASE).rstrip("-_. ")
    suite: list[ExperimentManifest] = []
    for fold in range(n_folds):
        payload = manifest.model_dump(mode="python")
        payload["run_name"] = f"{stem}-fold-{fold}"
        payload["training_config"] = {**payload["training_config"], "fold": fold}
        payload["tags"] = {**payload["tags"], "fold": fold, "suite": stem}
        suite.append(ExperimentManifest.model_validate(payload))
    return suite
