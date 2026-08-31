"""Explicit partial-label handling for the ICH dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


ICH_AREA_COLUMNS = (
    "IntraventricularHemorrhage_Area",
    "IntraparenchymalHemorrhage_Area",
    "SubduralHemorrhage_Area",
    "EpiduralHemorrhage_Area",
    "SubarachnoidHemorrhage_Area",
)


def clean_negative_study_ids(metadata: pd.DataFrame) -> set[str]:
    """Return studies that metadata proves contain no ICH and are non-urgent.

    ``AnyICH`` is deliberately ignored because the supplied column omits
    IVH-only cases.  The gate instead requires zero annotated area for all five
    subtypes and triage class 0 on every slice row.
    """
    required = {"dicom_series.id", "triage_class", *ICH_AREA_COLUMNS}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing clean-negative columns: {sorted(missing)}")

    frame = metadata.loc[:, list(required)].copy()
    frame["study_id"] = frame.pop("dicom_series.id").astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    for column in ICH_AREA_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["triage_class"] = pd.to_numeric(frame["triage_class"], errors="raise")

    aggregate = frame.groupby("study_id", as_index=True).agg(
        **{f"sum_{column}": (column, "sum") for column in ICH_AREA_COLUMNS},
        triage_min=("triage_class", "min"),
        triage_max=("triage_class", "max"),
    )
    area_columns = [column for column in aggregate if column.startswith("sum_")]
    is_zero_ich = aggregate[area_columns].abs().sum(axis=1) <= 1e-8
    is_non_urgent = (aggregate["triage_min"] == 0) & (aggregate["triage_max"] == 0)
    return set(aggregate.index[is_zero_ich & is_non_urgent].astype(str))


def stack_partial_targets(
    parsed_slices: Iterable[Mapping[str, object]],
    *,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack slice labels and a voxel mask that marks known supervision."""
    targets: list[np.ndarray] = []
    supervision: list[np.ndarray] = []
    for parsed in parsed_slices:
        mask = np.asarray(parsed["mask_2d"], dtype=np.uint8)
        if mask.shape != shape:
            raise ValueError(f"Annotation shape {mask.shape} does not match DICOM shape {shape}")
        known = bool(parsed.get("has_label", False))
        targets.append(mask)
        supervision.append(np.full(shape, 1 if known else 0, dtype=np.uint8))
    if not targets:
        raise ValueError("Cannot stack an empty DICOM series")
    return np.stack(targets, axis=0), np.stack(supervision, axis=0)


def full_negative_targets(
    depth: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create fully supervised background targets for a proven clean series."""
    if depth <= 0 or min(shape) <= 0:
        raise ValueError("A clean-negative target needs positive spatial dimensions")
    target = np.zeros((depth, *shape), dtype=np.uint8)
    return target, np.ones_like(target, dtype=np.uint8)
