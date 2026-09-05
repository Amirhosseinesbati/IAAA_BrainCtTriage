"""Immutable image-channel contract for single-slice and 2.5D MLS inputs.

The deployed MLS geometry is defined in the central slice coordinate frame.
For a 2.5D model we preserve that frame and add only the immediately adjacent
CT slices as context.  This module is deliberately free of model code so the
cache builder and CUDA inference share the exact same window ordering.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.core.dicom_reader import BrainDicomReader


WINDOW_ORDER: tuple[str, str, str] = ("brain", "subdural", "bone")
CONTEXT_CHANNELS = 9


def create_windowed_input(hu_image: np.ndarray, input_channels: int = 3) -> np.ndarray:
    """Return central-slice MLS channels in the canonical window order.

    ``input_channels=9`` is intentionally rejected here because a 2.5D input
    needs the full ordered study volume and must use
    :func:`create_study_windowed_input`.  Silently repeating a central image
    would create a train/inference mismatch while appearing to work.
    """
    image = np.asarray(hu_image)
    if image.ndim != 2:
        raise ValueError(f"Expected one HU slice [H, W], got shape {image.shape}")
    brain = BrainDicomReader.apply_windowing(image, "brain").astype(np.float32, copy=False)
    if input_channels == 1:
        return brain[None, ...]
    if input_channels == 3:
        return np.stack(
            [
                brain,
                BrainDicomReader.apply_windowing(image, "subdural"),
                BrainDicomReader.apply_windowing(image, "bone"),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    if input_channels == CONTEXT_CHANNELS:
        raise ValueError(
            "Nine-channel MLS input requires create_study_windowed_input(volume, index, 9)"
        )
    raise ValueError(f"Unsupported MLS input_channels={input_channels}; expected 1, 3, or 9")


def create_study_windowed_input(
    volume_hu: np.ndarray,
    index: int,
    input_channels: int = 3,
) -> np.ndarray:
    """Return one canonical MLS input from an ordered ``[H, W, D]`` volume.

    For nine channels the order is ``z-1`` windows, central windows, ``z+1``
    windows, each in ``brain, subdural, bone`` order.  Boundary slices use
    edge replication, a deterministic choice also used by the training cache.
    """
    volume = np.asarray(volume_hu)
    if volume.ndim != 3:
        raise ValueError(f"Expected HU volume [H, W, D], got shape {volume.shape}")
    depth = int(volume.shape[2])
    if depth <= 0:
        raise ValueError("MLS input volume must contain at least one slice")
    position = int(index)
    if position < 0 or position >= depth:
        raise IndexError(f"MLS slice index {position} outside [0, {depth})")
    if input_channels in (1, 3):
        return create_windowed_input(volume[:, :, position], input_channels)
    if input_channels != CONTEXT_CHANNELS:
        raise ValueError(f"Unsupported MLS input_channels={input_channels}; expected 1, 3, or 9")
    indices = (max(0, position - 1), position, min(depth - 1, position + 1))
    return np.concatenate(
        [create_windowed_input(volume[:, :, neighbour], 3) for neighbour in indices],
        axis=0,
    )


def context_channels_for_radius(radius: int) -> int:
    """Return channels for a three-window MLS context of ``radius`` slices."""
    value = int(radius)
    if value < 0:
        raise ValueError("MLS context radius cannot be negative")
    return len(WINDOW_ORDER) * (2 * value + 1)
