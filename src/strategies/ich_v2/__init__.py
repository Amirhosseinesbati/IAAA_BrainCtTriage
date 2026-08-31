"""Leakage-safe, physically calibrated ICH research pipeline.

The v2 package is intentionally isolated from the legacy MONAI strategy.  It
keeps three contracts explicit: unknown annotations are never treated as
background, volumes are measured from the image affine, and every prediction
is attributable to a study-level validation fold.
"""

from src.strategies.ich_v2.geometry import volumes_from_labelmap, voxel_volume_ml

__all__ = ["volumes_from_labelmap", "voxel_volume_ml"]
