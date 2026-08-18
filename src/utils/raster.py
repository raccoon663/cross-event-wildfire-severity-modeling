"""Small raster I/O helpers shared across feature builders.

``read_float_bands`` centralises the pattern of reading every band, converting to
float32 and turning the stored nodata value into ``NaN``. ``require_same_grid``
enforces the invariant that every input raster in the Canadian pipeline shares
one grid, so a mismatched cache is a hard error instead of a silent alignment
problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

__all__ = ["read_float_bands", "require_same_grid"]


def read_float_bands(path: Path) -> tuple[np.ndarray, dict, rasterio.Affine, list]:
    """Read a raster as float32 with nodata converted to ``NaN``.

    Returns ``(values, profile, transform, descriptions)`` where ``values`` has
    shape ``(bands, height, width)``.
    """
    path = Path(path)
    with rasterio.open(path) as dataset:
        values = dataset.read().astype("float32")
        nodata = dataset.nodata
        profile = dataset.profile.copy()
        transform = dataset.transform
        descriptions = list(dataset.descriptions)
    if nodata is not None:
        values[values == nodata] = np.nan
    return values, profile, transform, descriptions


def require_same_grid(reference_profile: dict, profile: dict, path: Path) -> None:
    """Raise when a raster does not match the reference grid exactly."""
    keys = ("width", "height", "crs", "transform")
    if any(reference_profile[key] != profile[key] for key in keys):
        raise ValueError(f"Grid mismatch: {path}")
