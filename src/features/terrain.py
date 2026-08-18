"""Terrain derivatives from a DEM window.

Slope and aspect are computed with a central-difference gradient at the raster
resolution. Aspect is a circular variable, so it is supplied to the models as its
sine and cosine components rather than as degrees, where 359 deg and 1 deg would
look maximally different to a tree-based splitter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["slope_aspect", "terrain_from_dem", "TERRAIN_FEATURE_NAMES"]

TERRAIN_FEATURE_NAMES = ["elevation", "slope", "aspect_sin", "aspect_cos"]


def slope_aspect(
    elevation: np.ndarray, resolution: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(slope_degrees, aspect_degrees, aspect_sin, aspect_cos)``.

    Aspect is measured clockwise from north in ``[0, 360)``.
    """
    gradient_y, gradient_x = np.gradient(elevation, resolution, resolution)
    slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y))).astype("float32")
    aspect = ((np.degrees(np.arctan2(-gradient_x, gradient_y)) + 360) % 360).astype("float32")
    aspect_sin = np.sin(np.deg2rad(aspect)).astype("float32")
    aspect_cos = np.cos(np.deg2rad(aspect)).astype("float32")
    return slope, aspect, aspect_sin, aspect_cos


def terrain_from_dem(
    dem_path: Path, expected_profile: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a DEM window and derive terrain layers on the same grid.

    Returns ``(elevation, slope, aspect, aspect_sin, aspect_cos)``. When
    ``expected_profile`` is supplied the DEM grid is verified against it, so a
    silently mismatched cache can never enter the feature table.
    """
    from ..utils.raster import read_float_bands, require_same_grid

    values, profile, _, _ = read_float_bands(dem_path)
    if expected_profile is not None:
        require_same_grid(expected_profile, profile, dem_path)
    elevation = values[0]
    resolution = abs(float(profile["transform"].a))
    slope, aspect, aspect_sin, aspect_cos = slope_aspect(elevation, resolution)
    return elevation, slope, aspect, aspect_sin, aspect_cos
