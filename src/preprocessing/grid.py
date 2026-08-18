"""Event-aligned analysis grids.

Both the resolution ablation and the cross-event transfer require a grid that is
snapped to whole multiples of the pixel size. Without snapping, the 30 m and 90 m
runs would not share pixel edges and the resolution comparison would confound
resampling phase with resolution.

A two-pixel pad is added on every side so that the perimeter rasterisation and
the neighbourhood of edge pixels stay inside the raster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds, transform_geom

__all__ = ["geometry_coordinates", "event_grid", "rasterize_geometry"]


def geometry_coordinates(geometry: dict[str, Any]) -> list[list[float]]:
    """Flatten any GeoJSON geometry into a list of coordinate pairs."""
    coordinates: list[list[float]] = []

    def collect(node: Any) -> None:
        if isinstance(node, list) and node and isinstance(node[0], (int, float)):
            coordinates.append(node)
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(geometry["coordinates"])
    return coordinates


def event_grid(
    feature: dict[str, Any], resolution: int, crs: str, pad_pixels: int = 2
) -> tuple[Affine, int, int, tuple[float, float, float, float]]:
    """Build a pixel-aligned grid covering one event polygon.

    Returns ``(transform, width, height, bounds)`` in the target CRS. Bounds are
    floored/ceiled onto the resolution lattice, so grids at different
    resolutions remain nested.
    """
    coordinates = geometry_coordinates(feature["geometry"])
    longitudes, latitudes = zip(*[(point[0], point[1]) for point in coordinates])
    bounds = transform_bounds(
        "EPSG:4326", crs,
        min(longitudes), min(latitudes), max(longitudes), max(latitudes),
        densify_pts=21,
    )
    left, bottom, right, top = bounds
    pad = resolution * pad_pixels
    left = np.floor((left - pad) / resolution) * resolution
    bottom = np.floor((bottom - pad) / resolution) * resolution
    right = np.ceil((right + pad) / resolution) * resolution
    top = np.ceil((top + pad) / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)
    return transform, width, height, (left, bottom, right, top)


def rasterize_geometry(
    geometry: dict[str, Any], transform: Affine, width: int, height: int,
    crs: str, source_crs: str = "EPSG:4326", precision: int = 3,
) -> np.ndarray:
    """Burn one polygon onto the analysis grid and return a boolean mask."""
    projected = transform_geom(source_crs, crs, geometry, precision=precision)
    burned = rasterize(
        [(projected, 1)], out_shape=(height, width), transform=transform,
        fill=0, dtype="uint8",
    )
    return burned.astype(bool)
