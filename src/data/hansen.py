"""Hansen Global Forest Change tree cover and loss year.

Two layers are used for the Canadian experiment:

``treecover2000``
    Percent canopy cover in 2000; the forest mask threshold is applied to it.
``lossyear``
    Year of stand-replacing loss encoded as ``1 = 2001`` … ``23 = 2023``.

Loss *strictly before* the fire year is excluded from the analysis population,
because those pixels were no longer forest when the study fire burned. Loss in
the fire year itself is retained: that code can be the study fire, and removing
it would delete exactly the pixels being modelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from ..preprocessing.reproject import warp_to_reference

__all__ = [
    "HANSEN_VERSION",
    "HANSEN_ROOT",
    "LAYERS",
    "tile_name",
    "tile_url",
    "fetch_layers",
    "prefire_loss_mask",
    "forest_mask",
]

HANSEN_VERSION = "GFC-2023-v1.11"
HANSEN_ROOT = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"
LAYERS = ("treecover2000", "lossyear")


def tile_name(bbox: list[float] | tuple[float, float, float, float]) -> str:
    """Return the 10-degree Hansen tile name covering a bounding box corner.

    Hansen tiles are named by their north-west corner, e.g. ``60N_130W``.
    """
    longitude = int(np.floor(bbox[0] / 10)) * 10
    latitude = int(np.ceil(bbox[3] / 10)) * 10
    north_south = f"{abs(latitude):02d}{'N' if latitude >= 0 else 'S'}"
    east_west = f"{abs(longitude):03d}{'E' if longitude >= 0 else 'W'}"
    return f"{north_south}_{east_west}"


def tile_url(layer: str, tile: str) -> str:
    """Return the public URL of one Hansen tile layer."""
    return f"{HANSEN_ROOT}/Hansen_{HANSEN_VERSION}_{layer}_{tile}.tif"


def fetch_layers(
    bbox: list[float] | tuple[float, float, float, float],
    reference_raster: Path,
    event_dir: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Crop tree cover and loss year onto the event analysis grid.

    Returns ``(treecover2000, lossyear, tile_name)``. Cached files are reused, so
    repeated runs do not re-download tiles.
    """
    tile = tile_name(bbox)
    arrays: list[np.ndarray] = []
    for layer in LAYERS:
        destination = Path(event_dir) / f"hansen_{layer}.tif"
        if not destination.exists():
            warp_to_reference(
                tile_url(layer, tile),
                destination,
                reference_raster,
                resampling="near",
                source_nodata=None,
                destination_nodata=255,
            )
        with rasterio.open(destination) as dataset:
            arrays.append(dataset.read(1))
    return arrays[0], arrays[1], tile


def prefire_loss_mask(loss_year: np.ndarray, fire_year: int) -> np.ndarray:
    """Pixels with stand-replacing loss strictly before the fire year.

    ``lossyear`` code ``n`` means calendar year ``2000 + n``; code 0 means no
    loss. The comparison is strict so that loss in the fire year, which can be
    the study fire itself, is not treated as pre-fire deforestation.
    """
    return (loss_year > 0) & ((2000 + loss_year) < fire_year)


def forest_mask(
    tree_cover: np.ndarray, loss_year: np.ndarray, fire_year: int, threshold: int = 30
) -> np.ndarray:
    """Forest population: sufficient canopy in 2000 and no pre-fire loss."""
    return (tree_cover >= threshold) & ~prefire_loss_mask(loss_year, fire_year)
