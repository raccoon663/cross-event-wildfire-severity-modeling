"""Landsat Collection 2 Level-2 surface reflectance access.

Two reading modes are needed because the two experiments have different
requirements:

:func:`read_scene`
    Single named scene warped onto an explicit analysis grid. Used for the
    United States experiment, where the exact MTBS pre- and post-fire scene
    identifiers are pinned in the configuration so the model sees the same
    imagery the reference product was derived from.

:func:`prefire_median_composite`
    Cloud-screened median composite of up to ``max_scenes`` pre-season scenes,
    written to disk and reused. Used for the Canadian experiment, where no
    post-fire information may be read at all and single-date pre-fire imagery is
    too often cloud-contaminated.

Scaling follows the Collection 2 Level-2 specification: surface reflectance is
``DN * 0.0000275 - 0.2``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from ..features.spectral_indices import normalized_difference
from ..preprocessing.qa_mask import clear_mask, clear_pixel_mask
from ..preprocessing.reproject import warp_asset_to_grid, warp_to_reference
from .stac import landsat_item, sign_href

__all__ = [
    "USA_BANDS",
    "CANADA_BANDS",
    "PREFIRE_FEATURE_NAMES",
    "REFLECTANCE_SCALE",
    "REFLECTANCE_OFFSET",
    "read_scene",
    "prefire_median_composite",
]

USA_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
CANADA_BANDS = ["red", "nir08", "swir16", "swir22"]
PREFIRE_FEATURE_NAMES = [
    "pre_red", "pre_nir", "pre_swir1", "pre_swir2",
    "pre_NDVI", "pre_NBR", "pre_NDMI",
]
REFLECTANCE_SCALE = 0.0000275
REFLECTANCE_OFFSET = -0.2


def read_scene(
    item: dict[str, Any],
    transform,
    width: int,
    height: int,
    crs: str,
    resolution_m: float,
    bands: list[str] | None = None,
) -> np.ndarray:
    """Read one Landsat scene as a masked surface-reflectance cube.

    Bilinear resampling is used for reflectance bands and nearest neighbour for
    the two quality bands, so QA bit patterns are never interpolated.

    Returns an array of shape ``(len(bands), height, width)`` where masked or
    physically implausible values (outside ``[0, 1]``) are ``NaN``.
    """
    bands = bands or USA_BANDS
    quality = warp_asset_to_grid(
        item["assets"]["qa_pixel"]["href"], transform, width, height, crs,
        resolution_m, nearest=True,
    ).astype("uint16")
    saturation = warp_asset_to_grid(
        item["assets"]["qa_radsat"]["href"], transform, width, height, crs,
        resolution_m, nearest=True,
    ).astype("uint16")
    clear = clear_mask(quality, saturation)
    arrays = []
    for band in bands:
        raw = warp_asset_to_grid(
            item["assets"][band]["href"], transform, width, height, crs, resolution_m
        ).astype("float32")
        reflectance = raw * REFLECTANCE_SCALE + REFLECTANCE_OFFSET
        reflectance[(raw == 0) | (~clear) | (reflectance < 0) | (reflectance > 1)] = np.nan
        arrays.append(reflectance)
    return np.stack(arrays)


def prefire_median_composite(
    candidate_item_ids: list[str],
    reference_raster: Path,
    output_dir: Path,
    max_scenes: int = 3,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the leakage-controlled pre-fire composite for one Canadian event.

    Only pre-season scenes from the year before the fire are accepted by the
    caller; this function never requests a post-fire acquisition. The seven
    output bands are ``pre_red, pre_nir, pre_swir1, pre_swir2, pre_NDVI,
    pre_NBR, pre_NDMI``.

    Reflectance is accepted in the wider range ``[-0.1, 1.2]`` here than in
    :func:`read_scene`, because a per-pixel median over several dates tolerates
    mild residual atmospheric effects better than a single date does.

    Returns ``(features, band_names, scene_ids_used)``. If the composite already
    exists it is read back from disk unchanged so the pipeline is resumable.
    """
    reference_raster = Path(reference_raster)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(candidate_item_ids)[:max_scenes]
    path = output_dir / "prefire_landsat_features.tif"
    if path.exists():
        with rasterio.open(path) as dataset:
            values = dataset.read().astype("float32")
            values[values == dataset.nodata] = np.nan
            return values, list(dataset.descriptions), selected

    cubes: list[np.ndarray] = []
    used: list[str] = []
    with tempfile.TemporaryDirectory(prefix="prefire_composite_") as temporary:
        workspace = Path(temporary)
        for scene_id in selected:
            item = landsat_item(scene_id)
            quality_path = workspace / f"{scene_id}_qa.tif"
            warp_to_reference(
                sign_href(item["assets"]["qa_pixel"]["href"]), quality_path,
                reference_raster, resampling="near", source_nodata=0, destination_nodata=0,
            )
            with rasterio.open(quality_path) as dataset:
                quality = dataset.read(1)
            clear = clear_pixel_mask(quality)
            arrays = []
            for band in CANADA_BANDS:
                band_path = workspace / f"{scene_id}_{band}.tif"
                warp_to_reference(
                    sign_href(item["assets"][band]["href"]), band_path,
                    reference_raster, resampling="bilinear",
                    source_nodata=0, destination_nodata=0,
                )
                with rasterio.open(band_path) as dataset:
                    raw = dataset.read(1).astype("float32")
                reflectance = raw * REFLECTANCE_SCALE + REFLECTANCE_OFFSET
                reflectance[(raw == 0) | (~clear) | (reflectance < -0.1) | (reflectance > 1.2)] = np.nan
                arrays.append(reflectance)
            cubes.append(np.stack(arrays))
            used.append(scene_id)

    if not cubes:
        raise RuntimeError("No pre-fire scenes available for this event")
    with np.errstate(all="ignore"):
        composite = np.nanmedian(np.stack(cubes), axis=0)
    red, near_infrared, shortwave1, shortwave2 = composite
    features = np.stack([
        red, near_infrared, shortwave1, shortwave2,
        normalized_difference(near_infrared, red),
        normalized_difference(near_infrared, shortwave2),
        normalized_difference(near_infrared, shortwave1),
    ])
    with rasterio.open(reference_raster) as reference:
        profile = reference.profile.copy()
    profile.update(
        count=len(PREFIRE_FEATURE_NAMES), dtype="float32", nodata=-9999,
        compress="deflate", tiled=True,
    )
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.where(np.isfinite(features), features, -9999).astype("float32"))
        destination.descriptions = PREFIRE_FEATURE_NAMES
    return features, PREFIRE_FEATURE_NAMES, used
