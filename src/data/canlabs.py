"""CanLaBS v2 national continuous dNBR (Natural Resources Canada).

CanLaBS v2 is a 1985-2024 national Landsat burn-severity mosaic distributed as
one Cloud-Optimized GeoTIFF of roughly 1.75 GB. Event windows are cut directly
from the remote file with ``gdalwarp -cutline``, so the national raster is never
downloaded.

The stored values are dNBR multiplied by 1000 with nodata ``-32768``; call
:func:`scale_dnbr` before any statistics.

Interpretation limit: CanLaBS dNBR is a *remotely sensed spectral response*, not
a field measurement of burn severity. It is used here as a continuous regression
target and is never converted into severity classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from ..gdal_tools import run_gdal

__all__ = [
    "CANLABS_URL",
    "NODATA",
    "DNBR_SCALE",
    "PIXEL_AREA_HA",
    "CANADA_LCC_PROJ",
    "crop_event_window",
    "scale_dnbr",
    "read_scaled_dnbr",
]

CANLABS_URL = (
    "/vsicurl/https://download-telecharger.services.geo.ca/pub/"
    "nrcan_rncan/Forest-fires_Incendie-de-foret/"
    "CanLaBS_v2-Burned_Severity-Severite_des_feux/"
    "CanLaBS_1985_2024_v20260121.tif"
)
NODATA = -32768
DNBR_SCALE = 1000.0
PIXEL_AREA_HA = 30.0 * 30.0 / 10_000.0
CANADA_LCC_PROJ = (
    "+proj=lcc +lat_0=0 +lon_0=-95 +lat_1=49 +lat_2=77 +datum=NAD83 +units=m +no_defs"
)


def crop_event_window(
    perimeter_shapefile: Path,
    fire_id: str,
    output_path: Path,
    timeout: int = 180,
) -> Path:
    """Cut one fire perimeter out of the national CanLaBS raster.

    The cutline is applied server-side by GDAL over ``/vsicurl``; only the
    intersecting blocks are transferred. An existing output is reused so the
    screening stage is resumable.
    """
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        "-overwrite",
        "-q",
        "-cutline",
        str(perimeter_shapefile),
        "-cwhere",
        f"GID='{fire_id}'",
        "-crop_to_cutline",
        "-dstnodata",
        str(NODATA),
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILED=YES",
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-wo",
        "CUTLINE_ALL_TOUCHED=TRUE",
        CANLABS_URL,
        str(output_path),
    ]
    try:
        run_gdal("gdalwarp", arguments, timeout=timeout)
    except BaseException:
        if output_path.exists():
            output_path.unlink()
        raise
    return output_path


def scale_dnbr(values: np.ndarray) -> np.ndarray:
    """Convert stored integers to dNBR units."""
    return values.astype("float32") / DNBR_SCALE


def read_scaled_dnbr(path: Path) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """Read an event CanLaBS window as float dNBR with nodata as ``NaN``."""
    with rasterio.open(path) as dataset:
        raw = dataset.read(1)
        nodata = dataset.nodata
        transform = dataset.transform
        profile = dataset.profile.copy()
    values = scale_dnbr(raw)
    if nodata is not None:
        values[raw == nodata] = np.nan
    return values, transform, profile
