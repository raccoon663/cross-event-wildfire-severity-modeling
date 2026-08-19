"""Windowed reprojection of remote assets with the GDAL command line.

Two entry points cover the two grid conventions in the project:

:func:`warp_asset_to_grid`
    Warp a Planetary Computer asset onto an explicit ``(transform, width,
    height, crs, resolution)`` grid. The asset URL is signed immediately before
    the call because SAS tokens are short-lived.

:func:`warp_to_reference`
    Warp any URL onto the grid of an existing reference raster. Used for the
    Canadian events, where the CanLaBS event window defines the grid.

Transient HTTP failures (403 after token expiry, 429 rate limiting) and timeouts
are retried with exponential backoff; a partially written output is deleted so a
truncated raster is never mistaken for a cached success.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from ..data.stac import sign_href
from ..gdal_tools import gdal_binary

__all__ = ["warp_asset_to_grid", "warp_to_reference"]


def _run(command: list[str], timeout: int | None) -> tuple[int, str]:
    process = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )
    return process.returncode, (process.stderr or process.stdout or "")


def warp_asset_to_grid(
    href: str,
    transform: Affine,
    width: int,
    height: int,
    crs: str,
    resolution: float,
    nearest: bool = False,
    sign: bool = True,
    timeout: int | None = None,
) -> np.ndarray:
    """Crop and resample one remote asset onto an explicit analysis grid.

    Reflectance bands use bilinear resampling; categorical bands (``QA_PIXEL``,
    ``QA_RADSAT``, MTBS severity) must pass ``nearest=True`` so class codes and
    bit patterns are preserved.
    """
    left, top = transform.c, transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e
    url = sign_href(href) if sign else href
    with tempfile.TemporaryDirectory(prefix="warp_asset_") as temporary:
        output = Path(temporary) / "crop.tif"
        command = [
            str(gdal_binary("gdalwarp")), "-overwrite", "-q",
            "-t_srs", crs,
            "-te", str(left), str(bottom), str(right), str(top),
            "-tr", str(resolution), str(resolution),
            "-r", "near" if nearest else "bilinear",
            "-srcnodata", "0", "-dstnodata", "0",
            f"/vsicurl/{url}", str(output),
        ]
        code, message = _run(command, timeout)
        if code:
            raise RuntimeError(message.strip() or "gdalwarp failed")
        with rasterio.open(output) as dataset:
            return dataset.read(1)


def warp_to_reference(
    source: str,
    destination: Path,
    reference_raster: Path,
    target_crs: str | None = None,
    resampling: str = "near",
    source_nodata: float | None = None,
    destination_nodata: float = 0,
    attempts: int = 3,
    timeout: int = 180,
    retry_delay: int = 30,
) -> Path:
    """Warp a remote source onto the exact grid of a reference raster.

    ``-ts width height`` is used rather than ``-tr`` so the output shares the
    reference shape exactly, which the feature-table builder later asserts.
    """
    from .. data.canlabs import CANADA_LCC_PROJ

    destination = Path(destination)
    reference_raster = Path(reference_raster)
    with rasterio.open(reference_raster) as reference:
        bounds = reference.bounds
        width = reference.width
        height = reference.height
    command = [
        str(gdal_binary("gdalwarp")), "-overwrite", "-q",
        "-t_srs", target_crs or CANADA_LCC_PROJ,
        "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
        "-ts", str(width), str(height),
        "-r", resampling,
    ]
    if source_nodata is not None:
        command += ["-srcnodata", str(source_nodata)]
    command += ["-dstnodata", str(destination_nodata),
                f"/vsicurl/{source}", str(destination)]

    last_error = ""
    for attempt in range(attempts):
        try:
            code, message = _run(command, timeout)
            if code == 0:
                return destination
            last_error = message[-2000:]
            retryable = ("403" in last_error) or ("429" in last_error)
        except subprocess.TimeoutExpired:
            last_error = f"GDAL remote asset crop timed out after {timeout} seconds"
            retryable = True
        if not retryable or attempt == attempts - 1:
            break
        time.sleep(retry_delay * (2 ** attempt))
    if destination.exists():
        destination.unlink()
    raise RuntimeError(last_error)
