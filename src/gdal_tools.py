"""Locate GDAL command-line utilities without hard-coding an install path.

Both study regions crop remote Cloud-Optimized GeoTIFFs with the GDAL command
line rather than a Python binding, because ``/vsicurl`` windowed reads keep the
national CanLaBS raster (1.75 GB) and the Landsat scene archive off local disk.

Resolution order for a utility such as ``gdalwarp``:

1. tool-specific environment variable, e.g. ``GDALWARP``
2. ``GDAL_BIN_DIR`` environment variable holding the directory of the utilities
3. the executable found on ``PATH``

No absolute path is embedded in the source tree, so the pipeline runs against a
conda ``gdal`` package, a system GDAL, an OSGeo4W/QGIS install or a container
image without editing any file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["gdal_binary", "gdal_version", "run_gdal", "GdalNotFoundError"]

_WINDOWS_SUFFIX = ".exe" if os.name == "nt" else ""


class GdalNotFoundError(RuntimeError):
    """Raised when a required GDAL utility cannot be located."""


def _candidate_paths(tool: str) -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get(tool.upper().replace("-", "_"))
    if explicit:
        candidates.append(Path(explicit))
    bin_dir = os.environ.get("GDAL_BIN_DIR")
    if bin_dir:
        candidates.append(Path(bin_dir) / f"{tool}{_WINDOWS_SUFFIX}")
    found = shutil.which(tool)
    if found:
        candidates.append(Path(found))
    return candidates


@lru_cache(maxsize=None)
def gdal_binary(tool: str = "gdalwarp") -> Path:
    """Return the executable path for a GDAL utility.

    Parameters
    ----------
    tool
        Utility name without extension, e.g. ``gdalwarp``, ``gdal_rasterize``
        or ``ogr2ogr``.

    Raises
    ------
    GdalNotFoundError
        If the utility is not on ``PATH`` and no environment variable points at
        it. The message lists the exact variables to set.
    """
    for candidate in _candidate_paths(tool):
        if candidate.is_file():
            return candidate
    variable = tool.upper().replace("-", "_")
    raise GdalNotFoundError(
        f"GDAL utility '{tool}' not found. Set the {variable} environment variable to the "
        f"executable, set GDAL_BIN_DIR to the directory containing the GDAL utilities, or "
        f"install GDAL so that '{tool}' is on PATH."
    )


def gdal_version() -> str:
    """Return the version string reported by ``gdalwarp --version``."""
    result = subprocess.run(
        [str(gdal_binary("gdalwarp")), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return (result.stdout or result.stderr).strip()


def run_gdal(tool: str, arguments: list[str], timeout: int | None = None) -> None:
    """Run a GDAL utility and raise with trimmed stderr when it fails."""
    command = [str(gdal_binary(tool)), *arguments]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if process.returncode:
        message = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(message[-2000:] or f"{tool} failed with code {process.returncode}")
