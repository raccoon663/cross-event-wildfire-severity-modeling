"""National Burned Area Composite (NBAC) event attributes and perimeters.

NBAC supplies the Canadian event catalogue: identifier (``GID``), year, province
or territory, adjusted burned area, cause, agency start/end dates, the perimeter
mapping sensor and the source of the burned-area estimate. The annual shapefiles
supply the perimeter geometry used both for cutting CanLaBS windows and for
rasterising the exact within-perimeter mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..gdal_tools import run_gdal
from .canlabs import CANADA_LCC_PROJ

__all__ = [
    "NBAC_SHEET",
    "annual_shapefile",
    "load_event_attributes",
    "export_perimeter_geojson",
    "export_perimeter_projected",
    "rasterize_perimeter_mask",
    "geometry_bbox",
]

NBAC_SHEET = "NBAC_merged_1972_to_2025"


def annual_shapefile(annual_dir: Path, year: int, release: str = "20260513") -> Path:
    """Return the NBAC annual perimeter shapefile for one year.

    Raises
    ------
    FileNotFoundError
        If the expected shapefile is absent, so a missing download is never
        silently replaced by a different year.
    """
    path = Path(annual_dir) / str(year) / f"NBAC_{year}_{release}.shp"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_event_attributes(workbook: Path, fire_ids: list[str] | None = None) -> pd.DataFrame:
    """Read the merged NBAC attribute workbook.

    The published workbook carries two banner rows before the header, hence
    ``header=2``. ``area_ha`` prefers the adjusted area ``ADJ_HA`` and falls back
    to the raw polygon area ``POLY_HA``.
    """
    frame = pd.read_excel(workbook, sheet_name=NBAC_SHEET, header=2)
    frame["area_ha"] = frame["ADJ_HA"].fillna(frame["POLY_HA"])
    if fire_ids is None:
        return frame
    subset = frame.loc[frame["GID"].isin(fire_ids)].copy()
    missing = set(fire_ids) - set(subset["GID"])
    if missing:
        raise ValueError(f"Events missing from NBAC workbook: {sorted(missing)}")
    return subset


def export_perimeter_geojson(
    shapefile: Path, fire_id: str, output_path: Path, timeout: int = 60
) -> Path:
    """Extract one perimeter as WGS84 GeoJSON (used for STAC/Hansen bounding boxes)."""
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_gdal(
        "ogr2ogr",
        ["-f", "GeoJSON", "-t_srs", "EPSG:4326", "-where", f"GID='{fire_id}'",
         str(output_path), str(shapefile)],
        timeout=timeout,
    )
    return output_path


def export_perimeter_projected(
    shapefile: Path, fire_id: str, output_path: Path, timeout: int = 60
) -> Path:
    """Extract one perimeter reprojected to the CanLaBS Lambert conformal grid."""
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_gdal(
        "ogr2ogr",
        ["-t_srs", CANADA_LCC_PROJ, "-where", f"GID='{fire_id}'",
         str(output_path), str(shapefile)],
        timeout=timeout,
    )
    return output_path


def rasterize_perimeter_mask(
    perimeter_shapefile: Path,
    reference_bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    output_path: Path,
    timeout: int = 60,
) -> Path:
    """Burn the projected perimeter onto the CanLaBS event grid as a byte mask.

    An exact mask is required because the area-based screening ratio can exceed
    one when perimeter geometry and label rasterisation disagree.
    """
    if output_path.exists():
        return output_path
    left, bottom, right, top = reference_bounds
    run_gdal(
        "gdal_rasterize",
        ["-burn", "1", "-init", "0", "-ot", "Byte",
         "-te", str(left), str(bottom), str(right), str(top),
         "-ts", str(width), str(height),
         "-a_srs", CANADA_LCC_PROJ, "-co", "COMPRESS=DEFLATE",
         str(perimeter_shapefile), str(output_path)],
        timeout=timeout,
    )
    return output_path


def geometry_bbox(geojson_path: Path) -> list[float]:
    """Return ``[min_lon, min_lat, max_lon, max_lat]`` for a GeoJSON perimeter."""
    document = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    points: list[list[float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list) and len(node) >= 2 and all(
            isinstance(value, (int, float)) for value in node[:2]
        ):
            points.append(list(node[:2]))
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(document["features"][0]["geometry"]["coordinates"])
    return [
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    ]
