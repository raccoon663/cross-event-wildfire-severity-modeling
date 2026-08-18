"""Copernicus GLO-30 DEM caching for the Canadian experiment.

The DEM is fetched from the Planetary Computer ``cop-dem-glo-30`` collection as a
signed ``/vsicurl`` mosaic and warped onto each event's CanLaBS grid. Because the
mosaic is remote, only the intersecting blocks are transferred.

An existing non-empty event DEM is never re-downloaded, which makes the stage
resumable across crashes. Failures are surfaced as exceptions and recorded by the
caller; the code never substitutes a different elevation source, because a
silently different DEM could change slope and aspect everywhere.
"""

from __future__ import annotations

from pathlib import Path

from ..data.nbac import geometry_bbox
from ..data.stac import search_items, sign_href
from ..gdal_tools import run_gdal
from .canlabs import CANADA_LCC_PROJ

__all__ = ["build_dem", "cache_event_dems"]

DEM_COLLECTION = "cop-dem-glo-30"


def build_dem(event_dir: Path, reference_raster: Path) -> tuple[Path, list[str]]:
    """Crop the Copernicus GLO-30 mosaic onto one event grid.

    Returns ``(dem_path, item_ids_used)``. When the DEM already exists it is
    returned with an empty item list, signalling a cache hit to the caller.
    """
    event_dir = Path(event_dir)
    output = event_dir / "copernicus_dem_30m.tif"
    if output.exists():
        return output, []

    import rasterio

    bbox = geometry_bbox(event_dir / "perimeter_wgs84.geojson")
    items = search_items(DEM_COLLECTION, bbox, limit=100)
    hrefs = [sign_href(item["assets"]["data"]["href"]) for item in items]
    if not hrefs:
        raise RuntimeError("No Copernicus DEM items found for this event")
    with rasterio.open(reference_raster) as reference:
        bounds = reference.bounds
        width = reference.width
        height = reference.height

    arguments = [
        "-overwrite", "-q",
        "-t_srs", CANADA_LCC_PROJ,
        "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
        "-ts", str(width), str(height),
        "-r", "bilinear",
        "-dstnodata", "-9999",
        *[f"/vsicurl/{href}" for href in hrefs],
        str(output),
    ]
    try:
        run_gdal("gdalwarp", arguments, timeout=180)
    except BaseException:
        if output.exists():
            output.unlink()
        raise
    return output, [item["id"] for item in items]


def cache_event_dems(catalog, events_dir: Path, state_path: Path) -> list[dict]:
    """Cache DEMs for every selected event, recording a JSON state after each.

    Resumable: completed events are skipped and previously cached item lists are
    preserved for cache-hit events.
    """
    events_dir = Path(events_dir)
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    import time

    prior: dict[str, dict] = {}
    if state_path.exists():
        prior = {row["fire_id"]: row for row in json.loads(state_path.read_text(encoding="utf-8"))}

    results: list[dict] = []
    for record in catalog.to_dict("records"):
        fire_id = str(record["fire_id"])
        event_dir = events_dir / fire_id
        dem = event_dir / "copernicus_dem_30m.tif"
        reference = event_dir / "canlabs_dnbr.tif"
        started = time.perf_counter()
        try:
            path, item_ids = build_dem(event_dir, reference)
            results.append({
                "fire_id": fire_id, "status": "pass",
                "seconds": round(time.perf_counter() - started, 3),
                "size_bytes": path.stat().st_size,
                "dem_path": str(path),
                "item_ids": item_ids or prior.get(fire_id, {}).get("item_ids", []),
                "cached_before_run": dem.exists() and (time.perf_counter() - started) < 2,
                "error": "",
            })
        except Exception as exc:  # recorded per event; other events still proceed
            results.append({
                "fire_id": fire_id, "status": "error",
                "seconds": round(time.perf_counter() - started, 3),
                "size_bytes": dem.stat().st_size if dem.exists() else 0,
                "dem_path": str(dem),
                "item_ids": [],
                "cached_before_run": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        state_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{fire_id} {results[-1]['status']}", flush=True)
    return results
