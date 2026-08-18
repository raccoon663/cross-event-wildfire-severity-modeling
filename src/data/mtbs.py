"""MTBS event polygons and thematic burn-severity rasters (United States).

Two distinct MTBS products are used:

* the *event polygon* served by the USFS ArcGIS FeatureServer, which also
  carries the per-event operator-chosen dNBR thresholds
  (``LOW_THRESHOLD`` / ``MODERATE_THRESHOLD`` / ``HIGH_THRESHOLD``, integers
  scaled by 1000);
* the *thematic severity raster* published as a Planetary Computer STAC item,
  used as the classification reference.

The raster encodes 1 unburned/low, 2 low, 3 moderate, 4 high; codes 5 and 6
(increased greenness, non-mapping area) are excluded from every analysis.
"""

from __future__ import annotations

from typing import Any

import requests

from . import stac

__all__ = [
    "MTBS_FEATURESERVER",
    "SEVERITY_CLASSES",
    "event_feature",
    "event_dnbr_thresholds",
    "severity_item",
]

MTBS_FEATURESERVER = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
    "EDW_MTBS_v1/FeatureServer/0/query"
)
SEVERITY_CLASSES = (1, 2, 3, 4)


def event_feature(fire_id: str, timeout: int = 90) -> dict[str, Any]:
    """Return the single MTBS event GeoJSON feature for ``fire_id``.

    Raises
    ------
    RuntimeError
        If the query does not match exactly one event, which would make the
        analysis grid ambiguous.
    """
    params = {
        "f": "geojson",
        "where": f"FIRE_ID = '{fire_id}'",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
    }
    response = requests.get(MTBS_FEATURESERVER, params=params, timeout=timeout)
    response.raise_for_status()
    features = response.json().get("features", [])
    if len(features) != 1:
        raise RuntimeError(f"Expected one MTBS event for {fire_id}, found {len(features)}")
    return features[0]


def event_dnbr_thresholds(properties: dict[str, Any]) -> list[float]:
    """Convert the stored integer thresholds to dNBR units.

    MTBS stores the operator-selected low/moderate/high break points as
    integers scaled by 1000; the returned list is directly comparable with a
    computed dNBR image.
    """
    return [
        properties["LOW_THRESHOLD"] / 1000,
        properties["MODERATE_THRESHOLD"] / 1000,
        properties["HIGH_THRESHOLD"] / 1000,
    ]


def severity_item(fire_year: int, resolution_m: int = 30) -> dict[str, Any]:
    """Fetch the CONUS MTBS thematic severity STAC item for one fire year."""
    return stac.stac_item("mtbs", f"mtbs_severity_conus_{fire_year}_{resolution_m}m")
