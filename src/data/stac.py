"""Microsoft Planetary Computer STAC access.

All Landsat Collection 2 Level-2, MTBS and Copernicus DEM assets are read as
signed ``/vsicurl`` windows. Nothing is mirrored locally, which is why the two
experiments can run from a laptop.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

__all__ = [
    "STAC_ROOT",
    "SIGN_ROOT",
    "LANDSAT_COLLECTION",
    "get_json",
    "sign_href",
    "landsat_item",
    "stac_item",
    "search_items",
]

STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_ROOT = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
LANDSAT_COLLECTION = "landsat-c2-l2"


def get_json(url: str, timeout: int = 90) -> dict[str, Any]:
    """Fetch and decode a JSON document."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def sign_href(href: str) -> str:
    """Return a short-lived signed URL for a Planetary Computer asset."""
    query = urllib.parse.urlencode({"href": href})
    return get_json(f"{SIGN_ROOT}?{query}")["href"]


def stac_item(collection: str, item_id: str) -> dict[str, Any]:
    """Fetch one STAC item by collection and identifier."""
    return get_json(f"{STAC_ROOT}/collections/{collection}/items/{item_id}")


def landsat_item(item_id: str) -> dict[str, Any]:
    """Fetch one Landsat Collection 2 Level-2 STAC item."""
    return stac_item(LANDSAT_COLLECTION, item_id)


def search_items(
    collections: str,
    bbox: list[float] | tuple[float, float, float, float],
    datetime_range: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Run a STAC search and return the feature list.

    Parameters
    ----------
    collections
        Collection identifier, e.g. ``landsat-c2-l2`` or ``cop-dem-glo-30``.
    bbox
        Longitude/latitude bounding box in WGS84.
    datetime_range
        Optional ``start/end`` interval, e.g. ``2017-06-01/2017-09-30``.
    limit
        Maximum number of items requested from the API.
    """
    query: dict[str, str] = {
        "collections": collections,
        "bbox": ",".join(str(value) for value in bbox),
        "limit": str(limit),
    }
    if datetime_range:
        query["datetime"] = datetime_range
    url = f"{STAC_ROOT}/search?{urllib.parse.urlencode(query)}"
    return get_json(url).get("features", [])
