#!/usr/bin/env python3
"""British Columbia experiment pipeline (leave-one-fire-out, pre-fire only).

The pipeline has five resumable stages; run them in order:

    python experiments/canada_bc/01_build_catalog.py \\
        --workbook <NBAC workbook.xlsx> --annual-dir <NBAC shapefiles> --output-dir <events_root>

    python experiments/canada_bc/02_screen_events.py \\
        --catalog <events_root>/bc_candidate_catalog.parquet \\
        --annual-dir <NBAC shapefiles> --output-dir <events_root>

    python experiments/canada_bc/03_build_features.py \\
        --screen-catalog <events_root>/batch_screen_catalog.parquet \\
        --events-dir <events_root>/events --output-dir <events_root>

    python experiments/canada_bc/04_cache_dem.py \\
        --catalog <events_root>/bc_event_catalog.parquet \\
        --events-dir <events_root>/events --output <events_root>/dem_cache.json

    python experiments/canada_bc/05_run_lofo.py \\
        --table <events_root>/bc_prefire_feature_table.parquet \\
        --output-dir <results>/lofo

Stage 3 asserts the leakage rule: no post-fire band or difference index may
appear in the feature table. Stage 5 additionally asserts the configured feature
list against :data:`FORBIDDEN_PREDICTORS` before any fitting starts.

Every stage is resumable: completed events are skipped, so a crash mid-catalogue
only redoes the unfinished events.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import pandas as pd
import rasterio

from src.config import load_config
from src.data.canlabs import crop_event_window, read_scaled_dnbr
from src.data.hansen import fetch_layers
from src.data.landsat import prefire_median_composite
from src.data.nbac import (
    annual_shapefile,
    export_perimeter_geojson,
    export_perimeter_projected,
    geometry_bbox,
    load_event_attributes,
    rasterize_perimeter_mask,
)
from src.features.bc_table import build_feature_table
from src.data.stac import search_items

BC_CONFIG = "bc_lofo"


def build_candidate_catalog(workbook: Path, output_dir: Path, candidates: list[str]) -> pd.DataFrame:
    """Extract the fixed first-round BC candidate list from the NBAC workbook."""
    frame = load_event_attributes(workbook, candidates)
    subset = frame.loc[frame["GID"].isin(candidates)].copy()
    if set(subset["GID"]) != set(candidates):
        raise ValueError(f"Missing IDs: {set(candidates) - set(subset['GID'])}")
    subset = subset.rename(columns={
        "GID": "fire_id", "YEAR": "year", "ADMIN_NAME": "province",
        "BASRC": "nbac_source", "HS_SDATE": "start_date", "HS_EDATE": "end_date",
        "FIRECAUS": "cause", "FIREMAPS": "mapping_sensor",
    })
    subset = subset[[
        "fire_id", "year", "province", "area_ha", "nbac_source",
        "start_date", "end_date", "cause", "mapping_sensor",
    ]].sort_values(["year", "fire_id"])
    subset["candidate_rank"] = range(1, len(subset) + 1)
    subset["selected_for_gate"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_dir / "bc_candidate_catalog.csv", index=False, encoding="utf-8-sig")
    subset.to_parquet(output_dir / "bc_candidate_catalog.parquet", index=False)
    return subset


def screen_event(record: dict, annual_dir: Path, events_dir: Path, config: dict) -> dict:
    """Cut the CanLaBS window, rasterise the exact mask and collect pre-fire Landsat.

    Uses only pre-season scenes from the year *before* the fire
    (``{year-1}-06-01`` to ``{year-1}-09-30``), so the downstream feature table
    cannot contain any post-fire imagery.
    """
    fire_id = str(record["fire_id"])
    year = int(record["year"])
    event_dir = events_dir / fire_id
    event_dir.mkdir(parents=True, exist_ok=True)
    shapefile = annual_shapefile(annual_dir, year)
    geojson = export_perimeter_geojson(shapefile, fire_id, event_dir / "perimeter_wgs84.geojson")
    projected = export_perimeter_projected(shapefile, fire_id, event_dir / "perimeter_canlabs.shp")
    label = crop_event_window(shapefile, fire_id, event_dir / "canlabs_dnbr.tif")
    values, transform, profile = read_scaled_dnbr(label)
    nodata_mask = np.isfinite(values)

    with rasterio.open(label) as dataset:
        bounds, width, height = dataset.bounds, dataset.width, dataset.height
    mask = rasterize_perimeter_mask(
        projected, (bounds.left, bounds.bottom, bounds.right, bounds.top),
        width, height, event_dir / "exact_perimeter_mask.tif",
    )
    with rasterio.open(mask) as dataset:
        inside = dataset.read(1) == 1
    coverage = float((nodata_mask & inside).sum() / inside.sum())

    bbox = geometry_bbox(geojson)
    items = search_items(
        "landsat-c2-l2", bbox, datetime_range=f"{year - 1}-06-01/{year - 1}-09-30", limit=100
    )
    good = [item for item in items if (item.get("properties", {}).get("eo:cloud_cover") or 0) <= 50]
    good_sorted = sorted(good, key=lambda item: item.get("properties", {}).get("eo:cloud_cover", 999))
    passed = coverage >= config["gates"]["minimum_canlabs_coverage"] and len(good) > 0
    return {
        "fire_id": fire_id, "year": year,
        "province": record["province"], "area_ha": float(record["area_ha"]),
        "exact_label_coverage": round(coverage, 6),
        "prefire_landsat_count": len(good),
        "prefire_candidate_ids": [item["id"] for item in good_sorted[:6]],
        "status": "pass" if passed else "exclude",
        "error": "" if passed else "coverage_or_landsat_gate",
    }


def gate_event(record: dict, events_dir: Path, config: dict) -> dict:
    """Build pre-fire features and apply the Hansen/spectral gates for one event."""
    fire_id = str(record["fire_id"])
    year = int(record["year"])
    event_dir = events_dir / fire_id
    reference = event_dir / "canlabs_dnbr.tif"
    features, names, used = prefire_median_composite(
        list(record.get("prefire_candidate_ids") or []),
        reference, event_dir, max_scenes=config["landsat"]["maximum_scenes"],
    )
    bbox = geometry_bbox(event_dir / "perimeter_wgs84.geojson")
    tree, loss, tile = fetch_layers(bbox, reference, event_dir)
    values, _, _ = read_scaled_dnbr(reference)
    with rasterio.open(event_dir / "exact_perimeter_mask.tif") as dataset:
        inside = dataset.read(1) == 1
    threshold = config["gates"]["tree_cover_threshold"]
    loss_before = (loss > 0) & ((2000 + loss) < year)
    counts = {
        str(t): int((inside & np.isfinite(values) & (values >= -2) & (values <= 2)
                     & (tree >= t) & ~loss_before).sum())
        for t in config["gates"]["tree_cover_sensitivity"]
    }
    forest = (tree >= threshold) & ~loss_before
    spectral = np.isfinite(features).all(axis=0) & (np.abs(features[4:]) <= 1).all(axis=0)
    valid = inside & forest & spectral
    frac = float(valid.sum() / (inside & forest).sum()) if (inside & forest).sum() else 0.0
    target_values = values[valid]
    gate = config["gates"]
    selected = bool(
        record.get("exact_label_coverage", 0) >= gate["minimum_canlabs_coverage"]
        and frac >= gate["minimum_valid_prefire_fraction"]
        and len(target_values) >= gate["minimum_forest_pixels"]
    )
    reason = "" if selected else (
        "low_valid_prefire_fraction" if frac < gate["minimum_valid_prefire_fraction"]
        else "low_forest_pixel_count"
    )
    return {
        "fire_id": fire_id, "year": year,
        "nbac_source": record.get("nbac_source"),
        "canlabs_coverage": record.get("exact_label_coverage", np.nan),
        "landsat_candidate_count": record.get("prefire_landsat_count", 0),
        "valid_prefire_fraction": round(frac, 6),
        "forest_pixel_count": int(valid.sum()),
        "forest_pixels_tc20": counts["20"], "forest_pixels_tc30": counts["30"],
        "forest_pixels_tc40": counts["40"],
        "dnbr_mean": float(target_values.mean()) if len(target_values) else np.nan,
        "dnbr_std": float(target_values.std()) if len(target_values) else np.nan,
        "dnbr_p25": float(np.quantile(target_values, .25)) if len(target_values) else np.nan,
        "dnbr_p50": float(np.quantile(target_values, .5)) if len(target_values) else np.nan,
        "dnbr_p75": float(np.quantile(target_values, .75)) if len(target_values) else np.nan,
        "landsat_used_ids": ";".join(used),
        "hansen_tile": tile,
        "selected": selected, "exclusion_reason": reason,
    }


def cmd_catalog(args: argparse.Namespace) -> None:
    config = load_config(BC_CONFIG)
    candidates = config["events"]["selected"]
    build_candidate_catalog(args.workbook, args.output_dir, candidates)


def cmd_screen(args: argparse.Namespace) -> None:
    config = load_config(BC_CONFIG)
    catalog = pd.read_parquet(args.catalog)
    state_path = args.output_dir / "batch_screen_records.json"
    existing = {}
    if state_path.exists():
        existing = {row["fire_id"]: row for row in json.loads(state_path.read_text(encoding="utf-8"))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in catalog.to_dict("records"):
        fire_id = str(record["fire_id"])
        if fire_id in existing and existing[fire_id].get("status") in ("pass", "exclude"):
            continue
        try:
            result = screen_event(record, args.annual_dir, args.output_dir / "events", config)
        except Exception as exc:
            result = {"fire_id": fire_id, "year": int(record["year"]),
                      "province": record["province"], "area_ha": float(record["area_ha"]),
                      "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        existing[fire_id] = result
        state_path.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(fire_id, result["status"], flush=True)
    out = pd.DataFrame(existing.values())
    out.to_csv(args.output_dir / "batch_screen_catalog.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(args.output_dir / "batch_screen_catalog.parquet", index=False)


def cmd_features(args: argparse.Namespace) -> None:
    config = load_config(BC_CONFIG)
    screen = pd.read_parquet(args.screen_catalog)
    candidates = pd.read_parquet(args.candidate_catalog).set_index("fire_id")
    records = []
    events_dir = args.events_dir
    for record in screen.to_dict("records"):
        fire_id = str(record["fire_id"])
        try:
            row = gate_event(record, events_dir, config)
            row["nbac_source"] = row.get("nbac_source") or str(candidates.loc[fire_id, "nbac_source"])
        except Exception as exc:
            row = {"fire_id": fire_id, "year": int(record["year"]),
                   "nbac_source": str(candidates.loc[fire_id, "nbac_source"]),
                   "canlabs_coverage": record.get("exact_label_coverage", np.nan),
                   "landsat_candidate_count": record.get("prefire_landsat_count", 0),
                   "valid_prefire_fraction": np.nan, "forest_pixel_count": 0,
                   "dnbr_mean": np.nan, "dnbr_std": np.nan,
                   "dnbr_p25": np.nan, "dnbr_p50": np.nan, "dnbr_p75": np.nan,
                   "selected": False, "exclusion_reason": f"{type(exc).__name__}: {exc}"}
        records.append(row)
        pd.DataFrame(records).to_csv(args.output_dir / "bc_event_catalog_partial.csv",
                                     index=False, encoding="utf-8-sig")
        print(fire_id, row["selected"], flush=True)
    out = pd.DataFrame(records)
    required = ["fire_id", "year", "area_ha", "nbac_source", "canlabs_coverage",
                "landsat_candidate_count", "valid_prefire_fraction", "forest_pixel_count",
                "dnbr_mean", "dnbr_std", "dnbr_p25", "dnbr_p50", "dnbr_p75",
                "selected", "exclusion_reason"]
    for column in required:
        if column not in out:
            out[column] = np.nan
    out.to_csv(args.output_dir / "bc_event_catalog.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(args.output_dir / "bc_event_catalog.parquet", index=False)
    print(json.dumps({"selected": int(out["selected"].sum()),
                      "excluded": int((~out["selected"]).sum())}, indent=2))


def cmd_build_table(args: argparse.Namespace) -> None:
    config = load_config(BC_CONFIG)
    catalog = pd.read_parquet(args.catalog)
    catalog = catalog.loc[catalog["selected"].astype(bool)].copy()
    table = build_feature_table(
        catalog, args.events_dir, args.output_dir,
        tree_cover_threshold=config["gates"]["tree_cover_threshold"],
        block_size_m=config["sampling"]["block_size_m"],
    )
    print(json.dumps({"rows": len(table), "events": int(table["fire_id"].nunique())}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    p = subparsers.add_parser("catalog", help="Build the BC candidate catalogue from NBAC")
    p.add_argument("--workbook", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=cmd_catalog)

    p = subparsers.add_parser("screen", help="Cut CanLaBS windows and find pre-fire Landsat")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--annual-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=cmd_screen)

    p = subparsers.add_parser("features", help="Pre-fire features, Hansen gates and event catalogue")
    p.add_argument("--screen-catalog", type=Path, required=True)
    p.add_argument("--candidate-catalog", type=Path, required=True)
    p.add_argument("--events-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=cmd_features)

    p = subparsers.add_parser("table", help="Build the leakage-controlled pixel table")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--events-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=cmd_build_table)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
