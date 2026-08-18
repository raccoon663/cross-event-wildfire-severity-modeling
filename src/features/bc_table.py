"""Leakage-controlled pre-fire pixel table for the British Columbia experiment.

Design rule
-----------
A predictor may only describe the landscape *before* the fire. Post-fire
reflectance, post-fire indices and every difference index (dNBR, dNDVI, dNDMI)
are forbidden: including any of them would turn a prediction task into a mapping
task and inflate the score by construction.

:data:`FORBIDDEN_PREDICTORS` is asserted against the configured feature list in
the modelling stage, so the constraint is enforced in code rather than in prose.

Analysis population per event
-----------------------------
inside the exact NBAC perimeter
    Rasterised from the perimeter geometry, not from the label footprint.
forest in 2000 without pre-fire loss
    Hansen tree cover at or above the threshold, minus stand-replacing loss in
    any year strictly before the fire year.
valid pre-fire spectra
    All seven composite bands finite and the three indices inside ``[-1, 1]``.
valid terrain
    Finite elevation, slope and aspect.
valid target
    Finite CanLaBS dNBR inside ``[-2, 2]``.

The complete valid population is written; no test-side subsampling is applied,
so a held-out fire is always scored over all of its usable pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..data.hansen import prefire_loss_mask
from ..data.landsat import PREFIRE_FEATURE_NAMES
from ..utils.raster import read_float_bands, require_same_grid
from .terrain import terrain_from_dem

__all__ = [
    "FORBIDDEN_PREDICTORS",
    "TABLE_COLUMNS",
    "build_event_frame",
    "build_feature_table",
]

FORBIDDEN_PREDICTORS = ("post_NDVI", "post_NBR", "post_NDMI", "dNDVI", "dNDMI", "dNBR_feature")
TABLE_COLUMNS = [
    "fire_id", "year", "province", "pixel_id", "row", "col", "x", "y",
    "spatial_block_id", *PREFIRE_FEATURE_NAMES,
    "tree_cover2000", "forest_persistence_years_since_2000",
    "elevation", "slope", "aspect", "aspect_sin", "aspect_cos", "target_dNBR",
]


def build_event_frame(
    event_dir: Path,
    fire_id: str,
    year: int,
    province: str = "BC",
    tree_cover_threshold: int = 30,
    block_size_m: int = 300,
) -> tuple[pd.DataFrame, dict]:
    """Build the pixel table and audit record for one event.

    Every input raster is checked against the pre-fire Landsat grid before use.
    Spatial block identifiers are assigned here, before any model is fitted, so
    the block-level evaluation cannot be tuned after seeing results.
    """
    event_dir = Path(event_dir)
    landsat, reference_profile, transform, descriptions = read_float_bands(
        event_dir / "prefire_landsat_features.tif"
    )
    if descriptions != PREFIRE_FEATURE_NAMES:
        raise ValueError(f"Unexpected Landsat bands for {fire_id}: {descriptions}")

    target_raw, target_profile, _, _ = read_float_bands(event_dir / "canlabs_dnbr.tif")
    perimeter_raw, perimeter_profile, _, _ = read_float_bands(event_dir / "exact_perimeter_mask.tif")
    tree_raw, tree_profile, _, _ = read_float_bands(event_dir / "hansen_treecover2000.tif")
    loss_raw, loss_profile, _, _ = read_float_bands(event_dir / "hansen_lossyear.tif")
    for profile, name in [
        (target_profile, "canlabs_dnbr.tif"),
        (perimeter_profile, "exact_perimeter_mask.tif"),
        (tree_profile, "hansen_treecover2000.tif"),
        (loss_profile, "hansen_lossyear.tif"),
    ]:
        require_same_grid(reference_profile, profile, event_dir / name)
    elevation, slope, aspect, aspect_sin, aspect_cos = terrain_from_dem(
        event_dir / "copernicus_dem_30m.tif", reference_profile
    )

    target = target_raw[0] / 1000.0
    perimeter = perimeter_raw[0] == 1
    tree_cover, loss_year = tree_raw[0], loss_raw[0]
    loss_before_fire = prefire_loss_mask(loss_year, year)
    forest = (tree_cover >= tree_cover_threshold) & ~loss_before_fire
    spectral_valid = np.isfinite(landsat).all(axis=0)
    index_valid = (np.abs(landsat[4:]) <= 1).all(axis=0)
    terrain_valid = np.isfinite(elevation) & np.isfinite(slope) & np.isfinite(aspect)
    target_valid = np.isfinite(target) & (target >= -2) & (target <= 2)
    valid = perimeter & forest & spectral_valid & index_valid & terrain_valid & target_valid

    rows, cols = np.where(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    block_col = np.floor(xs / block_size_m).astype("int32")
    block_row = np.floor(ys / block_size_m).astype("int32")

    data: dict[str, object] = {
        "fire_id": fire_id, "year": year, "province": province,
        "pixel_id": [f"{fire_id}_{r}_{c}" for r, c in zip(rows, cols)],
        "row": rows.astype("int32"), "col": cols.astype("int32"),
        "x": xs, "y": ys,
        "spatial_block_id": [f"{fire_id}_{r}_{c}" for r, c in zip(block_row, block_col)],
    }
    for index, name in enumerate(PREFIRE_FEATURE_NAMES):
        data[name] = landsat[index, rows, cols]
    data.update({
        "tree_cover2000": tree_cover[rows, cols],
        "forest_persistence_years_since_2000": np.full(len(rows), year - 2000, dtype="int16"),
        "elevation": elevation[rows, cols], "slope": slope[rows, cols],
        "aspect": aspect[rows, cols], "aspect_sin": aspect_sin[rows, cols],
        "aspect_cos": aspect_cos[rows, cols], "target_dNBR": target[rows, cols],
    })
    frame = pd.DataFrame(data)

    forest_in_perimeter = int((perimeter & forest).sum())
    audit = {
        "fire_id": fire_id, "year": year, "rows_written": len(frame),
        "perimeter_pixels": int(perimeter.sum()),
        "forest_pixels": forest_in_perimeter,
        "excluded_prefire_loss_pixels": int((perimeter & loss_before_fire).sum()),
        "spectral_valid_fraction_in_forest":
            float((spectral_valid & perimeter & forest).sum() / max(1, forest_in_perimeter)),
        "terrain_valid_fraction_in_forest":
            float((terrain_valid & perimeter & forest).sum() / max(1, forest_in_perimeter)),
        "target_mean": float(frame["target_dNBR"].mean()),
        "target_std": float(frame["target_dNBR"].std()),
        "spatial_blocks": int(frame["spatial_block_id"].nunique()),
    }
    return frame, audit


def build_feature_table(
    catalog: pd.DataFrame,
    events_dir: Path,
    output_dir: Path,
    tree_cover_threshold: int = 30,
    block_size_m: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """Concatenate all selected events into the modelling table.

    Writes the table, a per-event audit (CSV and JSON) and a schema document that
    states the target definition, the predictor timing rule and the absence of
    forbidden predictors.
    """
    events_dir = Path(events_dir)
    output_dir = Path(output_dir)
    frames, audits = [], []
    for record in catalog.sort_values(["year", "fire_id"]).to_dict("records"):
        fire_id, year = str(record["fire_id"]), int(record["year"])
        frame, audit = build_event_frame(
            events_dir / fire_id, fire_id, year,
            province=str(record.get("province", "BC")),
            tree_cover_threshold=tree_cover_threshold,
            block_size_m=block_size_m,
        )
        frames.append(frame)
        audits.append(audit)
        if verbose:
            print(json.dumps(audit), flush=True)

    table = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_dir / "bc_prefire_feature_table.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_csv(output_dir / "bc_prefire_feature_audit.csv", index=False)
    (output_dir / "bc_prefire_feature_audit.json").write_text(
        json.dumps(audits, indent=2), encoding="utf-8"
    )
    schema = {
        "target": "CanLaBS v2 continuous dNBR (source integer divided by 1000)",
        "predictor_timing": "pre-fire Landsat plus pre-fire/static ancillary data only",
        "tree_cover_threshold": tree_cover_threshold,
        "block_size_m": block_size_m,
        "rows": len(table),
        "events": int(table["fire_id"].nunique()),
        "columns": {name: str(dtype) for name, dtype in table.dtypes.items()},
        "forbidden_predictors_absent": list(FORBIDDEN_PREDICTORS),
    }
    (output_dir / "bc_prefire_feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    return table
