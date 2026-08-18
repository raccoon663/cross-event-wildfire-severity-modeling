# Data sources and use notes

All products are accessed as remote Cloud-Optimized GeoTIFFs (COGs) through
`/vsicurl`, so raw scenes are never stored in this repository. Item and
collection identifiers that fix the data version are frozen in
`configs/*.yaml`; the source tables and per-event audit files under
`experiments/*/results/` record exactly which assets each run consumed.

## Landsat Collection 2 Level-2 surface reflectance

- Provider: United States Geological Survey (USGS)
- Access route: Microsoft Planetary Computer `landsat-c2-l2` STAC collection
- Use in project:
  - USA: the exact MTBS pre-/post-fire scenes named in `configs/camp_fire_*.yaml`
    and `configs/carr_fire.yaml`, QA-masked and scaled to surface reflectance.
  - Canada: up to three cloud-screened *pre-season* scenes from the year before
    each fire, merged by per-pixel median into `prefire_landsat_features.tif`.
- Scaling: Collection 2 Level-2 `DN * 0.0000275 - 0.2`; `QA_PIXEL` bits 0-5 and
  `QA_RADSAT` define the clear-mask.
- Redistribution: raw scenes are not included; only the scene identifiers are.
- Reference: https://www.usgs.gov/landsat-missions/landsat-collection-2

## MTBS — Monitoring Trends in Burn Severity (USA reference)

- Provider: USGS / USDA Forest Service
- Access route: Planetary Computer `mtbs` STAC collection for the thematic
  rasters; the MTBS ArcGIS FeatureServer for event polygons and operator-chosen
  dNBR thresholds.
- Use in project: four-class operational burn-severity reference for the Camp
  Fire within-event ablation and the Camp-to-Carr external transfer.
- Important limitation: MTBS severity is an *interpreted remote-sensing
  product*, not plot-level tree mortality. Every reported classification score
  is agreement with these thematic labels.
- Reference: https://www.mtbs.gov/

## NBAC — National Burned Area Composite (Canada events)

- Provider: Natural Resources Canada (NRCan), Canadian Forest Service
- Access route: the merged attribute workbook (`NBAC_merged_1972_to_2025` sheet)
  and the annual perimeter shapefiles.
- Use in project: event catalogue (GID, year, province, adjusted area, cause,
  dates, mapping sensor) and the perimeter geometry that defines each event's
  analysis window.
- Redistribution: neither the workbook nor the shapefiles are included; the
  selected 12 fire identifiers are listed in `configs/bc_lofo.yaml`.
- Reference: https://cwfis.cfs.nrcan.gc.ca/datasets/nbac/

## CanLaBS v2 — Canada-wide Landsat Burn Severity (Canada target)

- Provider: Natural Resources Canada
- Access route: national 1985-2024 mosaic COG
  (`CanLaBS_1985_2024_v20260121.tif`), cut per event with `gdalwarp -cutline`
  so the 1.75 GB national raster is never downloaded.
- Use in project: continuous dNBR target for the BC leave-one-fire-out
  regression (stored values divided by 1000).
- Important limitation: CanLaBS dNBR is a *spectral response*, not ground truth;
  it is used as a continuous regression target and never converted into severity
  classes.
- Reference: https://open.canada.ca/data/en/dataset/9f20b0d8-a002-4f6f-90ad-e4a1b3f5a013

## Hansen Global Forest Change v1.11

- Provider: University of Maryland
- Access route: public Google Storage tiles `Hansen_GFC-2023-v1.11_*`.
- Use in project: `treecover2000` defines the forest population;
  `lossyear` excludes pixels with stand-replacing loss strictly before the fire
  year. Loss in the fire year itself is retained because it can be the study
  fire.
- Reference: Hansen et al. (2013), Science 342:850-853.
  https://earthenginepartners.appspot.com/science-2013-global-forest/

## Copernicus GLO-30 Digital Elevation Model

- Provider: ESA / European Commission
- Access route: Planetary Computer `cop-dem-glo-30` STAC collection.
- Use in project: event-window elevation; slope and aspect are derived with
  `np.gradient` at 30 m resolution (aspect entered as sine/cosine).
- Reference: https://doi.org/10.5270/ESA-c5d3d65

## Microsoft Planetary Computer

- Use in project: anonymous STAC search, temporary SAS signing, and remote COG
  range access for Landsat, MTBS and Copernicus DEM.
- Reference: https://planetarycomputer.microsoft.com/docs
