# Methodology

This document specifies the exact protocol behind every number in the README.
Anything that affects a reported score — scene identifiers, thresholds, block
sizes, seeds, hyper-parameters — lives either here or in `configs/*.yaml`, and
the two are kept in sync.

## 1. Overall design

Three experiments form a difficulty ladder. Each stage relaxes exactly one kind
of information the previous stage could use:

| Stage | Predictors | Validation unit | What is being tested |
|---|---|---|---|
| 1. Camp ablation | pre + post (retrospective mapping) | spatial blocks within one fire | spatial generalization |
| 2. Camp → Carr | pre + post, trained on Camp only | whole-event external test | cross-event transfer of a mapping model |
| 3. BC LOFO | pre-fire only | whole-event external test × 12 | prospective pre-fire prediction |

Stage 2 removes "same fire"; Stage 3 additionally removes "post-fire imagery".
This ladder makes the failure at Stage 3 interpretable: it is not that the
feature family is too weak (Stage 2 uses the same spectral indices and transfers
fine), it is that the *pre-fire landscape alone* does not determine post-fire
severity.

## 2. Common data handling

- **Reflectance**: Landsat Collection 2 Level-2 surface reflectance =
  `DN * 0.0000275 - 0.2`.
- **Clear mask**: `QA_PIXEL` bits 0-5 (fill, dilated cloud, cirrus, cloud, cloud
  shadow, snow/ice) all unset AND `QA_RADSAT == 0`. Implemented once in
  `src/preprocessing/qa_mask.py`.
- **Resampling**: bilinear for continuous bands; nearest for QA masks and
  categorical labels (never interpolate a class code).
- **Access**: all remote assets are signed `/vsicurl` windows (Planetary
  Computer SAS); the national CanLaBS raster is cut with `gdalwarp -cutline`
  so it is never downloaded.
- **GDAL discovery**: `src/gdal_tools.py` locates `gdalwarp` / `ogr2ogr` /
  `gdal_rasterize` via `GDALWARP`-style env vars, `GDAL_BIN_DIR`, or `PATH`. No
  absolute install path appears in the code.

## 3. Stage 1 — Camp resolution ablation (exact protocol)

1. Fetch the MTBS event polygon for `CA3982012144020181108`.
2. Build a pixel-aligned grid in UTM 10N at the target resolution (30 or 90 m)
   with a 2-pixel pad (`src/preprocessing/grid.py::event_grid`).
3. Read the pinned pre/post Landsat items, QA-mask, stack the 13 features
   (`src/features/usa_stack.py`).
4. Warp `mtbs_severity_conus_2018_30m` onto the grid; keep classes 1-4.
5. Rasterise the event polygon; valid = mask ∧ all features finite ∧ class ∈
   {1,2,3,4}.
6. Spatial groups: `floor(x / 5000) * 100000 + floor(y / 5000)`.
7. 5-fold `GroupKFold`; per fold, per class ≤ 25,000 training pixels with
   `default_rng(2026 + fold)`; RF as in `configs/camp_fire_30m.yaml`.
8. Score all test-fold pixels; pooled OOF metrics + per-fold metrics.
9. Baseline: dNBR thresholded by the event's own MTBS thresholds on the same
   pixels.
10. Repeat at 90 m with identical settings except resolution.

## 4. Stage 2 — Camp-to-Carr transfer (exact protocol)

1. Rebuild the Camp design matrix from the 30 m feature raster
   (`camp_features_30m.tif` written by Stage 1) so training uses the exact same
   pixels as the ablation.
2. Balanced subsample: ≤ 25,000 pixels per class from Camp only,
   `default_rng(2026)`.
3. Fit one RF (`random_state=2026`); do **not** touch it again.
4. Build the Carr grid + features from Carr's pinned scenes; label with Carr's
   MTBS severity raster.
5. Predict every valid Carr pixel; metrics:
   - `camp_trained_rf` — the external-test number.
   - `frozen_camp_dnbr` — dNBR with Camp's thresholds `[0.060, 0.301, 0.570]`
     (also fully external).
   - `carr_calibrated_dnbr_reference_only` — Carr's own thresholds; descriptive
     only, excluded from comparisons because it is in-sample.
6. `rf_performance_retention = rf_macro_f1 / camp_oof_macro_f1`.
7. Feature shift table: per feature, standardized mean difference
   `(mean_carr - mean_camp) / sqrt((var_camp + var_carr) / 2)`.

## 5. Stage 3 — BC LOFO (exact protocol)

### 5.1 Feature-table construction (`src/features/bc_table.py`)

For each of the 12 events, per pixel:

- inside the rasterised NBAC perimeter (exact mask, not label footprint);
- Hansen tree cover ≥ 30 and no `lossyear` strictly before the fire year;
- pre-fire composite bands finite and indices in [-1, 1];
- terrain finite;
- target finite and in [-2, 2].

Spatial blocks (300 m) are assigned at table-build time from pixel coordinates,
before any modelling.

### 5.2 LOFO (`src/validation/leave_one_fire_out.py`)

- Folds: one per fire, in sorted fire-id order.
- Training sample: equal 15,000 px per training fire; within each fire,
  proportional to target deciles (`proportional_stratified_sample`); seed
  `42 + fold_index * 100`.
- Models per fold: naive mean, linear (StandardScaler + OLS), RF spectral, RF
  structure+terrain, RF full.
- RF: 160 / 20 / 10 / 0.7 / bootstrap / seed 42 — fixed a priori, identical for
  all folds and feature sets.
- High threshold: training `target_dNBR` q75, applied to the test fire.
- Resumable: a `fold_complete.json` marker per fold; interrupted runs resume.

### 5.3 Aggregation (`experiments/canada_bc/summarize_lofo.py`)

- Per-event pixel metrics (RMSE, MAE, R² vs event's own mean, Pearson r,
  top-quartile IoU/F1) and per-300 m-block metrics.
- Event-macro means with 95% percentile bootstrap over 12 events, 10,000 reps,
  seed 42.
- Feature importance: mean ± sd of impurity importance across folds; labeled
  descriptive.

## 6. Interpretation rules

- R² in Stage 3 is against the **held-out fire's own mean** — a model "wins"
  only by beating the trivial baseline of that fire's average severity. This is
  why negative values are meaningful and why the naive baseline itself scores
  −1.10.
- MTBS/CanLaBS are remote-sensing interpretations; every score is agreement
  with a product, not field truth.
- Impurity importance is descriptive: correlated predictors share importance,
  and no causal ranking is claimed.
