# British Columbia: prospective pre-fire burn-severity prediction (CanLaBS dNBR)

## 1. Purpose

Stage 3 is the hardest question in this repository and the reason the project
exists: **can burn severity be predicted for a fire that has not happened yet,
using only information that is knowable before ignition?**

If the answer were "yes, at usable accuracy", retrospective severity products
could be complemented by *pre-fire risk maps*. The measured answer, with a
leakage-controlled protocol, is **no at the pixel level** — and the shape of that
failure is the finding.

## 2. Data

| Product | Role | Access |
|---|---|---|
| NBAC (National Burned Area Composite) | 12 selected fires, event attributes + perimeters | attribute workbook + annual shapefiles |
| CanLaBS v2 (1985-2024, v20260121) | continuous dNBR **target** (source int ÷ 1000) | national COG, per-event `gdalwarp -cutline` |
| Landsat C2 L2 (Planetary Computer) | pre-fire median composite (year − 1, Jun 1 - Sep 30), ≤ 3 scenes | `landsat-c2-l2` STAC |
| Hansen GFC-2023-v1.11 | `treecover2000`, `lossyear` (pre-fire loss exclusion) | Google Storage tiles |
| Copernicus GLO-30 (Planetary Computer) | elevation; slope/aspect derived at 30 m | `cop-dem-glo-30` STAC |

### The 12 selected events

`2014_2151, 2014_612, 2016_174, 2017_849, 2017_1816, 2018_1274, 2018_1726,
2019_105, 2020_293, 2021_1183, 2021_1501, 2022_607` (list frozen in
`configs/bc_lofo.yaml`).

Selection gates (all applied before modelling): area ≥ 1,000 ha; CanLaBS
coverage of the rasterised NBAC perimeter ≥ 0.90; ≥ 50% of forest pixels with
valid pre-fire spectra; ≥ 5,000 valid forest pixels; Hansen tree cover ≥ 30 with
no loss strictly before the fire year.

## 3. Leakage control — the design rule

The feature table contains **only**:

- pre-fire spectral: `pre_red, pre_nir, pre_swir1, pre_swir2, pre_NDVI, pre_NBR,
  pre_NDMI`;
- structure: `tree_cover2000`, `forest_persistence_years_since_2000`;
- terrain: `elevation, slope, aspect_sin, aspect_cos`.

**Forbidden** (enforced by assertion in code): `post_NDVI, post_NBR, post_NDMI,
dNDVI, dNDMI, dNBR_feature` — anything derived from post-fire imagery or from the
difference between pre and post.

The only label written is the CanLaBS dNBR target. The full valid forest-pixel
population per event is retained; no test-side subsampling exists.

### Analysis population

`inside NBAC perimeter & forest-in-2000 (tc≥30) & no pre-fire loss & valid
pre-fire spectra & valid terrain & |target| ≤ 2` → **821,214 pixels** across 12
fires.

## 4. Protocol

- **Leave-one-fire-out**: 12 folds. Training = equal 15,000-pixel quota per
  training fire, each quota proportionally stratified by within-event target
  deciles (preserves the severity distribution of each contributing fire);
  seed `42 + fold_index * 100`.
- **Models** (all fitted inside each fold, no test-fire data):
  `naive_training_mean`, `linear_full` (standardized OLS), `rf_spectral`,
  `rf_structure_terrain`, `rf_full`.
- **RF**: 160 trees, `max_depth=20`, `min_samples_leaf=10`,
  `max_features=0.7`, `bootstrap=True`, `random_state=42` — fixed *a priori*,
  never tuned on held-out fires.
- **High-severity threshold**: the training fires' `target_dNBR` upper quartile,
  applied to the held-out fire (never the test fire's own quantile).
- **Scoring**: every valid pixel of the held-out fire; event-macro aggregates
  with a 95% percentile bootstrap over the 12 events (10,000 reps); block-level
  metrics at the preassigned 300 m spatial blocks.

## 5. Results

### 5.1 Event-macro summary (mean over 12 held-out fires; 95% CI)

| Model | mean RMSE | mean MAE | mean R² [95% CI] | mean Pearson r | top25 IoU | top25 F1 |
|---|---|---|---|---|---|---|
| naive_training_mean | 0.257 | 0.219 | −1.10 | — | — | — |
| linear_full | 0.260 | 0.216 | −0.94 | 0.19 | 0.20 | 0.33 |
| rf_spectral | 0.267 | 0.225 | −1.18 | 0.13 | 0.18 | 0.31 |
| rf_structure_terrain | 0.272 | 0.228 | −1.27 | 0.12 | 0.19 | 0.32 |
| **rf_full** | **0.264** | 0.218 | **−0.98** | **0.18** | **0.21** | **0.35** |

### 5.2 Per-event spread (rf_full, pixel level)

| fire_id | R² | Pearson r | RMSE |
|---|---|---|---|
| 2016_174 | −5.25 | −0.20 | 0.302 |
| 2018_1726 | −2.41 | 0.09 | 0.336 |
| 2014_2151 | −1.57 | 0.20 | 0.348 |
| 2021_1183 | −0.77 | 0.15 | 0.221 |
| 2018_1274 | −0.76 | 0.19 | 0.318 |
| 2019_105 | −0.61 | 0.02 | 0.256 |
| 2020_293 | −0.19 | 0.18 | 0.268 |
| 2022_607 | −0.09 | 0.24 | 0.259 |
| 2017_1816 | −0.07 | 0.27 | 0.249 |
| 2017_849 | −0.06 | 0.24 | 0.195 |
| 2021_1501 | −0.02 | 0.32 | 0.232 |
| 2014_612 | **0.02** | **0.46** | 0.187 |

### 5.3 Reading the result

- **Absolute prediction fails.** Every model's mean R² is negative; the best
  event (2014_612) barely reaches 0.02. The models cannot out-predict the
  naive training mean on absolute dNBR, and the RF is not distinguishable from
  linear regression on the aggregate.
- **A weak rank signal survives.** Pearson r ≈ 0.18 (rf_full) and top-quartile
  overlap ≈ 0.21 mean the models do learn *something* about relative severity —
  which fires will contain high-dNBR areas — but not enough to be deployed.
- **The negative R² is the result, not a bug.** It is computed against each
  held-out fire's own mean and quantifies the variance across events that the
  pre-fire landscape does not carry. The protocol, the assertion checks, and
  the per-event spread make the negative value reproducible rather than an
  artefact of a single bad fire.

## 6. Reproduce

```bash
# 1. catalogue + screening (needs NBAC workbook + shapefiles locally)
python experiments/canada_bc/pipeline.py catalog --workbook NBAC.xlsx --output-dir data/canada
python experiments/canada_bc/pipeline.py screen  --catalog data/canada/bc_candidate_catalog.parquet --annual-dir data/nbac_shp --output-dir data/canada
# 2. pre-fire features + gates + DEM cache
python experiments/canada_bc/pipeline.py features --screen-catalog data/canada/batch_screen_catalog.parquet --candidate-catalog data/canada/bc_candidate_catalog.parquet --events-dir data/canada/events --output-dir data/canada
python experiments/canada_bc/pipeline.py table    --catalog data/canada/bc_event_catalog.parquet --events-dir data/canada/events --output-dir data/canada/feature_table
# 3. modelling + summary + figures
python experiments/canada_bc/run_lofo.py         --table data/canada/feature_table/bc_prefire_feature_table.parquet --output-dir experiments/canada_bc/results/lofo
python experiments/canada_bc/summarize_lofo.py   --model-dir experiments/canada_bc/results/lofo --output-dir experiments/canada_bc/results/analysis
python experiments/canada_bc/plot_results.py     --analysis-dir experiments/canada_bc/results/analysis --model-dir experiments/canada_bc/results/lofo --table data/canada/feature_table/bc_prefire_feature_table.parquet --events-dir data/canada/events --output-dir figures/canada
```

## 7. Files

- `experiments/canada_bc/results/event_macro_summary_with_ci.csv` — the table in
  §5.1 with bootstrap CIs.
- `experiments/canada_bc/results/event_pixel_metrics.csv` / `event_block_metrics.csv`
  — per-event and per-300 m-block scores.
- `experiments/canada_bc/results/feature_importance_summary.csv` — mean ± sd
  impurity importance across folds (descriptive only).
- `experiments/canada_bc/results/event_catalog/` — event catalogue and feature
  audit.
- `figures/canada/` and `figures/overview/figure1_study_design_lofo.png`.
