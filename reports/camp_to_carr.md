# Camp Fire within-event ablation and Camp-to-Carr external transfer

## 1. Purpose

The United States experiments answer two questions in sequence:

1. Within one event (Camp Fire), how well can a Random Forest map four-class burn
   severity under a spatial cross-validation that prevents leakage, and how does
   it compare with the operational MTBS dNBR threshold rule?
2. When that Camp-trained model is applied **unchanged** to a different event
   (Carr Fire), how much of its performance survives?

The second question is the scientifically interesting one: retrospective
severity products are event-specific by construction (each MTBS event has its own
operator-chosen dNBR thresholds), so the interesting test is whether a *single*
frozen model generalizes across events without any test-event tuning.

## 2. Data

| Item | Identifier / source |
|---|---|
| Camp Fire event polygon + thresholds | MTBS ArcGIS FeatureServer, `CA3982012144020181108` |
| Carr Fire event polygon + thresholds | MTBS ArcGIS FeatureServer, `CA4065012263020180723` |
| Camp pre-fire scene | `LC08_L2SP_044032_20180719_02_T1` (Landsat 8 C2 L2) |
| Camp post-fire scene | `LC08_L2SP_044032_20190722_02_T1` |
| Carr pre-fire scene | `LC08_L2SP_045032_20180710_02_T1` |
| Carr post-fire scene | `LC08_L2SP_045032_20190729_02_T1` |
| Severity reference | MTBS `mtbs_severity_conus_2018_30m`, classes 1-4 |

All Landsat/MTBS assets are read as signed `/vsicurl` windows from Microsoft
Planetary Computer; no scene is stored in the repository.

## 3. Feature stack (13 bands)

| Group | Bands |
|---|---|
| Pre-fire indices | `pre_ndvi`, `pre_nbr`, `pre_ndmi` |
| Post-fire indices | `post_ndvi`, `post_nbr`, `post_ndmi` |
| Differences | `d_ndvi`, `dnbr`, `d_ndmi` (pre minus post) |
| Post-fire reflectance | `post_red`, `post_nir`, `post_swir1`, `post_swir2` |

The post-fire terms are deliberate: the task is retrospective severity
*mapping*, exactly what MTBS does. (The Canadian experiment forbids all post-fire
terms because its task is prospective *prediction* — see the BC report.)

Surface reflectance = `DN * 0.0000275 - 0.2`, masked by `QA_PIXEL` bits 0-5 and
`QA_RADSAT == 0`, reflectance clipped to `[0, 1]`.

## 4. Protocols

### 4.1 Camp resolution ablation

- Grid: pixel-aligned event grid at 30 m and 90 m in UTM 10N (`EPSG:32610`),
  with a two-pixel pad, derived from the MTBS event polygon.
- Validation: 5-fold `GroupKFold` on 5 km spatial blocks. Grouping guarantees no
  spatial neighbour appears in both train and test.
- Training subsample: per fold, up to 25,000 pixels per severity class drawn with
  `rng = default_rng(random_seed + fold)`; RF: 350 trees, `min_samples_leaf=5`,
  `max_features=sqrt`, `class_weight=balanced_subsample`, same per-fold seed
  offset.
- Test: **all** valid pixels of the test fold are scored (no test subsampling).
- Baseline: MTBS official dNBR thresholds (read from the event polygon
  properties, divided by 1000) applied to the same held-out pixels.

### 4.2 Camp-to-Carr external transfer

- Training: exactly the Stage-1 30 m protocol's balanced subsample of Camp
  pixels (25,000/class), one RF with `random_state=2026`.
- Test: the complete Carr event valid population (1,032,984 pixels), predicted
  with the frozen model. **No Carr data of any kind enters the model.**
- Baselines, both frozen (no test-event information):
  - `frozen_camp_dnbr_thresholds = [0.060, 0.301, 0.570]` (Camp's own MTBS
    thresholds).
  - `carr_official_thresholds_reference_only` — reported descriptively only,
    because a rule calibrated on the test event is in-sample and cannot be a
    fair external baseline.
- A per-feature standardized mean difference (Camp vs Carr) is reported to
  quantify the covariate shift the transfer must survive.

## 5. Results

### 5.1 Camp internal CV

| Resolution | n valid pixels | RF OOF Macro-F1 | RF balanced acc. | dNBR baseline Macro-F1 | fold mean ± sd |
|---|---|---|---|---|---|
| 30 m | 684,537 | **0.8089** | 0.8225 | 0.8162 | 0.8001 ± 0.0253 |
| 90 m | 75,901 | **0.7523** | 0.7548 | 0.7504 | 0.7472 ± 0.0174 |

Honest note: at both resolutions the RF and the dNBR rule are close; at 30 m the
dNBR rule is *slightly better* (0.8162 vs 0.8089). dNBR thresholding was designed
for exactly this task. The RF's advantage is not within-event accuracy — it is
robustness, most clearly visible in the transfer (Section 5.2) and in the 90 m
column where the two are essentially tied.

### 5.2 Camp-to-Carr transfer

| Method | Macro-F1 | Balanced acc. | High-severity recall |
|---|---|---|---|
| Camp-trained RF (untouched) | **0.797554** | **0.795827** | **0.809102** |
| Frozen Camp dNBR thresholds | 0.789306 | 0.775353 | 0.765114 |
| Carr official thresholds (reference only) | 0.793454 | — | — |

- RF performance retention vs Camp OOF: 0.797554 / 0.808934 = **0.986**.
- Interpretation: the model loses ~1.4% of its Macro-F1 when the fire changes
  from Camp to Carr. All three numbers sit in a narrow band (0.79-0.80), so the
  claim is not "the RF wins" — it is "the RF transfers essentially
  undamaged, with no test-event information"; its clearest gains are balanced
  accuracy and high-severity recall.

## 6. Reproduce

```bash
python experiments/usa_camp_carr/run_ablation.py --resolutions 30 90
python experiments/usa_camp_carr/run_transfer.py
```

Requires `GDALWARP` on `PATH` (or `GDAL_BIN_DIR`), network access to Planetary
Computer, and `pip install -r requirements.txt`.

## 7. Files

- `experiments/usa_camp_carr/results/metrics_{30,90}m.json` — full per-fold and
  pooled metrics, confusion matrices, class counts.
- `experiments/usa_camp_carr/results/camp_to_carr_metrics.json` — transfer
  metrics, baselines, feature shift table, runtime.
- `experiments/usa_camp_carr/results/resolution_summary.json` — compact ablation
  summary.
- `figures/usa/` — resolution_ablation, camp_event_maps_30m, camp_to_carr_maps.
