# Cross-Event Wildfire Severity Modeling with Remote Sensing

**A three-stage transfer study: within-event spatial generalization, cross-event
transfer, and prospective pre-fire prediction of burn severity from Landsat.**

This repository contains two related machine-learning studies and connects them
into a single question: *how much of what a model learns about burn severity in
one fire transfers to another fire it has never seen?*

| Stage | Question | Setting | Validation |
|---|---|---|---|
| 1. Spatial generalization | Can a model trained on one part of a fire score the rest of it? | Camp Fire (California, 2018), MTBS 4-class severity | 5 km spatial-block GroupKFold |
| 2. Cross-event transfer | Does a Camp-trained model survive a different fire untouched? | Camp → Carr (California, 2018) | Strict external test, zero Carr tuning |
| 3. Prospective prediction | Can severity be predicted *before* a fire from pre-fire data alone? | 12 fires in British Columbia (2014-2022), CanLaBS continuous dNBR | Leave-one-fire-out, pre-fire predictors only |

The three stages are deliberately ordered along a spectrum of increasing
difficulty and decreasing information leakage. Stage 3 is the hardest: the model
may only see the landscape *before* it burns.

---

## Headline results

| Experiment | Model | Metric | Value |
|---|---|---|---|
| Camp Fire internal CV (30 m) | Random Forest OOF | Macro-F1 | **0.809** |
| Camp Fire internal CV (30 m) | Official MTBS dNBR thresholds | Macro-F1 | 0.816 |
| Camp Fire internal CV (90 m) | Random Forest OOF | Macro-F1 | 0.752 |
| Camp → Carr external transfer | Camp-trained RF, untouched | Macro-F1 | **0.798** |
| Camp → Carr external transfer | Camp-frozen dNBR thresholds | Macro-F1 | 0.789 |
| BC LOFO (12 fires, pre-fire only) | RF full (pre-fire + structure/terrain) | mean R² | **−0.98** |
| BC LOFO (12 fires, pre-fire only) | Linear (pre-fire + structure/terrain) | mean R² | −0.94 |
| BC LOFO (12 fires, pre-fire only) | RF full | mean Pearson r | 0.18 |

Two findings deserve emphasis precisely because they are uncomfortable:

1. **Inside the Camp Fire, the operational dNBR threshold rule slightly beats the
   Random Forest (0.816 vs 0.809).** dNBR thresholding was designed for exactly
   this within-event mapping task, and it is very good at it. The value of the
   Random Forest appears elsewhere: at coarser resolution (90 m: 0.752 vs 0.750),
   and above all in cross-event transfer (below).
2. **Pre-fire predictors alone cannot predict absolute burn severity across
   events (BC mean R² < 0 for every model).** The models beat nothing at the
   pixel level — RF full (mean R² −0.98) and linear regression (−0.94) are
   statistically indistinguishable from the naive training-mean baseline
   (−1.10). A weak positive rank correlation (Pearson r ≈ 0.18) survives, but it
   is not actionable for absolute prediction.

These are the honest headline results of this project, and they are the point of
the research story: **transfer is where the information is lost, and the loss is
not a modelling bug — it is a property of the problem.**

---

## Why this project exists

Operational burn-severity products (MTBS in the USA, CanLaBS in Canada) are
retrospective: they are produced *after* a fire, from pre- and post-fire
imagery, and validated against expert interpretation. They answer "how badly did
this fire burn?" extremely well.

This project asks the harder question that retrospective products cannot answer:
**can a model trained on past fires predict the severity of a future fire before
it happens, using only what is knowable in advance?**

The answer, carefully and honestly, is *mostly no — and here is exactly how no,
and where the remaining signal lives.* That negative result, measured with a
leakage-controlled protocol, is the scientific content of this repository.

---

## Repository layout

```
configs/                    # YAML experiment definitions (scenes, blocks, RF hyper-parameters)
src/
  data/                     # remote data access: STAC, MTBS, NBAC, CanLaBS, Hansen, DEM
  preprocessing/            # QA masks, event grids, GDAL windowed reprojection
  features/                 # spectral indices, USA feature stack, BC pre-fire table
  models/                   # RF classifier/regressor, dNBR rule, baselines
  validation/               # spatial CV, leave-one-fire-out, metrics, bootstrap
  visualization/            # shared plotting style
  gdal_tools.py             # GDAL binary discovery via env vars (no hard-coded paths)
  config.py                 # YAML loading
experiments/
  usa_camp_carr/            # Stage 1 + 2 entry points and results
  canada_bc/                # Stage 3 entry points and results
figures/
  overview/  usa/  canada/  # the eight figures referenced from this README
reports/                    # detailed write-ups per experiment
docs/                       # methodology, data sources, limitations
LICENSE                     # MIT
```

Everything is runnable with `pip install -r requirements.txt` and a GDAL
installation on `PATH` (or `GDAL_BIN_DIR`). See [docs/methodology.md](docs/methodology.md)
for the full protocol and [docs/data_sources.md](docs/data_sources.md) for the
data provenance. No raw raster, no pixel table and no serialized model is
committed — the repository is self-contained at the level of code + numbers.

---

## Stage 1 — Camp Fire within-event spatial generalization

**Setup.** The 2018 Camp Fire (California, MTBS ID `CA3982012144020181108`), the
exact MTBS pre-/post-fire Landsat scenes, a 13-band feature stack (pre/post
indices, dNBR-family differences, post-fire reflectance), and MTBS 4-class
thematic labels.

**Protocol.** 5-fold cross-validation where the fold split is on 5 km spatial
blocks — no pixel is ever trained and tested with a spatial neighbour in the
other set. Class-balanced training subsampling (≤25,000 pixels per severity
class per fold). The baseline is the official MTBS dNBR thresholds applied to the
same held-out pixels.

**Results.** At 30 m the Random Forest reaches Macro-F1 0.809 with balanced
accuracy 0.823 over 684,537 valid pixels. The dNBR threshold baseline scores
0.816 — *higher* — a reminder that for within-event mapping, a tuned difference
index is a strong competitor. At 90 m the gap narrows further (0.752 vs 0.750):
resolution degradation hurts both, and the RF retains a small edge.

```
figures/usa/resolution_ablation.png
figures/usa/camp_event_maps_30m.png
```

Full metrics: `experiments/usa_camp_carr/results/metrics_30m.json` (and `_90m`).

---

## Stage 2 — Camp-to-Carr cross-event transfer

**Setup.** Train on Camp Fire only; apply the frozen model to the *complete*
2018 Carr Fire event (`CA4065012263020180723`) with **no Carr tuning of any
kind**. Two baselines share the same constraint: the dNBR thresholds frozen from
Camp's own MTBS event, and (descriptively only) Carr's own thresholds.

**Results.** The Camp-trained Random Forest scores Macro-F1 **0.798** on the
1.03-million-pixel Carr event — a performance retention of **98.6%** relative to
its Camp out-of-fold score, and it edges out the Camp-frozen dNBR rule (0.789).
Carr's own official thresholds (0.794, reported as reference only) sit in the
same band, which is the fair reading: none of these numbers is a leap, but the
model loses almost nothing when the fire changes underneath it.

```
figures/usa/camp_to_carr_maps.png
```

Full metrics, including the per-feature distribution shift table:
`experiments/usa_camp_carr/results/camp_to_carr_metrics.json`.

---

## Stage 3 — British Columbia: prospective pre-fire severity prediction

**Setup.** 12 fires across British Columbia (2014-2022, NBAC catalogue), the
CanLaBS v2 continuous dNBR as target, and predictors restricted to *pre-fire*
information only: pre-season Landsat median composites (year before the fire),
Hansen 2000 tree cover minus pre-fire loss, and Copernicus GLO-30 terrain.
821,214 forest pixels in total.

**Leakage rule (enforced in code).** No post-fire band, no post-fire index, no
dNBR/dNDVI/dNDMI difference is allowed in the feature table. The target is
written only as the label. `src/features/bc_table.py` asserts this, and
`run_lofo.py` re-asserts it before fitting.

**Protocol.** Leave-one-fire-out: for each held-out fire, fit on an equal
per-fire pixel quota (15,000 px/fire, proportionally stratified by target
decile) from the other 11, predict every valid pixel of the held-out fire, and
aggregate the 12 event scores with a 95% event-bootstrap interval (10,000 reps).
RF hyper-parameters were fixed a priori; the high-severity threshold is the
*training* upper quartile, never the test fire's.

**Results.**

| Model | mean R² [95% CI] | mean RMSE | mean Pearson r | top-25% IoU |
|---|---|---|---|---|
| Naive training mean | −1.10 | 0.257 | — | — |
| Linear (full) | −0.94 | 0.260 | 0.19 | 0.20 |
| RF spectral | −1.18 | 0.267 | 0.13 | 0.18 |
| RF structure + terrain | −1.27 | 0.272 | 0.12 | 0.19 |
| **RF full** | **−0.98** | 0.264 | **0.18** | **0.21** |

**Reading.** Every model fails on absolute prediction (all mean R² < 0; the
best fire, 2014_612, reaches R² 0.02 while the worst, 2016_174, is −5.25). The
weak positive rank correlation and top-quartile overlap suggest the models learn
*something* about relative severity — but not enough to be deployed. The
negative R² values are the finding, not an error: they quantify how much of
cross-fire severity variance is simply not predictable from the pre-fire
landscape.

```
figures/overview/figure1_study_design_lofo.png
figures/canada/bc_lofo_macro_performance.png
figures/canada/bc_rf_full_event_performance.png
figures/canada/bc_rf_full_feature_importance.png
figures/canada/bc_event_dnbr_distributions.png
```

All per-event and block-level metrics:
`experiments/canada_bc/results/` (`event_pixel_metrics.csv`,
`event_block_metrics.csv`, `event_macro_summary_with_ci.csv`).

---

## What this project does *not* claim

- MTBS and CanLaBS are **thematic remote-sensing interpretations**, not field
  truth. All scores are agreement with these products, not with ground
  mortality.
- The Camp internal dNBR baseline being *higher* than the RF is reported as-is;
  it is not spun.
- Carr's official thresholds are presented as a **reference only**; a rule
  calibrated on the test event would not be an external baseline, so it is
  excluded from every comparison.
- The BC result is **not** presented as a success for pixel-level prediction.
  It is a measured negative result with a reproducible protocol.

## Limitations and reproducibility

See [docs/limitations.md](docs/limitations.md) for the full list (reference-label
noise, single-scene composites for the USA stages, event selection, the
contaminated-scene risk in the BC pre-season windows, and the interpretability of
impurity importances). Reproducing every number requires network access to the
Planetary Computer and NBAC/CanLaBS endpoints plus a GDAL install; the protocol
is documented in [docs/methodology.md](docs/methodology.md), and the exact data
identifiers are frozen in `configs/`.

---

## License

MIT — see [LICENSE](LICENSE).
