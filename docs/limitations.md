# Limitations

An honest account of what this project does not establish. Read this before
citing any number.

## 1. Reference labels are remote-sensing interpretations, not field truth

- MTBS severity classes are derived by analysts from pre/post imagery and
  dNBR thresholds. They correlate with, but are not, measured tree mortality.
- CanLaBS dNBR is a spectral response (a difference of reflectance indices).
  A pixel can show high dNBR because it burned, because the pre-fire state was
  unusually bright, or because of post-fire surface changes unrelated to
  consumption (e.g., ash exposure on unburned litter).
- Consequently every reported accuracy is *agreement with a product*, and the
  BC "target" contains label noise that caps any model's attainable R².

## 2. Event coverage is narrow

- USA: two fires in the same region (northern California, 2018), both in
  chaparral/forest mix, both UTM zone 10N. Cross-event transfer across *one*
  geographic boundary is a proof-of-concept, not a general claim.
- Canada: 12 fires within British Columbia, 2014-2022. Not representative of
  boreal Canada, of other provinces, or of non-forest land covers. The NBAC
  catalogue spans more fire types than this selection covers.

## 3. Imagery constraints

- USA stages use one pre and one post scene per event (the exact MTBS scenes).
  A cloudy or phenologically odd date is not averaged out. This mirrors what
  MTBS itself faces but adds noise.
- BC pre-fire composites use up to three scenes from June-September of the year
  *before* the fire. If the previous year's growing season was anomalous
  (drought, late snowmelt), the composite describes an unrepresentative
  pre-fire state. Scene screening is by reported cloud cover, which is not the
  same as a per-pixel QA pass over all candidates.
- CanLaBS v2 covers 1985-2024; events near the ends of the record may have
  thinner Landsat coverage inside the composite dates.

## 4. Resolution and grid effects

- The 30 m → 90 m ablation changes both resolution and (slightly) the pixel
  lattice. The comparison isolates the combined effect, not resolution alone.
- Terrain (slope/aspect) is derived at 30 m from GLO-30; in steep terrain the
  aspect of a 30 m cell is a crude descriptor of the actual radiative
  environment.

## 5. Modelling limitations

- Hyper-parameters were fixed a priori for BC (stated in the config), but the
  *feature family* (pre-fire spectra + structure + terrain) was chosen by the
  analyst; the negative result applies to this feature set, not to all possible
  pre-fire predictors. ERA5-Land weather, soil moisture and FWI are explicitly
  out of scope and might carry signal (see `configs/bc_lofo.yaml`).
- RF impurity importance is descriptive; correlated predictors (e.g.,
  pre_NDVI vs pre_NBR) split importance arbitrarily. No causal ordering is
  claimed.
- The linear model uses the same features as the RF; a nonlinear model that
  *generalizes across events* was not found here, but the search space is not
  exhaustive (XGBoost / deep learning are out of scope by design).
- Bootstrapped CIs are over 12 events; with n=12 the intervals are wide and
  sensitive to single events, which is precisely why they are reported.

## 6. What the negative BC result does and does not say

It does say: with this leakage-controlled protocol, the pre-fire landscape
available here cannot predict absolute cross-event dNBR, and models barely beat
the training mean in rank terms.

It does not say: severity is unpredictable in principle. It says the *pre-fire
signal in Landsat/Hansen/DEM* does not determine it, at this sample size, with
this target (whose own noise is substantial).

## 7. Reproducibility boundaries

- Reproducing the numbers requires live access to Microsoft Planetary Computer
  (STAC + SAS signing), the MTBS FeatureServer, NBAC (workbook + annual
  shapefiles) and CanLaBS. Any upstream change in these products can shift
  results; the pinned identifiers are in `configs/`.
- The committed results files (`experiments/*/results/`) are the authoritative
  record for the numbers quoted in the README; re-running may produce small
  numerical differences if upstream data changes.
