"""Continuous-dNBR models and reference baselines for the Canadian experiment.

Model ladder
------------
``naive_training_mean``
    Predicts the mean target of the training fires. Any model that cannot beat it
    on a held-out fire has learned nothing transferable, so this is the reference
    that makes a negative R-squared interpretable rather than merely embarrassing.
``linear_full``
    Standardised linear regression on all predictors; separates "the relationship
    is non-linear" from "the predictors carry no signal".
``rf_spectral`` / ``rf_structure_terrain`` / ``rf_full``
    Random Forest on pre-fire spectra, on structure plus terrain, and on both.
    The split tells which predictor family carries the transferable signal.

Hyper-parameters are fixed a priori in the configuration file and were not tuned
after inspecting any held-out fire; the same values are used for all folds and
all three feature sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "SPECTRAL_FEATURES",
    "STRUCTURE_TERRAIN_FEATURES",
    "FULL_FEATURES",
    "MODEL_SPECS",
    "build_regressor",
    "build_linear_model",
    "fit_predict",
    "proportional_stratified_sample",
    "training_sample",
]

SPECTRAL_FEATURES = ["pre_red", "pre_nir", "pre_swir1", "pre_swir2", "pre_NDVI", "pre_NBR", "pre_NDMI"]
STRUCTURE_TERRAIN_FEATURES = ["tree_cover2000", "elevation", "slope", "aspect_sin", "aspect_cos"]
FULL_FEATURES = SPECTRAL_FEATURES + STRUCTURE_TERRAIN_FEATURES

MODEL_SPECS: list[tuple[str, list[str]]] = [
    ("naive_training_mean", []),
    ("linear_full", FULL_FEATURES),
    ("rf_spectral", SPECTRAL_FEATURES),
    ("rf_structure_terrain", STRUCTURE_TERRAIN_FEATURES),
    ("rf_full", FULL_FEATURES),
]


def build_regressor(
    n_estimators: int = 160,
    max_depth: int = 20,
    min_samples_leaf: int = 10,
    max_features: float = 0.7,
    n_jobs: int = -1,
    seed: int = 42,
) -> RandomForestRegressor:
    """Instantiate the Random Forest regressor with the fixed configuration."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=min(max_features, 1.0),
        n_jobs=n_jobs,
        random_state=seed,
        bootstrap=True,
    )


def build_linear_model(n_jobs: int = -1):
    """Standardised ordinary least squares."""
    return make_pipeline(StandardScaler(), LinearRegression(n_jobs=n_jobs))


def fit_predict(
    model_name: str,
    features: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    rf_parameters: dict | None = None,
    n_jobs: int = -1,
    seed: int = 42,
):
    """Fit one model on the training fires and predict the held-out fire.

    Returns ``(prediction, fitted_model)``. The naive baseline returns ``None`` as
    the model because it holds no parameters beyond the training mean.
    """
    if model_name == "naive_training_mean":
        return np.full(len(test), train["target_dNBR"].mean(), dtype="float32"), None
    if model_name == "linear_full":
        model = build_linear_model(n_jobs=n_jobs)
    else:
        parameters = dict(rf_parameters or {})
        parameters.setdefault("n_jobs", n_jobs)
        parameters.setdefault("seed", seed)
        model = build_regressor(**parameters)
    model.fit(train[features], train["target_dNBR"])
    return model.predict(test[features]).astype("float32"), model


def proportional_stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Subsample an event while preserving its target distribution.

    Pixels are binned into within-event dNBR deciles and the quota is allocated
    proportionally, with the remainder given to the bins whose fractional share is
    largest. This keeps the extremes of a fire represented instead of letting a
    uniform random draw flood the training set with the modal severity.
    """
    if len(frame) <= n:
        return frame.copy()
    bins = pd.qcut(frame["target_dNBR"], q=10, labels=False, duplicates="drop")
    generator = np.random.default_rng(seed)
    chosen: list[int] = []
    counts = bins.value_counts().sort_index()
    raw = counts / counts.sum() * n
    allocation = np.floor(raw).astype(int)
    remainder = n - int(allocation.sum())
    if remainder:
        for key in (raw - allocation).sort_values(ascending=False).index[:remainder]:
            allocation.loc[key] += 1
    for key, count in allocation.items():
        positions = np.flatnonzero(bins.to_numpy() == key)
        chosen.extend(generator.choice(positions, size=int(count), replace=False).tolist())
    return frame.iloc[np.sort(np.asarray(chosen))].copy()


def training_sample(
    table: pd.DataFrame, test_fire: str, pixels_per_fire: int, seed: int
) -> pd.DataFrame:
    """Draw an equal pixel quota from every training fire.

    Equal quotas stop one large fire from dominating the fit, which matters
    because event area in the catalogue spans more than an order of magnitude. The
    held-out fire is excluded before sampling and is never subsampled.
    """
    pieces = []
    train = table.loc[table["fire_id"] != test_fire]
    for offset, (_, event) in enumerate(train.groupby("fire_id", sort=True)):
        pieces.append(proportional_stratified_sample(event, pixels_per_fire, seed + offset))
    return pd.concat(pieces, ignore_index=True)
