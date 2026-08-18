"""Random Forest severity classifier for the United States experiment.

Two design choices matter for the reported numbers:

class-balanced training subsample
    Burn-severity classes are strongly imbalanced within an event. A fixed cap of
    pixels *per class* is drawn from the training split, combined with
    ``class_weight="balanced_subsample"``, so the model is not driven by the
    dominant class. The cap also bounds runtime on a laptop.

seed offset per fold
    The random state is ``random_seed + fold``, so folds are independent but the
    whole run is reproducible from a single seed in the configuration file.

The held-out split is never subsampled: every valid pixel of the test fold is
scored.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

__all__ = ["build_classifier", "balanced_class_subsample", "SEVERITY_CLASSES"]

SEVERITY_CLASSES = (1, 2, 3, 4)


def build_classifier(rf_config: dict, random_state: int, n_jobs: int = -1) -> RandomForestClassifier:
    """Instantiate the classifier from a configuration block."""
    return RandomForestClassifier(
        n_estimators=rf_config["n_estimators"],
        min_samples_leaf=rf_config["min_samples_leaf"],
        max_features=rf_config["max_features"],
        class_weight=rf_config["class_weight"],
        n_jobs=n_jobs,
        random_state=random_state,
    )


def balanced_class_subsample(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    max_per_class: int,
    seed: int,
    classes: tuple[int, ...] = SEVERITY_CLASSES,
) -> np.ndarray:
    """Draw up to ``max_per_class`` indices for each severity class.

    Parameters
    ----------
    labels
        Full label vector aligned with the design matrix.
    candidate_indices
        Positions eligible for training, i.e. the training split only.
    max_per_class
        Cap per severity class.
    seed
        Seed for the local generator.

    Returns
    -------
    numpy.ndarray
        Concatenated selected indices; classes with fewer candidates than the cap
        contribute all of theirs.
    """
    generator = np.random.default_rng(seed)
    selected = []
    for severity_class in classes:
        candidates = candidate_indices[labels[candidate_indices] == severity_class]
        take = min(max_per_class, len(candidates))
        selected.append(generator.choice(candidates, take, replace=False))
    return np.concatenate(selected)
