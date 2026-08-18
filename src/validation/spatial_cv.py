"""Grouped spatial cross-validation for the within-event experiment.

Protocol
--------
1. Assign every valid pixel to a spatial block (5 km in the Camp Fire runs).
2. Split blocks into ``n_splits`` folds with ``GroupKFold``, so no block is ever
   split between training and test.
3. Inside each fold, draw a class-balanced training subsample from the *training*
   split only, fit the model, and predict every pixel of the test split.
4. Collect the out-of-fold predictions so a single confusion matrix and Macro-F1
   can be computed over the whole event without a pixel ever scoring itself.

Both the per-fold scores and the pooled out-of-fold score are returned: the spread
across folds shows how much of the reported performance depends on which part of
the fire was held out.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from .metrics import classification_metrics

__all__ = ["grouped_out_of_fold_predictions"]


def grouped_out_of_fold_predictions(
    design_matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    model_factory: Callable[[int], object],
    subsample: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    verbose_prefix: str | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run grouped k-fold and return out-of-fold predictions with per-fold metrics.

    Parameters
    ----------
    design_matrix, labels, groups
        Pixel-level features, severity classes and spatial block identifiers.
    n_splits
        Number of folds.
    model_factory
        ``fold -> estimator``; receives the 1-based fold index so the random state
        can be offset per fold.
    subsample
        ``(labels, train_indices, fold) -> selected_indices``; applied to the
        training split only.
    verbose_prefix
        When set, a one-line progress message is printed per fold.

    Returns
    -------
    tuple
        ``(out_of_fold_predictions, fold_records)``.
    """
    splitter = GroupKFold(n_splits=n_splits)
    out_of_fold = np.zeros(len(labels), dtype="uint8")
    fold_records: list[dict] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(design_matrix, labels, groups), 1
    ):
        selected = subsample(labels, train_index, fold)
        model = model_factory(fold)
        model.fit(design_matrix[selected], labels[selected])
        prediction = model.predict(design_matrix[test_index]).astype("uint8")
        out_of_fold[test_index] = prediction
        record = {
            "fold": fold,
            "n_train_sampled": int(len(selected)),
            "n_test": int(len(test_index)),
            **classification_metrics(labels[test_index], prediction),
        }
        fold_records.append(record)
        if verbose_prefix:
            print(f"{verbose_prefix} fold {fold}: F1={record['macro_f1']:.4f}", flush=True)
    return out_of_fold, fold_records
