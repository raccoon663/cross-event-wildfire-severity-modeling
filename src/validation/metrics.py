"""Metrics for the classification and regression tasks.

Classification (United States)
------------------------------
Macro-F1 is the headline metric because severity classes are imbalanced and the
high-severity class is the operationally important one; overall accuracy would be
dominated by the low-severity majority. Balanced accuracy, the per-class
precision/recall/F1 table and the full confusion matrix are reported alongside it
so a single aggregate number cannot hide a collapsed class.

Regression (Canada)
-------------------
RMSE, MAE, R-squared and Pearson r are reported together on purpose. R-squared is
computed against the *held-out event's own* mean, so it answers "does the model
beat knowing this fire's average severity", which is the question that matters for
prospective prediction and the reason the reported values are negative. Pearson r
answers the weaker question of whether the *pattern* is right even when the level
is wrong; keeping both prevents either an over-pessimistic or an over-optimistic
reading.

The top-quartile overlap metrics evaluate the practical task of flagging the most
severely burned quarter of an event. Thresholds are event-relative and derived at
evaluation time, so they describe ranking skill, not calibrated absolute
prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

__all__ = [
    "SEVERITY_CLASSES",
    "classification_metrics",
    "regression_metrics",
    "regression_metrics_with_high",
    "high_threshold_metrics",
    "top_quartile_metrics",
]

SEVERITY_CLASSES = [1, 2, 3, 4]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[int] | None = None,
    include_high_recall: bool = False,
) -> dict:
    """Macro-F1, balanced accuracy, per-class table and confusion matrix."""
    classes = classes or SEVERITY_CLASSES
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    result = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if include_high_recall:
        result["high_severity_recall"] = float(recall[-1])
    result["per_class"] = {
        str(severity_class): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, severity_class in enumerate(classes)
    }
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R-squared against the held-out mean, and Pearson r."""
    residual = y_pred - y_true
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan,
        "pearson_r": float(pearsonr(y_true, y_pred).statistic) if np.ptp(y_pred) > 1e-6 else np.nan,
    }


def high_threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray, high_threshold: float) -> dict:
    """Overlap of the high-severity area under a threshold fixed on training data.

    Using the training upper quartile keeps the threshold independent of the
    held-out fire, unlike :func:`top_quartile_metrics`.
    """
    observed_high = y_true >= high_threshold
    predicted_high = y_pred >= high_threshold
    intersection = int(np.logical_and(observed_high, predicted_high).sum())
    union = int(np.logical_or(observed_high, predicted_high).sum())
    denominator = int(observed_high.sum() + predicted_high.sum())
    return {
        "high_threshold_train_p75": float(high_threshold),
        "high_iou": float(intersection / union) if union else np.nan,
        "high_f1": float(2 * intersection / denominator) if denominator else np.nan,
        "observed_high_fraction": float(observed_high.mean()),
        "predicted_high_fraction": float(predicted_high.mean()),
    }


def top_quartile_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event-relative top-quartile overlap; a ranking diagnostic, not calibration.

    Both the observed and the predicted upper quartile are taken within the
    held-out event, so the metric ignores any constant offset in the prediction.
    """
    if np.ptp(y_pred) <= 1e-6:
        return {"top25_iou": np.nan, "top25_f1": np.nan}
    observed = y_true >= np.quantile(y_true, 0.75)
    predicted = y_pred >= np.quantile(y_pred, 0.75)
    intersection = int((observed & predicted).sum())
    union = int((observed | predicted).sum())
    denominator = int(observed.sum() + predicted.sum())
    return {
        "top25_iou": intersection / union if union else np.nan,
        "top25_f1": 2 * intersection / denominator if denominator else np.nan,
    }


def regression_metrics_with_high(
    y_true: np.ndarray, y_pred: np.ndarray, high_threshold: float
) -> dict:
    """Convenience combination used by the leave-one-fire-out runner."""
    result = regression_metrics(y_true, y_pred)
    result.update(high_threshold_metrics(y_true, y_pred, high_threshold))
    return result
