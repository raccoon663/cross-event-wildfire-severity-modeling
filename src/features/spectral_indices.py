"""Normalised-difference spectral indices.

Two variants of the range guard are needed because the two experiments accept
slightly different domains:

* the Canadian pre-fire composite uses the plain ratio and only rejects
  non-finite values or indices outside ``[-1, 1]``;
* the United States feature stack additionally rejects a denominator with
  absolute value below ``1e-6``, which the original ablation used to avoid
  amplifying noise in sensor-dead pixels.

The same helper serves both so that any future change is applied once.
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalized_difference"]


def normalized_difference(
    numerator: np.ndarray,
    denominator: np.ndarray,
    min_denominator: float | None = None,
) -> np.ndarray:
    """Return ``(numerator - denominator) / (numerator + denominator)``.

    Invalid cells (non-finite ratio, index outside ``[-1, 1]`` and, when
    requested, near-zero denominators) become ``NaN`` so downstream
    ``all(axis=0)`` valid-pixel masks behave identically to the reference
    implementation.
    """
    total = numerator + denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (numerator - denominator) / total
    invalid = ~np.isfinite(value) | (value < -1) | (value > 1)
    if min_denominator is not None:
        invalid |= np.abs(total) < min_denominator
    value[invalid] = np.nan
    return value
