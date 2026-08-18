"""Spatial block identifiers for grouped validation.

Adjacent satellite pixels are strongly autocorrelated. Random pixel-level
cross-validation therefore places near-duplicate pixels in both the training and
the test split and reports a score that mostly measures interpolation. Grouping
by coarse spatial blocks forces every held-out pixel to be at least one block
away from its training neighbours.

Block sizes used in the project
-------------------------------
5000 m (California)
    Large enough to break the within-event autocorrelation of a single 620 km²
    fire while still leaving five usable folds.
300 m (British Columbia)
    Used for block-mean aggregation, not for splitting: the Canadian design holds
    out whole fires, and the 300 m blocks answer the separate question of whether
    the model captures coarse spatial pattern even when pixel-level R-squared is
    poor.
"""

from __future__ import annotations

import numpy as np
from affine import Affine

__all__ = ["block_indices", "block_group_ids", "pixel_centre_coordinates"]

_ROW_MULTIPLIER = 100000


def pixel_centre_coordinates(
    transform: Affine, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map array indices to projected pixel-centre coordinates."""
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    return xs, ys


def block_indices(
    xs: np.ndarray, ys: np.ndarray, block_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer block row/column indices for projected coordinates."""
    block_col = np.floor(xs / block_size_m).astype("int64")
    block_row = np.floor(ys / block_size_m).astype("int64")
    return block_row, block_col


def block_group_ids(xs: np.ndarray, ys: np.ndarray, block_size_m: float) -> np.ndarray:
    """Return one integer group identifier per pixel for grouped k-fold splitting.

    The two block indices are combined into a single integer, which is what
    ``GroupKFold`` expects. The multiplier is large enough that no realistic study
    extent produces a collision between different blocks.
    """
    block_row, block_col = block_indices(xs, ys, block_size_m)
    return block_col * _ROW_MULTIPLIER + block_row
