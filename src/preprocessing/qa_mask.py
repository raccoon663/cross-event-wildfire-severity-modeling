"""Landsat Collection 2 quality-mask logic.

``QA_PIXEL`` bits 0-5 flag fill, dilated cloud, cirrus, cloud, cloud shadow and
snow/ice. ``QA_RADSAT`` being zero means no band was radiometrically saturated.
A pixel is *clear* only when every one of the six flags is unset and no band is
saturated. All downstream validity masks are built on top of this single
predicate so the two experiments share one definition of "usable".

Both helpers are kept because the two Landsat readers use them at different
points: the United States scene reader combines the masks before scaling
reflectance, while the Canadian composite applies them per scene inside the
median.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CLEAR_BITS", "clear_pixel_mask", "clear_mask"]

CLEAR_BITS = (0, 1, 2, 3, 4, 5)


def clear_pixel_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels with none of the six quality flags set."""
    bad_bits = sum(1 << bit for bit in CLEAR_BITS)
    return (qa_pixel.astype("uint16") & bad_bits) == 0


def clear_mask(qa_pixel: np.ndarray, qa_radsat: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that are clear and not saturated."""
    return clear_pixel_mask(qa_pixel) & (qa_radsat.astype("uint16") == 0)
