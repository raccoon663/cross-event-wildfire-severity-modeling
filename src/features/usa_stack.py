"""The 13-band feature stack used for the United States experiment.

Band order is fixed by :data:`USA_FEATURE_NAMES` and must not change, because the
Camp-trained model is applied unchanged to the Carr event and the frozen dNBR
rule indexes this stack by position.

Feature groups
--------------
pre-fire indices
    ``pre_ndvi, pre_nbr, pre_ndmi`` - what the stand looked like before the fire.
post-fire indices
    ``post_ndvi, post_nbr, post_ndmi`` - the state of the surface after it.
differences
    ``d_ndvi, dnbr, d_ndmi`` - pre minus post; ``dnbr`` is the operational index.
post-fire reflectance
    ``post_red, post_nir, post_swir1, post_swir2`` - raw signal that lets the
    model separate char, ash and exposed soil, which the ratio indices conflate.

This stack deliberately uses post-fire information: the task is retrospective
severity *mapping*, which is what MTBS also does. The Canadian experiment forbids
every post-fire term, because that task is prospective *prediction*.
"""

from __future__ import annotations

import numpy as np

from .spectral_indices import normalized_difference

__all__ = ["USA_FEATURE_NAMES", "USA_BAND_ORDER", "build_usa_feature_stack", "dnbr_column"]

USA_BAND_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]
USA_FEATURE_NAMES = [
    "pre_ndvi", "pre_nbr", "pre_ndmi",
    "post_ndvi", "post_nbr", "post_ndmi",
    "d_ndvi", "dnbr", "d_ndmi",
    "post_red", "post_nir", "post_swir1", "post_swir2",
]
_MIN_DENOMINATOR = 1e-6


def _index_triplet(cube: np.ndarray) -> list[np.ndarray]:
    """Return ``[NDVI, NBR, NDMI]`` for one six-band reflectance cube.

    Ratio numerators follow the operational definitions: NDVI uses NIR over red,
    NBR uses NIR over SWIR2, NDMI uses NIR over SWIR1. This matches the original
    ``index(cube, "nir08", band)`` calls exactly.
    """
    red = cube[USA_BAND_ORDER.index("red")]
    near_infrared = cube[USA_BAND_ORDER.index("nir08")]
    shortwave1 = cube[USA_BAND_ORDER.index("swir16")]
    shortwave2 = cube[USA_BAND_ORDER.index("swir22")]
    return [
        normalized_difference(near_infrared, red, min_denominator=_MIN_DENOMINATOR),
        normalized_difference(near_infrared, shortwave2, min_denominator=_MIN_DENOMINATOR),
        normalized_difference(near_infrared, shortwave1, min_denominator=_MIN_DENOMINATOR),
    ]


def build_usa_feature_stack(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Assemble the 13-band stack from masked pre- and post-fire cubes.

    Parameters
    ----------
    pre, post
        Arrays of shape ``(6, height, width)`` ordered as
        :data:`USA_BAND_ORDER`, already QA-masked with ``NaN`` for invalid.

    Returns
    -------
    numpy.ndarray
        Shape ``(13, height, width)`` in :data:`USA_FEATURE_NAMES` order.
    """
    pre_indices = _index_triplet(pre)
    post_indices = _index_triplet(post)
    differences = [pre_indices[i] - post_indices[i] for i in range(3)]
    post_reflectance = [
        post[USA_BAND_ORDER.index("red")],
        post[USA_BAND_ORDER.index("nir08")],
        post[USA_BAND_ORDER.index("swir16")],
        post[USA_BAND_ORDER.index("swir22")],
    ]
    return np.stack(pre_indices + post_indices + differences + post_reflectance)


def dnbr_column(design_matrix: np.ndarray) -> np.ndarray:
    """Extract the dNBR column from a design matrix built by this module."""
    return design_matrix[:, USA_FEATURE_NAMES.index("dnbr")]
