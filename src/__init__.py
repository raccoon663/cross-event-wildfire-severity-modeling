"""Reusable library code for cross-event wildfire burn-severity modelling.

The package is organised by pipeline stage so that the two study regions
(California, USA and British Columbia, Canada) share one implementation of
data access, masking, feature construction, validation and plotting.

Sub-packages
------------
data
    Remote catalogue and asset access (STAC, MTBS, NBAC, CanLaBS, Hansen, DEM).
preprocessing
    Analysis-grid definition, GDAL-based warping and Landsat QA masking.
features
    Spectral indices, terrain derivatives and the two feature-table builders.
models
    dNBR threshold baselines, Random Forest classifier/regressor, linear and
    naive reference models.
validation
    Spatial block cross-validation, leave-one-fire-out splitting, metrics and
    event-level bootstrap intervals.
visualization
    Figure and map helpers shared by both experiments.
"""

__all__ = [
    "data",
    "features",
    "models",
    "preprocessing",
    "validation",
    "visualization",
]

__version__ = "1.0.0"
