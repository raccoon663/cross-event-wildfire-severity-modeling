"""Analysis-grid definition, GDAL warping and Landsat quality masking.

Modules
-------
grid
    Event-aligned target grid derived from a fire perimeter.
reproject
    ``gdalwarp`` wrappers for windowed reads of remote assets.
qa_mask
    Landsat Collection 2 ``QA_PIXEL`` / ``QA_RADSAT`` bit decoding.
"""

__all__ = ["grid", "qa_mask", "reproject"]
