"""Remote catalogue and asset access for both study regions.

Modules
-------
stac
    Microsoft Planetary Computer STAC queries and SAS asset signing.
mtbs
    MTBS event polygons (ArcGIS FeatureServer) and thematic severity rasters.
nbac
    National Burned Area Composite event attributes and perimeters.
canlabs
    CanLaBS v2 national continuous dNBR windowed cropping.
hansen
    Hansen Global Forest Change tree cover and loss year tiles.
dem
    Copernicus GLO-30 elevation windows.
"""

__all__ = ["canlabs", "dem", "hansen", "mtbs", "nbac", "stac"]
