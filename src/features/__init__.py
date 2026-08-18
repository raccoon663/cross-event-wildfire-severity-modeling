"""Feature construction for both study regions.

Modules
-------
spectral_indices
    Normalised-difference indices and the dNBR difference convention.
terrain
    Slope and aspect derived from a DEM window.
usa_stack
    The 13-band post-fire-informed feature stack used in California.
bc_table
    The leakage-controlled pre-fire-only pixel table used in British Columbia.
"""

__all__ = ["bc_table", "spectral_indices", "terrain", "usa_stack"]
