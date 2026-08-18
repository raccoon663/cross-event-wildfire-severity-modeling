"""Validation protocols, metrics and uncertainty.

Modules
-------
spatial_blocks
    Spatial block identifiers used as cross-validation groups.
spatial_cv
    Grouped k-fold splitting over spatial blocks (within-event generalisation).
leave_one_fire_out
    Event-level splitting (cross-event generalisation).
metrics
    Classification and regression metrics, including the top-quartile overlap.
bootstrap
    Percentile bootstrap over events for the event-macro summary.
"""

__all__ = ["bootstrap", "leave_one_fire_out", "metrics", "spatial_blocks", "spatial_cv"]
