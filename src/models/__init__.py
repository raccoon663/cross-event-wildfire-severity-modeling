"""Models and reference baselines.

Modules
-------
dnbr_rule
    Operational three-threshold dNBR rule. Its two uses are kept strictly
    apart: thresholds *frozen from the training event* form the fair
    cross-event baseline, while thresholds calibrated on the test event are
    reported as a reference only.
rf_classifier
    Random Forest severity-class model for the United States experiment.
rf_regressor
    Random Forest continuous-dNBR model plus the linear and naive references
    for the Canadian experiment.
"""

__all__ = ["dnbr_rule", "rf_classifier", "rf_regressor"]
