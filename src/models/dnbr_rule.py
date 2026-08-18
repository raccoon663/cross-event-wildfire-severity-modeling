"""dNBR threshold rule and the operational baseline.

MTBS severity classifies a pixel by comparing its dNBR with operator-chosen
thresholds that differ from event to event. Applying a threshold rule to a
held-out event is exactly what the operational baseline does, which makes it the
natural reference for the Random Forest in both experiments.

Two threshold sets are used and never conflated:

``classify_dnbr(dnbr, thresholds)``
    Generic application of three ordered break points, producing classes 1-4.

Camp's official thresholds
    Used inside Camp (baseline on the OOF pixels) and frozen onto Carr for the
    *external* comparison. This is the fair baseline because it involves no
    test-event information.

Carr's own official thresholds
    Reported under ``carr_official_thresholds_reference_only``. Calibrating on
    the test event would make it an in-sample rule, so it is shown descriptively
    and is never compared against the model.
"""

from __future__ import annotations

import numpy as np

__all__ = ["classify_dnbr"]


def classify_dnbr(dnbr: np.ndarray, thresholds: list[float] | tuple[float, float, float]) -> np.ndarray:
    """Map continuous dNBR to MTBS severity classes 1-4.

    ``thresholds`` are the low, moderate and high break points, e.g.
    ``[0.060, 0.301, 0.570]``. A pixel below all thresholds is class 1
    (unburned/low), at or above the final threshold is class 4 (high).
    """
    low, moderate, high = thresholds
    return np.select(
        [dnbr < low, dnbr < moderate, dnbr < high],
        [1, 2, 3],
        default=4,
    ).astype("uint8")
