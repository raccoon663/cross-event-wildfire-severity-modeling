"""Percentile bootstrap over events.

The Canadian experiment has twelve events, so the relevant sampling unit for
uncertainty is the *event*, not the pixel. Resampling 821 214 pixels would produce
absurdly tight intervals that describe the size of the raster rather than the
number of independent fires.

Non-finite event scores (for example the Pearson r of the constant naive baseline)
are dropped before resampling instead of being silently coerced, so an interval is
always computed over a known number of usable events.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_event_mean"]


def bootstrap_event_mean(
    values, seed: int = 42, repetitions: int = 10000
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for a 95 percent percentile bootstrap.

    Parameters
    ----------
    values
        One score per event.
    seed
        Seed for the resampling generator, so intervals are reproducible.
    repetitions
        Bootstrap replicates; 10 000 is used for every reported interval.
    """
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
