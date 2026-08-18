"""Shared plotting conventions.

The two baseline/model colours in the United States figures come from a
colour-vision-safe palette (orange ``#E69F00`` for the operational dNBR rule, blue
``#0072B2`` for the Random Forest), so the baseline-versus-model comparison stays
readable in greyscale and for the most common forms of colour blindness.

``matplotlib`` is switched to the non-interactive Agg backend on import, because
every figure in this project is written to a file rather than shown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["MODEL_LABELS", "MODEL_COLORS", "BASELINE_COLOR", "MODEL_COLOR", "save_figure"]

BASELINE_COLOR = "#E69F00"
MODEL_COLOR = "#0072B2"

MODEL_LABELS = {
    "naive_training_mean": "Training mean",
    "linear_full": "Linear",
    "rf_spectral": "RF spectral",
    "rf_structure_terrain": "RF structure + terrain",
    "rf_full": "RF full",
}
MODEL_COLORS = {
    "naive_training_mean": "#7f8c8d",
    "linear_full": "#2980b9",
    "rf_spectral": "#f39c12",
    "rf_structure_terrain": "#27ae60",
    "rf_full": "#8e44ad",
}


def save_figure(figure, path: Path, dpi: int = 180, tight: bool = True) -> Path:
    """Write a figure on a white background and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path, dpi=dpi, facecolor="white",
        **({"bbox_inches": "tight"} if tight else {}),
    )
    plt.close(figure)
    return path
