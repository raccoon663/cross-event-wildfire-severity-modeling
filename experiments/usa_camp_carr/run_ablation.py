#!/usr/bin/env python3
"""United States experiment: Camp Fire within-event ablation + Camp-to-Carr transfer.

Usage
-----
    python experiments/usa_camp_carr/run_ablation.py
    python experiments/usa_camp_carr/run_transfer.py

Reproducing the reported numbers end-to-end needs:

* network access to the Planetary Computer STAC API (Landsat Collection 2
  Level-2, MTBS) and to the MTBS ArcGIS FeatureServer;
* a GDAL installation with ``gdalwarp`` on ``PATH`` (or ``GDAL_BIN_DIR`` /
  ``GDALWARP`` set);
* the configuration files ``configs/camp_fire.yaml`` and
  ``configs/carr_fire.yaml``.

Nothing is cached on disk before the first run except the two feature rasters
written by the ablation, which the transfer stage reads as its training source.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from src.data.usa_dataset import run_camp_resolution_ablation, run_camp_to_carr_transfer

RESULTS = PROJECT_ROOT / "experiments" / "usa_camp_carr" / "results"


def _write_summary(results: list[dict], path: Path) -> None:
    summary = [{
        "resolution_m": row["resolution_m"],
        "n_valid_pixels": row["n_valid_pixels"],
        "rf_macro_f1": row["random_forest_oof"]["macro_f1"],
        "rf_balanced_accuracy": row["random_forest_oof"]["balanced_accuracy"],
        "dnbr_macro_f1": row["dnbr_baseline"]["macro_f1"],
        "fold_mean": row["fold_macro_f1_mean"],
        "fold_std": row["fold_macro_f1_std"],
        "runtime_seconds": row["runtime_seconds"],
        "peak_rss_bytes_observed": row["peak_rss_bytes_observed"],
    } for row in results]
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _ablation_plot(summary: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(summary))
    width = 0.34
    axis.bar(positions - width / 2, [row["dnbr_macro_f1"] for row in summary], width,
             label="Official MTBS dNBR thresholds", color="#E69F00")
    axis.bar(positions + width / 2, [row["rf_macro_f1"] for row in summary], width,
             label="Random Forest OOF", color="#0072B2")
    axis.set_xticks(positions, [f"{row['resolution_m']} m" for row in summary])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Camp Fire resolution ablation")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[30, 90],
                        help="Resolutions to run (default: 30 90)")
    parser.add_argument("--output", type=Path, default=RESULTS,
                        help="Output directory (default: experiments/usa_camp_carr/results)")
    args = parser.parse_args()

    import numpy as np  # noqa: F401  (used by _ablation_plot)

    results = [run_camp_resolution_ablation(
        f"camp_fire_{resolution}m", args.output, n_jobs=-1,
    ) for resolution in args.resolutions]
    results.sort(key=lambda row: row["resolution_m"])
    _write_summary(results, args.output / "resolution_summary.json")
    _ablation_plot(results, args.output / "resolution_ablation.png")
    print(json.dumps([{k: r[k] for k in ("resolution_m", "random_forest_oof", "dnbr_baseline")}
                      for r in results], indent=2))


if __name__ == "__main__":
    main()
