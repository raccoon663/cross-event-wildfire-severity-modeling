#!/usr/bin/env python3
"""British Columbia experiment: publication-style diagnostic figures.

Produces the macro-performance bar chart with bootstrap intervals, the per-event
performance chart, feature importance, the target-distribution violin plot, the
300 m block scatter and one observed/predicted/error map per held-out fire.

Usage
-----
    python experiments/canada_bc/plot_results.py \
        --analysis-dir <results>/analysis --model-dir <results>/lofo \
        --table <events_root>/bc_prefire_feature_table.parquet \
        --events-dir <events_root>/events --output-dir <results>/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.visualization.style import (
    MODEL_COLORS,
    MODEL_LABELS,
    save_figure,
)

_ORDER = ["naive_training_mean", "linear_full", "rf_spectral", "rf_structure_terrain", "rf_full"]


def macro_figure(summary: pd.DataFrame, output: Path) -> None:
    data = summary.set_index("model").loc[_ORDER]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    panels = [("rmse", "RMSE", False), ("r2", "R²", True), ("pearson_r", "Pearson r", True)]
    for axis, (metric, label, zero) in zip(axes, panels):
        means = data[f"mean_{metric}"].to_numpy()
        low = data[f"{metric}_ci_low"].to_numpy()
        high = data[f"{metric}_ci_high"].to_numpy()
        positions = np.arange(len(data))
        axis.barh(positions, means, color=[MODEL_COLORS[x] for x in _ORDER], alpha=.9)
        finite = np.isfinite(means) & np.isfinite(low) & np.isfinite(high)
        axis.errorbar(means[finite], positions[finite],
                      xerr=np.vstack([means[finite] - low[finite], high[finite] - means[finite]]),
                      fmt="none", ecolor="#222", capsize=3, linewidth=1)
        if zero:
            axis.axvline(0, color="#333", linewidth=.8)
        axis.set_yticks(positions, [MODEL_LABELS[x] for x in _ORDER] if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("British Columbia leave-one-fire-out performance\n"
                    "Event-macro mean and 95% event-bootstrap interval")
    figure.tight_layout()
    save_figure(figure, output / "bc_lofo_macro_performance.png")


def event_figure(pixel: pd.DataFrame, output: Path) -> None:
    data = pixel.loc[pixel.model == "rf_full"].sort_values("r2", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    positions = np.arange(len(data))
    axes[0].barh(positions, data.r2,
                 color=np.where(data.r2 >= 0, "#2c7fb8", "#d95f0e"))
    axes[0].axvline(0, color="#222", lw=.8)
    axes[1].barh(positions, data.pearson_r, color="#756bb1")
    axes[1].axvline(0, color="#222", lw=.8)
    axes[0].set_yticks(positions, data.fire_id)
    axes[1].set_yticks(positions, [])
    axes[0].set_xlabel("R²")
    axes[1].set_xlabel("Pearson r")
    for axis in axes:
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("RF full model performance by held-out fire")
    figure.tight_layout()
    save_figure(figure, output / "bc_rf_full_event_performance.png")


def importance_figure(importance: pd.DataFrame, output: Path) -> None:
    data = importance.loc[importance.model == "rf_full"].sort_values("mean_importance")
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(data.feature, data.mean_importance, xerr=data.sd_importance,
              color="#6a51a3", alpha=.9, capsize=2)
    axis.set_xlabel("Mean impurity importance across LOFO folds")
    axis.set_title("RF full feature importance\nDescriptive only; correlated predictors share importance")
    axis.grid(axis="x", alpha=.2)
    figure.tight_layout()
    save_figure(figure, output / "bc_rf_full_feature_importance.png")


def distribution_figure(table_path: Path, output: Path) -> None:
    frame = pd.read_parquet(table_path, columns=["fire_id", "target_dNBR"])
    ids = sorted(frame.fire_id.unique())
    values = [frame.loc[frame.fire_id == x, "target_dNBR"].to_numpy() for x in ids]
    figure, axis = plt.subplots(figsize=(11, 5))
    parts = axis.violinplot(values, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#3182bd")
        body.set_alpha(.65)
    axis.set_xticks(np.arange(1, len(ids) + 1), ids, rotation=45, ha="right")
    axis.set_ylabel("CanLaBS dNBR")
    axis.set_title("Held-out event target distributions")
    axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    save_figure(figure, output / "bc_event_dnbr_distributions.png")


def block_scatter(model_dir: Path, output: Path) -> None:
    frames = []
    for path in sorted(Path(model_dir).glob("folds/*/predictions.parquet")):
        frame = pd.read_parquet(path, columns=["fire_id", "spatial_block_id", "target_dNBR", "pred_rf_full"])
        frames.append(frame.groupby(["fire_id", "spatial_block_id"], as_index=False)
                      [["target_dNBR", "pred_rf_full"]].mean())
    data = pd.concat(frames, ignore_index=True)
    figure, axis = plt.subplots(figsize=(6.5, 6))
    for fire_id, group in data.groupby("fire_id"):
        axis.scatter(group.target_dNBR, group.pred_rf_full, s=6, alpha=.28, label=fire_id)
    limits = [-.3, 1.15]
    axis.plot(limits, limits, "k--", lw=1)
    axis.set(xlim=limits, ylim=limits,
             xlabel="Observed block-mean dNBR", ylabel="Predicted block-mean dNBR",
             title="RF full predictions at 300 m block scale")
    axis.legend(ncol=2, fontsize=7, frameon=False)
    axis.grid(alpha=.15)
    figure.tight_layout()
    save_figure(figure, output / "bc_rf_full_block_scatter.png")


def event_maps(model_dir: Path, events_dir: Path, output: Path) -> None:
    map_dir = output / "event_maps"
    map_dir.mkdir(exist_ok=True)
    for path in sorted(Path(model_dir).glob("folds/*/predictions.parquet")):
        frame = pd.read_parquet(path, columns=["fire_id", "row", "col", "target_dNBR", "pred_rf_full"])
        fire_id = str(frame.fire_id.iloc[0])
        with rasterio.open(events_dir / fire_id / "canlabs_dnbr.tif") as dataset:
            shape = dataset.shape
        observed = np.full(shape, np.nan, dtype="float32")
        predicted = observed.copy()
        rows, cols = frame.row.to_numpy(), frame.col.to_numpy()
        observed[rows, cols] = frame.target_dNBR
        predicted[rows, cols] = frame.pred_rf_full
        arrays = [observed, predicted, predicted - observed]
        titles = ["Observed CanLaBS dNBR", "LOFO RF prediction", "Prediction error"]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for index, (axis, values, title) in enumerate(zip(axes, arrays, titles)):
            image = axis.imshow(values, cmap="RdYlGn_r" if index < 2 else "RdBu_r",
                                vmin=-.2 if index < 2 else -.5,
                                vmax=1.0 if index < 2 else .5)
            axis.set_title(title)
            axis.set_axis_off()
            figure.colorbar(image, ax=axis, fraction=.046, pad=.03)
        figure.suptitle(f"Held-out fire {fire_id}")
        figure.tight_layout()
        save_figure(figure, map_dir / f"{fire_id}_rf_full_map.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.analysis_dir / "event_macro_summary_with_ci.csv")
    pixel = pd.read_csv(args.analysis_dir / "event_pixel_metrics.csv")
    importance = pd.read_csv(args.analysis_dir / "feature_importance_summary.csv")
    macro_figure(summary, args.output_dir)
    event_figure(pixel, args.output_dir)
    importance_figure(importance, args.output_dir)
    distribution_figure(args.table, args.output_dir)
    block_scatter(args.model_dir, args.output_dir)
    event_maps(args.model_dir, args.events_dir, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
