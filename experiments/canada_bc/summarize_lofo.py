#!/usr/bin/env python3
"""British Columbia experiment: summarise LOFO predictions.

Aggregates per-fold predictions into pixel-level and 300 m block-level event
metrics, then produces the event-macro summary with 95 percent event-bootstrap
intervals (10 000 replicates) and the feature-importance summary.

Usage
-----
    python experiments/canada_bc/summarize_lofo.py \
        --model-dir <results>/lofo --output-dir <results>/analysis \
        [--table <events_root>/bc_prefire_feature_table.parquet]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import pandas as pd

from src.validation.bootstrap import bootstrap_event_mean
from src.validation.metrics import regression_metrics, top_quartile_metrics


def summarize_folds(model_dir: Path, output_dir: Path, seed: int = 42, repetitions: int = 10000) -> None:
    """Compute pixel/block event metrics and the event-macro summary with CI."""
    folds = sorted(Path(model_dir).glob("folds/*/predictions.parquet"))
    if len(folds) < 8:
        raise ValueError("Expected at least eight completed LOFO folds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_rows, block_rows = [], []
    for path in folds:
        frame = pd.read_parquet(path)
        fire_id = str(frame["fire_id"].iloc[0])
        prediction_columns = [column for column in frame.columns if column.startswith("pred_")]
        for prediction_column in prediction_columns:
            model = prediction_column.removeprefix("pred_")
            y = frame["target_dNBR"].to_numpy()
            p = frame[prediction_column].to_numpy()
            row = {"fire_id": fire_id, "model": model, "n_pixels": len(frame)}
            row.update(regression_metrics(y, p))
            row.update(top_quartile_metrics(y, p))
            pixel_rows.append(row)
            blocks = frame.groupby("spatial_block_id", as_index=False)[
                ["target_dNBR", prediction_column]
            ].mean()
            block_row = {"fire_id": fire_id, "model": model, "n_blocks": len(blocks)}
            block_row.update(regression_metrics(
                blocks["target_dNBR"].to_numpy(), blocks[prediction_column].to_numpy()
            ))
            block_rows.append(block_row)

    pixel = pd.DataFrame(pixel_rows)
    blocks = pd.DataFrame(block_rows)
    pixel.to_csv(output_dir / "event_pixel_metrics.csv", index=False)
    blocks.to_csv(output_dir / "event_block_metrics.csv", index=False)

    summaries = []
    for model, group in pixel.groupby("model"):
        row = {"model": model, "events": int(group["fire_id"].nunique())}
        for column in ["rmse", "mae", "r2", "pearson_r", "top25_iou", "top25_f1"]:
            mean, low, high = bootstrap_event_mean(
                group[column], seed=seed, repetitions=repetitions
            )
            row.update({f"mean_{column}": mean,
                        f"{column}_ci_low": low, f"{column}_ci_high": high})
        block_group = blocks.loc[blocks["model"] == model]
        for column in ["rmse", "mae", "r2", "pearson_r"]:
            mean, low, high = bootstrap_event_mean(
                block_group[column], seed=seed, repetitions=repetitions
            )
            row.update({f"block_mean_{column}": mean,
                        f"block_{column}_ci_low": low, f"block_{column}_ci_high": high})
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("mean_rmse")
    summary.to_csv(output_dir / "event_macro_summary_with_ci.csv", index=False)

    importance_files = sorted(Path(model_dir).glob("folds/*/feature_importance.csv"))
    if importance_files:
        importance = pd.concat([pd.read_csv(path) for path in importance_files], ignore_index=True)
        importance_summary = importance.groupby(["model", "feature"], as_index=False).agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
        )
        importance_summary.to_csv(output_dir / "feature_importance_summary.csv", index=False)

    metadata = {
        "events": len(folds),
        "pixel_metrics": "all valid pixels in each held-out fire",
        "block_metrics": "means within preassigned 300 m blocks",
        "top25_metrics": "event-relative observed and predicted upper quartiles; evaluation only",
        "confidence_intervals": f"95% percentile bootstrap over events, {repetitions} repetitions",
    }
    (output_dir / "summary_method.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summarize_folds(args.model_dir, args.output_dir,
                    seed=args.seed, repetitions=args.bootstrap_repetitions)


if __name__ == "__main__":
    main()
