"""Leave-one-fire-out validation for the Canadian experiment.

Protocol
--------
For each of the twelve fires in turn:

1. exclude that fire completely;
2. draw an equal, distribution-preserving pixel quota from every remaining fire;
3. fit all five models on that sample;
4. derive the high-severity threshold as the *training* upper quartile, so the
   threshold never sees the held-out fire;
5. score every valid pixel of the held-out fire and store the predictions.

Every preprocessing decision that depends on data (sampling, threshold,
standardisation inside the linear pipeline) happens inside the fold. The only
quantities fixed outside are the spatial block assignment and the Random Forest
hyper-parameters, both of which are independent of the target.

Folds are written to disk with a completion marker, so an interrupted run resumes
instead of recomputing finished fires.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .metrics import regression_metrics_with_high

__all__ = ["fold_directory", "is_complete", "run_leave_one_fire_out", "aggregate_fold_metrics"]

PREDICTION_ID_COLUMNS = [
    "fire_id", "pixel_id", "row", "col", "x", "y", "spatial_block_id", "target_dNBR",
]


def fold_directory(output_dir: Path, fire_id: str) -> Path:
    """Directory holding one fold's predictions, metrics and completion marker."""
    return Path(output_dir) / "folds" / fire_id


def is_complete(output_dir: Path, fire_id: str) -> bool:
    """True when a fold has a completion marker and can be skipped."""
    return (fold_directory(output_dir, fire_id) / "fold_complete.json").exists()


def aggregate_fold_metrics(output_dir: Path, fire_ids: Iterable[str]) -> None:
    """Concatenate finished folds into event and macro summaries.

    Silently returns when a fold is still missing, which keeps the function safe to
    call after every fold in a resumable run.
    """
    output_dir = Path(output_dir)
    metric_files = [fold_directory(output_dir, fire_id) / "metrics.csv" for fire_id in fire_ids]
    if not all(path.exists() for path in metric_files):
        return
    metrics_frame = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    metrics_frame.to_csv(output_dir / "lofo_event_metrics.csv", index=False)
    summary = metrics_frame.groupby("model", as_index=False).agg(
        events=("fire_id", "nunique"),
        mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"), mean_r2=("r2", "mean"), median_r2=("r2", "median"),
        mean_pearson_r=("pearson_r", "mean"), mean_high_iou=("high_iou", "mean"),
        mean_high_f1=("high_f1", "mean"),
    )
    summary.to_csv(output_dir / "lofo_macro_summary.csv", index=False)


def run_leave_one_fire_out(
    table: pd.DataFrame,
    output_dir: Path,
    model_specs: list[tuple[str, list[str]]],
    sampler: Callable[[pd.DataFrame, str, int], pd.DataFrame],
    fitter: Callable[[str, list[str], pd.DataFrame, pd.DataFrame], tuple],
    seed: int = 42,
    verbose: bool = True,
) -> list[str]:
    """Execute the full leave-one-fire-out loop.

    Parameters
    ----------
    table
        Complete pixel table containing ``fire_id`` and ``target_dNBR``.
    output_dir
        Destination for per-fold artefacts and aggregated summaries.
    model_specs
        ``(model_name, feature_list)`` pairs, evaluated in order.
    sampler
        ``(table, test_fire, fold_seed) -> training_frame``.
    fitter
        ``(model_name, features, train, test) -> (prediction, model)``.
    seed
        Base seed; each fold uses ``seed + fold_index * 100``.

    Returns
    -------
    list[str]
        The sorted fire identifiers that define the fold order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fire_ids = sorted(table["fire_id"].unique())

    for fold_index, test_fire in enumerate(fire_ids):
        directory = fold_directory(output_dir, test_fire)
        marker = directory / "fold_complete.json"
        if marker.exists():
            if verbose:
                print(json.dumps({"fire_id": test_fire, "status": "cached"}), flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        train = sampler(table, test_fire, seed + fold_index * 100)
        test = table.loc[table["fire_id"] == test_fire].copy()
        high_threshold = float(train["target_dNBR"].quantile(0.75))
        predictions = test[PREDICTION_ID_COLUMNS].copy()
        rows, importances = [], []
        for model_name, features in model_specs:
            model_started = time.perf_counter()
            prediction, model = fitter(model_name, features, train, test)
            predictions[f"pred_{model_name}"] = prediction
            row = {
                "fire_id": test_fire, "model": model_name,
                "n_train": len(train), "n_test": len(test),
                "training_events": int(train["fire_id"].nunique()),
                "seconds": time.perf_counter() - model_started,
            }
            row.update(regression_metrics_with_high(
                test["target_dNBR"].to_numpy(), prediction, high_threshold
            ))
            rows.append(row)
            if model_name.startswith("rf_") and model is not None:
                importances.extend(
                    {"fire_id": test_fire, "model": model_name, "feature": feature,
                     "importance": float(value)}
                    for feature, value in zip(features, model.feature_importances_)
                )
        predictions.to_parquet(directory / "predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
        pd.DataFrame(importances).to_csv(directory / "feature_importance.csv", index=False)
        completion = {
            "fire_id": test_fire, "seconds": round(time.perf_counter() - started, 3),
            "n_train": len(train), "n_test": len(test),
            "models": [name for name, _ in model_specs],
        }
        marker.write_text(json.dumps(completion, indent=2), encoding="utf-8")
        aggregate_fold_metrics(output_dir, fire_ids)
        if verbose:
            print(json.dumps(completion), flush=True)

    aggregate_fold_metrics(output_dir, fire_ids)
    return fire_ids
