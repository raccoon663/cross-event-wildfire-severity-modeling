#!/usr/bin/env python3
"""British Columbia experiment: leave-one-fire-out regression.

Runs the five-model ladder over the twelve selected fires, resumably. Training
pixels are capped per fire with a within-event target-decile sampling; the
held-out fire is always scored over its complete valid population.

The leakage assertion is enforced here, immediately before any fitting: the
configured feature list must not contain any post-fire term.

Usage
-----
    python experiments/canada_bc/run_lofo.py \
        --table <events_root>/bc_prefire_feature_table.parquet \
        --output-dir <results>/lofo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

from src.config import load_config
from src.features.bc_table import FORBIDDEN_PREDICTORS
from src.models.rf_regressor import (
    FULL_FEATURES,
    MODEL_SPECS,
    fit_predict,
    training_sample,
)
from src.validation.leave_one_fire_out import run_leave_one_fire_out

BC_CONFIG = "bc_lofo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixels-per-fire", type=int, default=15000)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(BC_CONFIG)
    if any(feature in FULL_FEATURES for feature in FORBIDDEN_PREDICTORS):
        raise AssertionError("Forbidden post-fire predictor configured")

    import pandas as pd

    table = pd.read_parquet(args.table)
    fire_ids = sorted(table["fire_id"].unique())
    if len(fire_ids) < config["events"]["minimum_required"]:
        raise ValueError(f"BC LOFO requires at least {config['events']['minimum_required']} events")
    if table[FULL_FEATURES + ["target_dNBR"]].isna().any().any():
        raise ValueError("Missing model values in feature table")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "validation": "leave-one-fire-out",
        "test_scoring": "all valid held-out pixels",
        "training_sampling": "equal pixels per fire, proportional within-event dNBR deciles",
        "pixels_per_training_fire": args.pixels_per_fire, "seed": args.seed,
        "features": {
            "spectral": FULL_FEATURES[:7],
            "structure_terrain": FULL_FEATURES[7:],
            "full": FULL_FEATURES,
        },
        # RF hyper-parameters are recorded exactly as used for training below,
        # so a CLI override cannot desynchronise the saved config from the fit.
        "rf": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
        },
        "note": "RF hyperparameters were fixed before inspecting any held-out-fire result.",
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    fitter = lambda name, features, train, test: fit_predict(  # noqa: E731
        name, features, train, test,
        rf_parameters={"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                       "min_samples_leaf": args.min_samples_leaf,
                       "max_features": args.max_features},
        n_jobs=args.n_jobs, seed=args.seed,
    )
    sampler = lambda table_, test_fire, seed: training_sample(  # noqa: E731
        table_, test_fire, args.pixels_per_fire, seed
    )
    run_leave_one_fire_out(
        table, args.output_dir, MODEL_SPECS,
        sampler=sampler, fitter=fitter, seed=args.seed,
    )


if __name__ == "__main__":
    main()
