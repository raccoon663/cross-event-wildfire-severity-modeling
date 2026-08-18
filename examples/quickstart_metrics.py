#!/usr/bin/env python3
"""Offline quick-start: load a config and exercise the metric helpers.

This example needs no network access and no GDAL. It shows the three building
blocks the experiments reuse, so you can inspect the scoring conventions before
touching the real pipelines:

1. ``src.config.load_config`` — every experiment reads its settings from YAML.
2. ``src.models.dnbr_rule.classify_dnbr`` — the operational threshold rule.
3. ``src.validation.metrics`` — the classification and regression metrics.

Usage:
    python examples/quickstart_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repository root importable when run as `python examples/quickstart_metrics.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.models.dnbr_rule import classify_dnbr
from src.validation.metrics import (
    classification_metrics,
    regression_metrics,
    top_quartile_metrics,
)


def main() -> None:
    # 1) Configuration
    camp = load_config("camp_fire_30m")
    bc = load_config("bc_lofo")
    print(f"[config] Camp resolution: {camp['resolution_m']} m, "
          f"spatial block: {camp['spatial_block_m']} m, "
          f"RF trees: {camp['rf']['n_estimators']}")
    print(f"[config] BC events: {len(bc['events']['selected'])}; "
          f"block: {bc['sampling']['block_size_m']} m; "
          f"forbidden: {len(bc['forbidden_features'])} terms")

    # 2) dNBR threshold rule with Camp's own MTBS thresholds
    rng = np.random.default_rng(0)
    dnbr = rng.uniform(-0.2, 1.0, 200_000)
    classes = classify_dnbr(dnbr, camp["frozen_camp_dnbr_thresholds"] if "frozen_camp_dnbr_thresholds" in camp
                            else [0.060, 0.301, 0.570])
    truth = classify_dnbr(dnbr, [0.05, 0.30, 0.60])
    print(f"[rule] dNBR rule on 200k synthetic pixels: "
          f"class counts = {np.bincount(classes)[1:].tolist()}")

    # 3) Metrics, classification and regression
    cls = classification_metrics(truth, classes)
    print(f"[metrics] classification Macro-F1 = {cls['macro_f1']:.4f}, "
          f"balanced accuracy = {cls['balanced_accuracy']:.4f}")
    y = rng.normal(0.3, 0.2, 50_000)
    p = 0.55 * y + rng.normal(0, 0.18, 50_000)
    reg = regression_metrics(y, p)
    top = top_quartile_metrics(y, p)
    print(f"[metrics] regression RMSE = {reg['rmse']:.4f}, R2 = {reg['r2']:.4f}, "
          f"Pearson r = {reg['pearson_r']:.4f}, top25 IoU = {top['top25_iou']:.4f}")


if __name__ == "__main__":
    main()
