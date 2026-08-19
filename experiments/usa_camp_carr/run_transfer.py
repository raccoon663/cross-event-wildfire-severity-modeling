#!/usr/bin/env python3
"""United States experiment, stage 2: Camp-trained model applied to the Carr event.

The Random Forest and the dNBR threshold rule are trained or frozen exclusively
on Camp Fire data and applied without modification to the complete Carr event.
The Carr event's own official MTBS thresholds are written to the metrics file
under ``*_reference_only`` and are never part of the model comparison.

Run ``experiments/usa_camp_carr/run_ablation.py`` first; the transfer reads the
30 m Camp feature raster and metrics it produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from src.config import PROJECT_ROOT
from src.data.usa_dataset import run_camp_to_carr_transfer

RESULTS = PROJECT_ROOT / "experiments" / "usa_camp_carr" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--camp-features", type=Path,
                        default=RESULTS / "data" / "camp_resolution_ablation" / "30m" / "camp_features_30m.tif")
    parser.add_argument("--camp-metrics", type=Path,
                        default=RESULTS / "metrics_30m.json")
    args = parser.parse_args()

    result = run_camp_to_carr_transfer(
        "carr_fire", args.output,
        camp_feature_path=args.camp_features,
        camp_metrics_path=args.camp_metrics,
        n_jobs=-1,
    )
    summary = {
        "rf_macro_f1": result["camp_trained_rf"]["macro_f1"],
        "frozen_camp_dnbr_macro_f1": result["frozen_camp_dnbr"]["macro_f1"],
        "carr_reference_dnbr_macro_f1": result["carr_calibrated_dnbr_reference_only"]["macro_f1"],
        "performance_retention_vs_camp_oof": result["rf_performance_retention"],
    }
    print(json_dumps(summary, indent=2))


def json_dumps(value, **kwargs):
    import json
    return json.dumps(value, indent=2, ensure_ascii=False, **kwargs)


if __name__ == "__main__":
    main()
