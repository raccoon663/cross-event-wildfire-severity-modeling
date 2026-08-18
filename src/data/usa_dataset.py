"""Dataset assembly for the United States experiment.

Stage 1 - Camp resolution ablation
----------------------------------
Two Landsat Collection 2 Level-2 scenes, the exact MTBS pre- and post-fire
acquisitions named in ``configs/camp_fire.yaml``, are QA-masked, scaled to
surface reflectance and stacked into the 13-band feature cube on a pixel-aligned
grid at 30 m and 90 m. MTBS thematic severity is the reference. Five-fold,
5 km-block grouped cross-validation produces the reported out-of-fold metrics
together with the frozen official dNBR-threshold baseline.

Stage 2 - Camp-to-Carr external transfer
----------------------------------------
The Camp-trained Random Forest, plus the dNBR threshold rule frozen from Camp's
own MTBS thresholds, is applied *without any Carr tuning* to the Carr event. The
Carr official thresholds appear only as a descriptive reference and are never
part of the comparison, because a Calibrated rule fitted on the test event would
not be an external test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rasterio

from ..config import load_config
from ..data.landsat import USA_BANDS, read_scene
from ..data.mtbs import event_dnbr_thresholds, event_feature, severity_item
from ..features.usa_stack import USA_FEATURE_NAMES, build_usa_feature_stack, dnbr_column
from ..models.dnbr_rule import classify_dnbr
from ..models.rf_classifier import SEVERITY_CLASSES, balanced_class_subsample, build_classifier
from ..preprocessing.grid import event_grid, rasterize_geometry
from ..preprocessing.reproject import warp_asset_to_grid
from ..utils.provenance import run_environment
from ..validation.metrics import classification_metrics
from ..validation.spatial_cv import grouped_out_of_fold_predictions

__all__ = [
    "scene_from_item",
    "prepare_camp_features",
    "run_camp_resolution_ablation",
    "run_camp_to_carr_transfer",
]

_OUTPUT_PROFILE = {"driver": "GTiff", "dtype": "float32", "nodata": -9999.0,
                   "compress": "deflate", "tiled": True}


def scene_from_item(item: dict[str, Any], transform, width: int, height: int, cfg: dict) -> np.ndarray:
    """QA-masked surface reflectance cube for one Landsat item on the event grid."""
    return read_scene(
        item, transform, width, height,
        crs=cfg["target_crs"], resolution_m=cfg["resolution_m"], bands=USA_BANDS,
    )


def _load_mtbs_labels(cfg: dict, transform, width, height, resolution: int) -> np.ndarray:
    item = severity_item(cfg["fire_year"], resolution)
    return warp_asset_to_grid(
        item["assets"]["burn-severity"]["href"], transform, width, height,
        cfg["target_crs"], resolution, nearest=True,
    ).astype("uint8")


def _valid_pixels(feature_cube: np.ndarray, labels: np.ndarray, event_mask: np.ndarray):
    valid = (
        event_mask
        & np.isfinite(feature_cube).all(axis=0)
        & np.isin(labels, SEVERITY_CLASSES)
    )
    rows, cols = np.where(valid)
    return valid, rows, cols


def _spatial_groups(cols, rows, transform, block_size_m: int) -> np.ndarray:
    """Group ids for GroupKFold: 5 km blocks of projected pixel centres.

    Delegates to :func:`src.validation.spatial_blocks.block_group_ids` so the
    block definition is shared across experiments.
    """
    from ..validation.spatial_blocks import block_group_ids

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    return block_group_ids(xs, ys, block_size_m)


def _write_feature_raster(path: Path, cube: np.ndarray, transform, height: int, width: int, crs: str) -> None:
    profile = dict(_OUTPUT_PROFILE)
    profile.update(height=height, width=width, count=len(USA_FEATURE_NAMES), crs=crs, transform=transform)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.where(np.isfinite(cube), cube, -9999).astype("float32"))
        destination.descriptions = tuple(USA_FEATURE_NAMES)


def _write_prediction_raster(path: Path, prediction: np.ndarray, rows, cols, transform, height, width, crs: str, tags: dict) -> None:
    profile = dict(_OUTPUT_PROFILE)
    profile.update(height=height, width=width, count=1, dtype="uint8", nodata=255, crs=crs, transform=transform)
    prediction_map = np.full((height, width), 255, dtype="uint8")
    prediction_map[rows, cols] = prediction
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(prediction_map, 1)
        destination.update_tags(**tags)


def run_camp_resolution_ablation(
    config_name: str,
    output_root: Path,
    data_root: Path | None = None,
    n_jobs: int = -1,
) -> dict:
    """Run one resolution of the Camp ablation (30 m or 90 m).

    Writes the feature raster (under ``data_root``), the out-of-fold prediction
    raster and ``metrics_{resolution}m.json`` under ``output_root``.
    """
    output_root = Path(output_root)
    cfg = load_config(config_name)
    resolution = int(cfg["resolution_m"])
    data_root = Path(data_root) if data_root is not None else output_root / "data"

    started = time.perf_counter()
    process = psutil.Process()
    event = event_feature(cfg["fire_id"])
    properties = event["properties"]
    transform, width, height, _ = event_grid(event, resolution, cfg["target_crs"])
    pre = scene_from_item(load_scene(cfg["pre_item_id"]), transform, width, height, cfg)
    post = scene_from_item(load_scene(cfg["post_item_id"]), transform, width, height, cfg)
    cube = build_usa_feature_stack(pre, post)
    labels = _load_mtbs_labels(cfg, transform, width, height, resolution)
    event_mask = rasterize_geometry(event["geometry"], transform, width, height, cfg["target_crs"])

    valid, rows, cols = _valid_pixels(cube, labels, event_mask)
    design = cube[:, rows, cols].T
    target = labels[rows, cols]
    groups = _spatial_groups(cols, rows, transform, cfg["spatial_block_m"])

    rf = cfg["rf"]
    out_of_fold, folds = grouped_out_of_fold_predictions(
        design, target, groups, n_splits=cfg["n_splits"],
        model_factory=lambda fold: build_classifier(rf, cfg["random_seed"] + fold, n_jobs=n_jobs),
        subsample=lambda y, train_index, fold: balanced_class_subsample(
            y, train_index, rf["max_train_pixels_per_class"], cfg["random_seed"] + fold
        ),
        verbose_prefix=f"{resolution}m",
    )
    thresholds = event_dnbr_thresholds(properties)
    baseline = classify_dnbr(dnbr_column(design), thresholds)
    fold_scores = [fold["macro_f1"] for fold in folds]

    result = {
        "resolution_m": resolution, "fire_id": cfg["fire_id"],
        "fire_name": cfg.get("fire_name"), "event_properties": properties,
        "pre_item_id": cfg["pre_item_id"], "post_item_id": cfg["post_item_id"],
        "feature_names": USA_FEATURE_NAMES,
        "n_valid_pixels": int(len(target)),
        "n_groups": int(len(np.unique(groups))),
        "class_counts": {str(c): int((target == c).sum()) for c in SEVERITY_CLASSES},
        "random_forest_oof": classification_metrics(target, out_of_fold),
        "dnbr_official_event_thresholds": thresholds,
        "dnbr_baseline": classification_metrics(target, baseline),
        "folds": folds,
        "fold_macro_f1_mean": float(np.mean(fold_scores)),
        "fold_macro_f1_std": float(np.std(fold_scores, ddof=1)),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes_observed": int(process.memory_info().rss),
        "environment": run_environment(),
        "interpretation": "Agreement with MTBS thematic labels, not independent field validation.",
    }

    data_dir = data_root / "camp_resolution_ablation" / f"{resolution}m"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_feature_raster(data_dir / f"camp_features_{resolution}m.tif", cube, transform, height, width, cfg["target_crs"])
    _write_prediction_raster(
        output_root / f"camp_oof_prediction_{resolution}m.tif", out_of_fold, rows, cols,
        transform, height, width, cfg["target_crs"],
        {"fire_id": cfg["fire_id"], "evaluation": "5 km block OOF"},
    )
    (output_root / f"metrics_{resolution}m.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_camp_to_carr_transfer(
    config_name: str,
    output_root: Path,
    camp_feature_path: Path,
    camp_metrics_path: Path,
    n_jobs: int = -1,
) -> dict:
    """Train on Camp, apply unchanged to the Carr event, and score it.

    ``camp_feature_path`` is the feature raster written by the 30 m ablation run
    (used as the training source) and ``camp_metrics_path`` its metrics file
    (used for the performance-retention ratio).
    """
    output_root = Path(output_root)
    cfg = load_config(config_name)
    resolution = int(cfg["resolution_m"])
    started = time.perf_counter()
    process = psutil.Process()

    camp_event = event_feature(cfg["train_fire_id"])
    carr_event = event_feature(cfg["test_fire_id"])
    with rasterio.open(camp_feature_path) as source:
        camp_cube = source.read().astype("float32")
        camp_cube[camp_cube < -100] = np.nan
        camp_transform, camp_width, camp_height = source.transform, source.width, source.height
    camp_label, camp_mask = _aligned_label_and_mask(camp_event, camp_transform, camp_width, camp_height, cfg)
    camp_valid = camp_mask & np.isfinite(camp_cube).all(axis=0) & np.isin(camp_label, SEVERITY_CLASSES)
    camp_design = camp_cube[:, camp_valid].T
    camp_target = camp_label[camp_valid]

    carr_transform, carr_width, carr_height, _ = event_grid(carr_event, resolution, cfg["target_crs"])
    pre = scene_from_item(load_scene(cfg["pre_item_id"]), carr_transform, carr_width, carr_height, cfg)
    post = scene_from_item(load_scene(cfg["post_item_id"]), carr_transform, carr_width, carr_height, cfg)
    carr_cube = build_usa_feature_stack(pre, post)
    carr_label, carr_mask = _aligned_label_and_mask(carr_event, carr_transform, carr_width, carr_height, cfg)
    carr_valid, carr_rows, carr_cols = _valid_pixels(carr_cube, carr_label, carr_mask)
    carr_design = carr_cube[:, carr_rows, carr_cols].T
    carr_target = carr_label[carr_rows, carr_cols]

    rf = cfg["rf"]
    training = balanced_class_subsample(
        camp_target, np.arange(len(camp_target)),
        rf["max_train_pixels_per_class"], cfg["random_seed"],
    )
    model = build_classifier(rf, cfg["random_seed"], n_jobs=n_jobs)
    model.fit(camp_design[training], camp_target[training])
    rf_prediction = model.predict(carr_design).astype("uint8")

    frozen = cfg["frozen_camp_dnbr_thresholds"]
    frozen_rule = classify_dnbr(dnbr_column(carr_design), frozen)
    properties = carr_event["properties"]
    carr_official = event_dnbr_thresholds(properties)
    carr_rule = classify_dnbr(dnbr_column(carr_design), carr_official)

    camp_metrics = json.loads(Path(camp_metrics_path).read_text(encoding="utf-8"))["random_forest_oof"]
    shift = []
    for index, name in enumerate(USA_FEATURE_NAMES):
        a, b = camp_design[:, index], carr_design[:, index]
        pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
        shift.append({
            "feature": name,
            "camp_mean": float(np.mean(a)), "carr_mean": float(np.mean(b)),
            "standardized_mean_difference": float((np.mean(b) - np.mean(a)) / pooled) if pooled else None,
        })

    rf_metrics = classification_metrics(carr_target, rf_prediction, include_high_recall=True)
    result = {
        "evaluation": "Camp-only training; Carr complete event external test; no Carr tuning",
        "resolution_m": resolution,
        "train_fire_id": cfg["train_fire_id"], "test_fire_id": cfg["test_fire_id"],
        "carr_event_properties": properties,
        "pre_item_id": cfg["pre_item_id"], "post_item_id": cfg["post_item_id"],
        "n_camp_train_sampled": int(len(training)), "n_carr_test_pixels": int(len(carr_target)),
        "carr_class_counts": {str(c): int((carr_target == c).sum()) for c in SEVERITY_CLASSES},
        "camp_trained_rf": rf_metrics,
        "frozen_camp_dnbr_thresholds": frozen,
        "frozen_camp_dnbr": classification_metrics(carr_target, frozen_rule),
        "carr_official_thresholds_reference_only": carr_official,
        "carr_calibrated_dnbr_reference_only": classification_metrics(carr_target, carr_rule),
        "camp_oof_macro_f1": camp_metrics["macro_f1"],
        "rf_performance_retention": rf_metrics["macro_f1"] / camp_metrics["macro_f1"],
        "feature_distribution_shift": shift,
        "runtime_seconds": float(time.perf_counter() - started),
        "rss_bytes_observed": int(process.memory_info().rss),
        "environment": run_environment(),
        "interpretation": (
            "Agreement with MTBS thematic labels, not field validation. Carr official "
            "thresholds are descriptive only and not part of the external model comparison."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "camp_to_carr_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_feature_raster(
        output_root / "carr_features_30m.tif", carr_cube, carr_transform,
        carr_height, carr_width, cfg["target_crs"],
    )
    _write_prediction_raster(
        output_root / "carr_camp_trained_rf_prediction.tif", rf_prediction, carr_rows, carr_cols,
        carr_transform, carr_height, carr_width, cfg["target_crs"],
        {"evaluation": "Camp-trained RF, untouched Carr event"},
    )
    return result


def _aligned_label_and_mask(event, transform, width, height, cfg):
    """Warp the MTBS severity raster and rasterise the event polygon on one grid."""
    item = severity_item(cfg["fire_year"])
    label = warp_asset_to_grid(
        item["assets"]["burn-severity"]["href"], transform, width, height,
        cfg["target_crs"], cfg["resolution_m"], nearest=True,
    ).astype("uint8")
    mask = rasterize_geometry(event["geometry"], transform, width, height, cfg["target_crs"])
    return label, mask


def load_scene(item_id: str) -> dict[str, Any]:
    """Fetch one Landsat STAC item (thin wrapper around the data layer)."""
    from ..data.stac import landsat_item
    return landsat_item(item_id)
