"""YAML experiment configuration loading.

Every numeric choice that affects a reported result (target CRS, scene
identifiers, spatial block size, Random Forest hyper-parameters, frozen dNBR
thresholds) lives in ``configs/*.yaml`` rather than in code, so a result can be
traced to a single versioned file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["PROJECT_ROOT", "CONFIG_DIR", "load_config", "resolve_config_path"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_config_path(name: str | Path) -> Path:
    """Resolve a configuration reference to a file path.

    Accepts a bare name (``camp_fire``), a file name (``camp_fire.yaml``) or an
    explicit path. Bare names and file names are looked up in ``configs/``.
    """
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    if candidate.suffix not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_file():
        return candidate
    in_config_dir = CONFIG_DIR / candidate.name
    if in_config_dir.is_file():
        return in_config_dir
    raise FileNotFoundError(f"Configuration not found: {name} (looked in {CONFIG_DIR})")


def load_config(name: str | Path) -> dict[str, Any]:
    """Load a YAML configuration as a plain dictionary."""
    path = resolve_config_path(name)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    data.setdefault("_config_path", str(path.name))
    return data
