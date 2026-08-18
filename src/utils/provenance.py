"""Run provenance recorded next to every result file.

Reported numbers are only meaningful if the software stack that produced them is
known, so each experiment writes the interpreter version, the versions of the
scientific libraries and the wall-clock runtime alongside its metrics.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any

__all__ = ["library_versions", "run_environment", "Timer"]


def library_versions() -> dict[str, str]:
    """Collect versions of the libraries that influence numeric output."""
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "rasterio", "pyarrow", "matplotlib"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def run_environment() -> dict[str, Any]:
    """Return a compact, non-identifying description of the runtime."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "machine": platform.machine(),
        "libraries": library_versions(),
    }


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self._started = 0.0

    def __enter__(self) -> "Timer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = time.perf_counter() - self._started
