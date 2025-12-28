from __future__ import annotations

import os
from pathlib import Path


_ROBOSTL_DIR_NAME = "robostl"


def find_robostl_root(start: Path | None = None) -> Path:
    """Find the RoboSTL root by walking up from the given start."""
    if start is None:
        start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if parent.name == _ROBOSTL_DIR_NAME:
            return parent
    raise FileNotFoundError(f"{_ROBOSTL_DIR_NAME} not found from {start}")


def get_robostl_root() -> Path:
    """Resolve RoboSTL root from env or filesystem."""
    env_root = os.getenv("ROBOSTL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return find_robostl_root()
