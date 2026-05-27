from __future__ import annotations

import sys
from pathlib import Path


def ensure_highway_env_importable() -> None:
    """Prefer an installed highway_env, with a local source-tree fallback for dev."""
    try:
        __import__("highway_env")
        return
    except ModuleNotFoundError:
        source_root = Path(__file__).resolve().parents[2] / "HighwayEnv-sourcecode"
        if source_root.exists():
            sys.path.insert(0, str(source_root))
            __import__("highway_env")
            return
        raise

