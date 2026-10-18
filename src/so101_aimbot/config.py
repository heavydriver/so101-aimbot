"""Load YAML configs from the repo `configs/` directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load `configs/{name}.yaml` (name may include or omit the `.yaml` suffix)."""
    path = CONFIG_DIR / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
