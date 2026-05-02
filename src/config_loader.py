"""
config_loader.py
────────────────
Shared utility — loads config/config.yaml from the project root.
All scripts import this instead of reading the file themselves.
"""

import os
from pathlib import Path
import yaml


def load_config() -> dict:
    """
    Load config/config.yaml relative to the project root.
    Works regardless of which directory the script is called from.
    """
    # Walk up from this file's location to find config/config.yaml
    here = Path(__file__).resolve().parent
    # src/ → project root
    root = here.parent
    config_path = root / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
