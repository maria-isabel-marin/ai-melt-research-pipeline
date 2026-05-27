from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""
    if config_path is None:
        config_path = get_project_root() / "config" / "settings.yaml"

    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_config_section(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return a nested configuration section, or an empty dict when absent."""
    section: Any = config
    for key in keys:
        if not isinstance(section, dict):
            return {}
        section = section.get(key, {})
    return section if isinstance(section, dict) else {}
