from pathlib import Path
from typing import Any

from ai_melt.config import get_project_root, load_config

ROOT_DIR: Path = get_project_root()
CONFIG = load_config()

DATA_RAW_DIR = ROOT_DIR / CONFIG["paths"]["data_raw"]
DATA_INTERIM_DIR = ROOT_DIR / CONFIG["paths"]["data_interim"]
DATA_PROCESSED_DIR = ROOT_DIR / CONFIG["paths"]["data_processed"]

OUTPUTS_TABLES_DIR = ROOT_DIR / CONFIG["paths"]["outputs_tables"]
OUTPUTS_FIGURES_DIR = ROOT_DIR / CONFIG["paths"]["outputs_figures"]
OUTPUTS_HTML_DIR = ROOT_DIR / CONFIG["paths"]["outputs_html"]
OUTPUTS_FINAL_DIR = ROOT_DIR / CONFIG["paths"]["outputs_final"]


def ensure_project_directories() -> None:
    """Create local data and output directories when they do not exist."""
    directories = [
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        OUTPUTS_TABLES_DIR,
        OUTPUTS_FIGURES_DIR,
        OUTPUTS_HTML_DIR,
        OUTPUTS_FINAL_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def project_path(path_value: str | Path, root: Path | None = None) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root or ROOT_DIR) / path


def configured_path(config: dict[str, Any], *keys: str) -> Path:
    """Resolve a nested path value from the project configuration."""
    value: Any = config
    for key in keys:
        value = value[key]
    return project_path(value)


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the path."""
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
