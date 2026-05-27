from pathlib import Path

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
