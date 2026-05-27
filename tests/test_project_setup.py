from ai_melt.config import load_config
from ai_melt.paths import ensure_project_directories


def test_config_loads() -> None:
    config = load_config()
    assert config["project"]["name"] == "ai-melt-research-pipeline"


def test_project_directories_can_be_created() -> None:
    ensure_project_directories()
