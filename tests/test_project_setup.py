from ai_melt.config import load_config
from ai_melt.paths import ensure_project_directories


def test_config_loads() -> None:
    config = load_config()
    assert config["project"]["name"] == "ai-melt-research-pipeline"
    assert config["corpus"]["id"] == "CEV-IF-2022"
    assert "stage_00" in config
    assert "stage_01" in config
    assert list(config["stage_01"]["approaches"]) == ["claude", "openai"]
    assert config["stage_01"]["sampling"]["sample_size"] == 50
    assert config["stage_01"]["human_evaluation"]["sample_per_approach"] == 15
    assert "debug_outputs" in config["stage_00"]
    assert "write_csv_copies" in config["stage_00"]["debug_outputs"]
    assert "stage_00_visualisation" in config
    assert (
        config["stage_00_visualisation"]["histogram_bins"]["corpus_sentence_lengths"]
        == 50
    )
    assert "corpus_overview_csv" in config["stage_00_visualisation"]["table_names"]


def test_project_directories_can_be_created() -> None:
    ensure_project_directories()
