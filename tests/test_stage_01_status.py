from pathlib import Path

from ai_melt.config import load_config
from ai_melt.status import (
    get_stage_01_status_report,
    get_stage_01_visualisation_status_report,
    update_named_stage_status,
)


def touch(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_stage_01_status_infers_existing_output(tmp_path: Path) -> None:
    config = load_config()
    touch(tmp_path, "outputs/tables/n1_config_summary.json")
    status_path = tmp_path / "outputs/logs/stage_01_status.json"

    report = get_stage_01_status_report(config, path=status_path, root=tmp_path)

    assert report["rows"][0]["status"] == "completed"
    assert report["next_step"] == "load-data"


def test_stage_01_visualisation_status_and_record(tmp_path: Path) -> None:
    config = load_config()
    table = touch(tmp_path, "outputs/tables/n1_viz_load_summary.csv")
    status_path = tmp_path / "outputs/logs/stage_01_visualisation_status.json"
    update_named_stage_status(
        "stage_01_visualisation",
        "load",
        {"table": table, "_summary": {"rows": 2}},
        status_path,
    )

    report = get_stage_01_visualisation_status_report(
        config, path=status_path, root=tmp_path
    )

    assert report["rows"][0]["recorded_status"] == "completed"
    assert report["rows"][0]["status"] == "completed"
