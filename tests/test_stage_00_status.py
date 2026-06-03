from pathlib import Path

import pandas as pd

from ai_melt.status import (
    format_status_report,
    get_stage_00_status_report,
    next_recommended_command,
    reset_stage_status,
    update_stage_status,
)


def stage_00_config(tmp_path: Path) -> dict:
    return {
        "stage_00": {
            "intermediate_outputs": {
                "discovered_files_csv": "data/interim/n0_discovered_files.csv",
                "pages_raw_parquet": "data/interim/n0_pages_raw.parquet",
                "pages_with_chapters_parquet": (
                    "data/interim/n0_pages_with_chapters.parquet"
                ),
                "pages_clean_parquet": "data/interim/n0_pages_clean.parquet",
                "footnotes_csv": "data/interim/n0_footnotes.csv",
                "sentences_parquet": "data/interim/n0_sentences.parquet",
                "sentences_annotated_parquet": (
                    "data/interim/n0_sentences_annotated.parquet"
                ),
            },
            "outputs": {
                "corpus_csv": "data/processed/n0_corpus.csv",
                "corpus_parquet": "data/processed/n0_corpus.parquet",
                "metadata_json": "data/processed/n0_metadata.json",
                "footnotes_csv": "outputs/tables/n0_footnotes.csv",
                "document_summary_csv": "outputs/tables/n0_summary_by_document.csv",
                "chapter_summary_csv": "outputs/tables/n0_summary_by_chapter.csv",
                "cleaning_inspection_csv": (
                    "outputs/tables/n0_cleaning_inspection.csv"
                ),
                "footnotes_inspection_csv": (
                    "outputs/tables/n0_footnotes_inspection.csv"
                ),
            },
        }
    }


def touch(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_status_report_uses_json_and_expected_outputs(tmp_path: Path) -> None:
    status_path = tmp_path / "outputs" / "logs" / "stage_00_status.json"
    output = touch(tmp_path, "data/interim/n0_discovered_files.csv")
    result = pd.DataFrame({"archivo": ["doc.txt"], "has_configured_metadata": [True]})
    result.attrs["outputs"] = [output]

    update_stage_status(
        "discover",
        result,
        command="python scripts/00_ingest_corpus.py --step discover",
        path=status_path,
    )

    report = get_stage_00_status_report(
        stage_00_config(tmp_path), path=status_path, root=tmp_path
    )

    assert report["rows"][0]["step"] == "discover"
    assert report["rows"][0]["status"] == "completed"
    assert report["next_step"] == "extract"
    assert (
        next_recommended_command(
            stage_00_config(tmp_path), path=status_path, root=tmp_path
        )
        == "python scripts/00_ingest_corpus.py --step extract"
    )


def test_status_report_warns_when_completed_output_is_missing(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "outputs" / "logs" / "stage_00_status.json"
    missing_output = tmp_path / "data" / "interim" / "n0_discovered_files.csv"
    result = pd.DataFrame({"archivo": ["doc.txt"]})
    result.attrs["outputs"] = [missing_output]

    update_stage_status("discover", result, path=status_path)

    report = get_stage_00_status_report(
        stage_00_config(tmp_path), path=status_path, root=tmp_path
    )
    formatted = format_status_report(report)

    assert report["rows"][0]["recorded_status"] == "completed"
    assert report["rows"][0]["status"] == "pending"
    assert report["next_step"] == "discover"
    assert "Warnings:" in formatted
    assert "discover is recorded as completed" in formatted


def test_status_report_infers_completed_step_from_existing_outputs(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "outputs" / "logs" / "stage_00_status.json"
    touch(tmp_path, "data/interim/n0_discovered_files.csv")

    report = get_stage_00_status_report(
        stage_00_config(tmp_path), path=status_path, root=tmp_path
    )

    assert report["rows"][0]["recorded_status"] == "pending"
    assert report["rows"][0]["status"] == "completed"
    assert report["next_step"] == "extract"


def test_reset_stage_status_deletes_status_file(tmp_path: Path) -> None:
    status_path = tmp_path / "outputs" / "logs" / "stage_00_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text('{"stages": {}}', encoding="utf-8")

    assert reset_stage_status(status_path) is True
    assert not status_path.exists()
    assert reset_stage_status(status_path) is False
