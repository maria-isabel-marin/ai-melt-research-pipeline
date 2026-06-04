from pathlib import Path

from ai_melt.status import (
    format_visualisation_status_report,
    get_stage_00_visualisation_status_report,
    next_visualisation_recommended_command,
    reset_stage_status,
    update_stage_00_visualisation_status,
)


def visualisation_config(tmp_path: Path) -> dict:
    return {
        "stage_00_visualisation": {
            "inputs": {"corpus_csv": "data/processed/n0_corpus.csv"},
            "outputs": {
                "tables_dir": "outputs/tables",
                "figures_dir": "outputs/figures",
                "html_dir": "outputs/html",
            },
            "figure_names": {
                "sentences_by_volume": "viz_oraciones_por_volumen.png",
                "words_by_volume": "viz_palabras_por_volumen.png",
                "sentences_by_chapter_global": (
                    "viz_oraciones_por_capitulo_global.png"
                ),
                "words_by_chapter_global": "viz_palabras_por_capitulo_global.png",
                "sentences_by_chapter_volume": "viz_oraciones_{slug}.png",
                "words_by_chapter_volume": "viz_palabras_{slug}.png",
                "sentence_length_corpus": "viz_longitud_corpus.png",
                "sentence_length_volume": "viz_longitud_{slug}.png",
                "sentence_length_by_volume": "viz_longitud_por_volumen.png",
                "named_entities_top": "viz_NER_top20_{slug}.png",
                "named_entities_types": "viz_NER_tipos_{slug}.png",
                "pos_distribution": "viz_pos_{slug}.png",
                "wordcloud": "viz_wordcloud_{slug}.png",
            },
            "table_names": {
                "corpus_overview_csv": "n0_viz_corpus_overview.csv",
                "document_distribution_csv": "n0_viz_document_distribution.csv",
                "chapter_distribution_csv": "n0_viz_chapter_distribution.csv",
                "sentence_lengths_csv": "n0_viz_sentence_lengths.csv",
                "named_entities_csv": "n0_viz_named_entities.csv",
                "pos_distribution_csv": "n0_viz_pos_distribution.csv",
                "word_counts_csv": "n0_viz_word_counts.csv",
                "content_lemmas_csv": "n0_viz_content_lemmas.csv",
                "footnotes_summary_csv": "n0_viz_footnotes_summary.csv",
                "export_summary_csv": "n0_viz_export_summary.csv",
            },
        }
    }


def touch(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_visualisation_status_infers_completed_from_existing_output(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "outputs" / "logs" / "stage_00_visualisation_status.json"
    touch(tmp_path, "data/processed/n0_corpus.csv")

    report = get_stage_00_visualisation_status_report(
        visualisation_config(tmp_path), path=status_path, root=tmp_path
    )

    assert report["rows"][0]["step"] == "load"
    assert report["rows"][0]["type"] == "diagnostic"
    assert report["rows"][0]["status"] == "completed"
    assert report["next_step"] == "corpus-overview"
    assert (
        next_visualisation_recommended_command(
            visualisation_config(tmp_path), path=status_path, root=tmp_path
        )
        == "python scripts/00_visualise_corpus.py --step corpus-overview"
    )


def test_visualisation_status_records_successful_step(tmp_path: Path) -> None:
    status_path = tmp_path / "outputs" / "logs" / "stage_00_visualisation_status.json"
    table = touch(tmp_path, "outputs/tables/n0_viz_corpus_overview.csv")
    result = {"table": table, "_summary": {"total_words": 12}}

    update_stage_00_visualisation_status(
        "corpus-overview",
        result,
        command="python scripts/00_visualise_corpus.py --step corpus-overview",
        path=status_path,
    )

    report = get_stage_00_visualisation_status_report(
        visualisation_config(tmp_path), path=status_path, root=tmp_path
    )
    formatted = format_visualisation_status_report(report)

    assert report["rows"][1]["recorded_status"] == "completed"
    assert "Stage 00 visualisation status" in formatted
    assert reset_stage_status(status_path) is True
