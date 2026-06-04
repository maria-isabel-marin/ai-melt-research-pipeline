import json

import pandas as pd

from ai_melt.visualisation import (
    build_sankey_consolidated,
    build_sankey_per_approach,
    chapter_distribution_table,
    corpus_overview_table,
    document_distribution_table,
    extract_content_lemmas_for_subset,
    extract_entities_for_subset,
    plot_corpus_size_by_volume,
    plot_top_domains,
    run_stage_00_visualisation_step,
    safe_json_loads,
    sentence_length_summary_table,
)


def test_safe_json_loads() -> None:
    assert safe_json_loads('["a"]') == ["a"]
    assert safe_json_loads("{bad") == []


def test_extract_helpers() -> None:
    df = pd.DataFrame(
        {
            "entidades_NER": [json.dumps([{"text": "Colombia", "label": "LOC"}])],
            "lemas": [json.dumps(["paz", "y", "construir"])],
            "pos_tags": [json.dumps(["NOUN", "CCONJ", "VERB"])],
        }
    )
    assert extract_entities_for_subset(df) == [("Colombia", "LOC")]
    assert extract_content_lemmas_for_subset(df) == ["paz", "construir"]


def test_corpus_plot_smoke(tmp_path) -> None:
    df_n0 = pd.DataFrame(
        {
            "ID_oracion": ["S-1", "S-2"],
            "volumen": ["Volumen A", "Volumen B"],
            "capitulo": ["Capítulo 1", "Capítulo 2"],
            "n_palabras": [5, 7],
            "pagina": [1, 2],
        }
    )
    written = plot_corpus_size_by_volume(df_n0, tmp_path, {"top_chapters": 2})
    assert len(written) == 4
    assert all(path.exists() for path in written)


def test_stage_00_visualisation_summary_tables() -> None:
    df_n0 = pd.DataFrame(
        {
            "ID_documento": ["DOC-1", "DOC-1", "DOC-2"],
            "ID_oracion": ["S-1", "S-2", "S-3"],
            "volumen": ["Volumen A", "Volumen A", "Volumen B"],
            "capitulo": ["Capitulo 1", "Capitulo 1", "Capitulo 2"],
            "n_palabras": [5, 7, 9],
            "pagina": [1, 2, 3],
        }
    )

    overview = corpus_overview_table(df_n0)
    documents = document_distribution_table(df_n0)
    chapters = chapter_distribution_table(df_n0)
    lengths = sentence_length_summary_table(df_n0)

    assert overview["total_documents"].iloc[0] == 2
    assert overview["total_words"].iloc[0] == 21
    assert documents["n_oraciones"].sum() == 3
    assert chapters["n_palabras"].sum() == 21
    assert lengths.loc[lengths["scope"] == "CORPUS COMPLETO", "max_words"].iloc[0] == 9


def test_run_stage_00_visualisation_corpus_overview_step(tmp_path) -> None:
    corpus_path = tmp_path / "n0_corpus.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    html_dir = tmp_path / "html"
    pd.DataFrame(
        {
            "ID_documento": ["DOC-1"],
            "ID_oracion": ["S-1"],
            "volumen": ["Volumen A"],
            "capitulo": ["Capitulo 1"],
            "n_palabras": [5],
            "pagina": [1],
            "entidades_NER": ["[]"],
            "lemas": ["[]"],
            "pos_tags": ["[]"],
        }
    ).to_csv(corpus_path, index=False)
    config = {
        "stage_00_visualisation": {
            "inputs": {"corpus_csv": corpus_path},
            "outputs": {
                "tables_dir": tables_dir,
                "figures_dir": figures_dir,
                "html_dir": html_dir,
            },
            "table_names": {"corpus_overview_csv": "overview.csv"},
            "figure_names": {},
        }
    }

    result = run_stage_00_visualisation_step(config, "corpus-overview")

    assert result["table"].exists()
    assert pd.read_csv(result["table"])["total_words"].iloc[0] == 5


def test_primary_metaphor_plot_and_sankey_smoke(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "ID_oracion": ["S-1", "S-2"],
            "dominio_fuente": ["CONSTRUCCIÓN", "CUERPO"],
            "dominio_meta": ["PAZ", "SOCIEDAD"],
            "metafora_conceptual": [
                "LA PAZ ES UNA CONSTRUCCIÓN",
                "LA SOCIEDAD ES UN CUERPO",
            ],
            "enfoque": ["openai", "openai"],
        }
    )
    results = {"openai": df}
    written = plot_top_domains(
        results,
        tmp_path,
        {"figure_names": {"top_domains": "domains.png"}, "top_domains": 5},
    )
    sankey = build_sankey_per_approach(df, "openai")
    consolidated = build_sankey_consolidated(results, ["openai"], {"openai": "#D85A30"})

    assert written[0].exists()
    assert sankey is not None
    assert consolidated is not None
