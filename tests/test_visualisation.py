import json

import pandas as pd

from ai_melt.visualisation import (
    build_sankey_consolidated,
    build_sankey_per_approach,
    extract_content_lemmas_for_subset,
    extract_entities_for_subset,
    plot_corpus_size_by_volume,
    plot_top_domains,
    safe_json_loads,
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
