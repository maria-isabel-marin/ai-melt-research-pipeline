import json

import pandas as pd

from ai_melt.corpus import (
    build_n0_dataframe,
    build_sentence_table,
    clean_page_text,
    clean_pages,
    discovered_files_table,
    expand_page_ranges,
    inspect_cleaned_text,
    normalise_regex_patterns,
    remove_footnotes_from_bottom,
    step_build,
    step_clean,
    step_detect_chapters,
    step_discover,
    step_export,
    step_extract,
    summarise_extraction,
    write_parquet_with_optional_csv,
)


class FakeSentence:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeDoc:
    def __init__(self, text: str) -> None:
        self.sents = [
            FakeSentence(part.strip()) for part in text.split(".") if part.strip()
        ]


class FakeNlp:
    def __call__(self, text: str) -> FakeDoc:
        return FakeDoc(text)


def test_expand_page_ranges() -> None:
    assert expand_page_ranges([[1, 3], 7]) == {1, 2, 3, 7}


def test_normalise_regex_patterns_accepts_strings() -> None:
    assert normalise_regex_patterns([r"^\d+$", r"^Índice"]) == [
        r"^\d+$",
        r"^Índice",
    ]


def test_normalise_regex_patterns_accepts_dicts() -> None:
    patterns = normalise_regex_patterns(
        [
            {"pattern": r"^\d+$", "description": "page numbers"},
            {"pattern": r"^Tabla de contenido", "description": "TOC"},
        ]
    )
    assert patterns == [r"^\d+$", r"^Tabla de contenido"]


def test_normalise_regex_patterns_rejects_malformed_entries() -> None:
    try:
        normalise_regex_patterns([{"description": "missing pattern"}])
    except ValueError as exc:
        assert "regex_patterns_to_remove" in str(exc)
        assert "pattern" in str(exc)
    else:
        raise AssertionError("Expected malformed regex config to raise ValueError")


def test_clean_page_text_removes_headers_and_noise() -> None:
    text = "INTRODUCCIÓN\n42\nEste es un párrafo sustantivo para conservar."
    cleaned = clean_page_text(
        text,
        headers_set={"INTRODUCCIÓN"},
        regex_patterns=[r"^\d+$"],
        min_line_length=10,
    )
    assert cleaned == "Este es un párrafo sustantivo para conservar."


def test_remove_footnotes_from_bottom() -> None:
    text, footnotes = remove_footnotes_from_bottom(
        "Cuerpo principal\n12 Nota al pie completa."
    )
    assert text == "Cuerpo principal"
    assert footnotes == ["12 Nota al pie completa."]


def test_clean_pages_accepts_dict_style_regex_config() -> None:
    df_pages = pd.DataFrame(
        [
            {
                "ID_documento": "DOC-1",
                "archivo": "doc.txt",
                "volumen": "Volumen",
                "capitulo": "Capítulo",
                "pagina": 1,
                "texto_pagina": "123\nTexto suficientemente largo para conservar.",
                "n_caracteres": 50,
            }
        ]
    )
    cleaned, _ = clean_pages(
        df_pages,
        {
            "headers_footers": [],
            "pages_to_exclude": {},
            "regex_patterns_to_remove": [
                {"pattern": r"^\d+$", "description": "page numbers"}
            ],
            "min_line_length": 5,
            "min_clean_page_characters": 5,
        },
    )
    assert (
        cleaned["texto_pagina"].iloc[0] == "Texto suficientemente largo para conservar."
    )


def test_clean_pages_and_sentence_table_smoke() -> None:
    df_pages = pd.DataFrame(
        [
            {
                "ID_documento": "DOC-1",
                "archivo": "doc.txt",
                "volumen": "Volumen",
                "capitulo": "Capítulo",
                "pagina": 1,
                "texto_pagina": (
                    "HEADER\nPrimera oración válida. Segunda oración válida."
                ),
                "n_caracteres": 60,
                "metodo_capitulo": "fallback_filename",
            },
            {
                "ID_documento": "DOC-1",
                "archivo": "doc.txt",
                "volumen": "Volumen",
                "capitulo": "Capítulo",
                "pagina": 2,
                "texto_pagina": "Página descartada.",
                "n_caracteres": 18,
                "metodo_capitulo": "fallback_filename",
            },
        ]
    )
    cleaned, footnotes = clean_pages(
        df_pages,
        {
            "headers_footers": ["HEADER"],
            "pages_to_exclude": {"doc.txt": [2]},
            "regex_patterns_to_remove": [],
            "min_line_length": 5,
            "min_clean_page_characters": 5,
        },
    )
    sentences = build_sentence_table(cleaned, FakeNlp())
    df_n0 = build_n0_dataframe(sentences, "CEV-IF-2022")

    assert footnotes.empty
    assert len(cleaned) == 1
    assert df_n0["ID_corpus"].iloc[0] == "CEV-IF-2022"
    assert df_n0["ID_oracion"].tolist() == ["S-000001", "S-000002"]
    assert json.dumps(df_n0["oracion_texto"].tolist(), ensure_ascii=False)


def test_discovered_files_table(tmp_path) -> None:
    corpus_file = tmp_path / "doc.txt"
    corpus_file.write_text("Texto de prueba.", encoding="utf-8")
    table = discovered_files_table([corpus_file], {"volumes": {}})
    assert table["archivo"].tolist() == ["doc.txt"]
    assert table["ID_documento"].tolist() == ["DOC-doc"]


def test_step_pipeline_without_real_corpus_or_spacy(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"
    raw_dir.mkdir()
    (raw_dir / "doc.txt").write_text(
        "HEADER\nPrimera oración válida. Segunda oración válida.\n"
        "12 Nota al pie completa.",
        encoding="utf-8",
    )
    config = {
        "corpus": {"id": "CEV-IF-2022", "volumes": {}},
        "stage_00": {
            "inputs": {"corpus_dir": raw_dir, "file_extensions": [".txt"]},
            "intermediate_outputs": {
                "discovered_files_csv": interim_dir / "n0_discovered_files.csv",
                "pages_raw_parquet": interim_dir / "n0_pages_raw.parquet",
                "pages_with_chapters_parquet": interim_dir
                / "n0_pages_with_chapters.parquet",
                "pages_clean_parquet": interim_dir / "n0_pages_clean.parquet",
                "footnotes_csv": interim_dir / "n0_footnotes.csv",
                "sentences_parquet": interim_dir / "n0_sentences.parquet",
                "sentences_annotated_parquet": interim_dir
                / "n0_sentences_annotated.parquet",
            },
            "debug_outputs": {
                "write_csv_copies": False,
                "csv_preview_max_text_length": 10,
                "include_full_text_in_csv": True,
            },
            "outputs": {
                "corpus_csv": processed_dir / "n0_corpus.csv",
                "corpus_parquet": processed_dir / "n0_corpus.parquet",
                "metadata_json": processed_dir / "n0_metadata.json",
                "footnotes_csv": tables_dir / "n0_footnotes.csv",
                "document_summary_csv": tables_dir / "n0_summary_by_document.csv",
                "chapter_summary_csv": tables_dir / "n0_summary_by_chapter.csv",
            },
            "cleaning": {
                "headers_footers": ["HEADER"],
                "regex_patterns_to_remove": [],
                "pages_to_exclude": {},
                "min_line_length": 5,
                "min_clean_page_characters": 5,
            },
            "extraction": {},
            "segmentation": {},
            "inspection": {"sample_size": 1, "window_size": 20, "random_state": 42},
        },
    }

    discovered = step_discover(config, limit_files=1)
    raw_pages = step_extract(config, limit_files=1, limit_pages=1, write_csv=True)
    chaptered = step_detect_chapters(config, limit_pages=1, write_csv=True)
    clean_pages_df, footnotes = step_clean(config, limit_pages=1, write_csv=True)
    inspection = inspect_cleaned_text(clean_pages_df, sample_size=1, window_size=20)

    annotated = pd.DataFrame(
        {
            "ID_documento": ["DOC-doc"],
            "volumen": ["Doc"],
            "capitulo": ["Doc"],
            "pagina": [1],
            "ID_oracion": ["S-000001"],
            "oracion_texto": ["Primera oración válida."],
            "n_palabras": [3],
            "n_caracteres": [24],
            "tokens": ["[]"],
            "lemas": ["[]"],
            "pos_tags": ["[]"],
            "entidades_NER": ["[]"],
        }
    )
    annotated.to_parquet(
        config["stage_00"]["intermediate_outputs"]["sentences_annotated_parquet"],
        index=False,
    )
    n0 = step_build(config, write_csv=True)
    written = step_export(config)

    assert len(discovered) == 1
    assert len(raw_pages) == 1
    assert (
        config["stage_00"]["intermediate_outputs"]["pages_raw_parquet"]
        .with_suffix(".csv")
        .exists()
    )
    assert chaptered["capitulo"].iloc[0] == "Doc"
    assert len(footnotes) == 1
    assert "Primera" in inspection["inicio"].iloc[0]
    assert n0["ID_corpus"].iloc[0] == "CEV-IF-2022"
    assert written["corpus_csv"].exists()
    assert written["chapter_summary_csv"].exists()
    assert "Documents processed" in summarise_extraction(raw_pages)


def test_write_parquet_with_optional_csv(tmp_path) -> None:
    df = pd.DataFrame({"texto_pagina": ["abcdef"], "value": [1]})
    parquet_path = tmp_path / "sample.parquet"
    outputs = write_parquet_with_optional_csv(
        df,
        parquet_path,
        {
            "write_csv_copies": True,
            "csv_preview_max_text_length": 3,
            "include_full_text_in_csv": False,
        },
    )
    csv_path = parquet_path.with_suffix(".csv")
    csv_df = pd.read_csv(csv_path)
    assert parquet_path in outputs
    assert csv_path in outputs
    assert "texto_pagina" not in csv_df.columns
    assert csv_df["texto_pagina_preview"].iloc[0] == "abc"


def test_write_parquet_without_optional_csv(tmp_path) -> None:
    df = pd.DataFrame({"value": [1]})
    parquet_path = tmp_path / "sample.parquet"
    outputs = write_parquet_with_optional_csv(
        df,
        parquet_path,
        {"write_csv_copies": False},
    )
    assert outputs == [parquet_path]
    assert not parquet_path.with_suffix(".csv").exists()
