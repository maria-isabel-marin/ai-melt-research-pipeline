"""Corpus ingestion utilities for the AI-MELT research pipeline."""

from __future__ import annotations

import datetime as dt
import gc
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_melt.paths import project_path


def discover_corpus_files(
    corpus_dir: str | Path, extensions: Iterable[str] = (".pdf", ".txt")
) -> list[Path]:
    """Return PDF/TXT corpus files in deterministic order."""
    corpus_path = project_path(corpus_dir)
    if not corpus_path.exists():
        return []
    normalized = {ext.lower() for ext in extensions}
    files = [
        path
        for path in corpus_path.iterdir()
        if path.is_file() and path.suffix.lower() in normalized
    ]
    return sorted(files, key=lambda path: path.name.lower())


def default_document_metadata(path: Path) -> dict[str, str]:
    """Create notebook-compatible fallback metadata for an unknown volume."""
    return {
        "title": path.stem.replace("_", " ").title(),
        "author": "Comisión de la Verdad",
        "date": "2022",
        "document_type": "Volumen del Informe Final",
    }


def get_chapter_map_from_toc(pdf_path: str | Path) -> dict[int, str] | None:
    """Detect chapters from PDF outline/bookmarks, using the outermost level."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("PyMuPDF is required to read PDF files.") from exc

    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()
    if not toc:
        return None

    entries = [(title.strip(), page) for level, title, page in toc if level == 1]
    if not entries:
        min_level = min(level for level, _, _ in toc)
        entries = [
            (title.strip(), page) for level, title, page in toc if level == min_level
        ]
    if not entries:
        return None

    chapter_map: dict[int, str] = {}
    for index, (title, start_page) in enumerate(entries):
        end_page = entries[index + 1][1] - 1 if index + 1 < len(entries) else 999999
        for page in range(start_page, end_page + 1):
            chapter_map[page] = title
    return chapter_map


def get_chapter_map_from_fonts(
    pdf_path: str | Path, top_percentile: float = 0.05
) -> dict[int, str] | None:
    """Detect likely chapter titles from large-font lines in a PDF."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("PyMuPDF is required to read PDF files.") from exc

    doc = fitz.open(pdf_path)
    font_sizes: list[float] = []
    page_lines_with_fonts: list[list[dict[str, Any]]] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_lines = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line.get("spans", [])
                if not spans:
                    continue
                max_size = max(span.get("size", 0) for span in spans)
                text = " ".join(span.get("text", "") for span in spans).strip()
                is_bold = any("bold" in span.get("font", "").lower() for span in spans)
                if text and len(text) > 2:
                    font_sizes.append(max_size)
                    page_lines.append(
                        {
                            "text": text,
                            "size": max_size,
                            "bold": is_bold,
                            "page": page_num + 1,
                        }
                    )
        page_lines_with_fonts.append(page_lines)
    doc.close()

    if not font_sizes:
        return None
    threshold = np.percentile(np.array(font_sizes), (1 - top_percentile) * 100)

    candidates = []
    for page_lines in page_lines_with_fonts:
        for line in page_lines:
            text = line["text"].strip()
            if (
                line["size"] >= threshold
                and 3 < len(text) < 200
                and not re.match(r"^\d{1,4}$", text)
            ):
                candidates.append(line.copy())
    if not candidates:
        return None

    chapters = []
    current = candidates[0]
    for candidate in candidates[1:]:
        if (
            candidate["page"] == current["page"]
            and candidate["size"] == current["size"]
        ):
            current["text"] += " " + candidate["text"]
        else:
            chapters.append(current)
            current = candidate
    chapters.append(current)

    chapter_map: dict[int, str] = {}
    for index, chapter in enumerate(chapters):
        page_end = (
            chapters[index + 1]["page"] - 1 if index + 1 < len(chapters) else 999999
        )
        name = chapter["text"].strip()
        name = re.sub(r"^(Capítulo|Cap\.?)\s*\d+\.?\s*", "", name, flags=re.I)
        name = re.sub(r"^\d+\.\s*", "", name)
        name = re.sub(r"^[IVXLC]+\.\s*", "", name)
        if name:
            for page in range(chapter["page"], page_end + 1):
                chapter_map[page] = name
    return chapter_map


def get_chapter_for_page(
    page_number: int, chapter_map: dict[int, str] | None, fallback_name: str
) -> str:
    """Return the detected chapter name for a page, or a configured fallback."""
    if chapter_map and page_number in chapter_map:
        return chapter_map[page_number]
    return fallback_name


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract text from a PDF, preserving one row per non-empty page."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("PyMuPDF is required to read PDF files.") from exc

    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        if text.strip():
            pages.append({"pagina": page_num + 1, "texto": text.strip()})
    doc.close()
    return pages


def extract_text_from_txt(txt_path: str | Path) -> list[dict[str, Any]]:
    """Read a TXT file and treat it as one notebook-compatible page."""
    text = Path(txt_path).read_text(encoding="utf-8")
    return [{"pagina": 1, "texto": text.strip()}]


def discovered_files_table(
    corpus_files: Iterable[Path], corpus_config: dict[str, Any]
) -> pd.DataFrame:
    """Build an inspectable table of discovered corpus files."""
    volume_metadata = corpus_config.get("volumes", {})
    columns = [
        "file_path",
        "archivo",
        "extension",
        "ID_documento",
        "has_configured_metadata",
        "volumen",
    ]
    rows = []
    for path in corpus_files:
        metadata = volume_metadata.get(path.name, {})
        rows.append(
            {
                "file_path": str(path),
                "archivo": path.name,
                "extension": path.suffix.lower(),
                "ID_documento": f"DOC-{path.stem}",
                "has_configured_metadata": path.name in volume_metadata,
                "volumen": metadata.get("title")
                or metadata.get("titulo")
                or path.stem.replace("_", " ").title(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paths_from_discovered_files(df_files: pd.DataFrame) -> list[Path]:
    """Return corpus file paths from the discovery table."""
    return [project_path(path) for path in df_files["file_path"].tolist()]


def extract_pages_raw(
    corpus_files: Iterable[Path],
    corpus_config: dict[str, Any],
    limit: int | None = None,
) -> pd.DataFrame:
    """Extract page text without assigning detected chapters."""
    volume_metadata = corpus_config.get("volumes", {})
    columns = [
        "ID_documento",
        "archivo",
        "file_path",
        "volumen",
        "pagina",
        "texto_pagina",
        "n_caracteres",
    ]
    rows: list[dict[str, Any]] = []
    for filepath in list(corpus_files)[:limit]:
        meta = volume_metadata.get(filepath.name, default_document_metadata(filepath))
        title = meta.get("title") or meta.get("titulo") or filepath.stem
        pages = (
            extract_text_from_pdf(filepath)
            if filepath.suffix.lower() == ".pdf"
            else extract_text_from_txt(filepath)
        )
        for page in pages:
            text = page["texto"]
            rows.append(
                {
                    "ID_documento": f"DOC-{filepath.stem}",
                    "archivo": filepath.name,
                    "file_path": str(filepath),
                    "volumen": title,
                    "pagina": page["pagina"],
                    "texto_pagina": text,
                    "n_caracteres": len(text),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def detect_chapters_for_pages(
    df_pages: pd.DataFrame,
    corpus_config: dict[str, Any],
    extraction_config: dict[str, Any] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Assign chapter labels and chapter-detection methods to raw page rows."""
    extraction_config = extraction_config or {}
    volume_metadata = corpus_config.get("volumes", {})
    font_top_percentile = extraction_config.get("font_top_percentile", 0.03)
    chapter_info: dict[str, tuple[dict[int, str] | None, str]] = {}
    if df_pages.empty:
        df_empty = df_pages.copy()
        df_empty["capitulo"] = pd.Series(dtype="object")
        df_empty["metodo_capitulo"] = pd.Series(dtype="object")
        return df_empty

    file_rows = df_pages[["archivo", "file_path", "volumen"]].drop_duplicates()
    if limit is not None:
        file_rows = file_rows.head(limit)
    for _, row in file_rows.iterrows():
        filepath = project_path(row["file_path"])
        chapter_map = None
        chapter_method = "fallback_filename"
        if filepath.suffix.lower() == ".pdf":
            chapter_map = get_chapter_map_from_toc(filepath)
            if chapter_map:
                chapter_method = "toc_bookmarks"
            else:
                chapter_map = get_chapter_map_from_fonts(filepath, font_top_percentile)
                chapter_method = "font_size" if chapter_map else "fallback_filename"
        chapter_info[row["archivo"]] = (chapter_map, chapter_method)

    df_with_chapters = df_pages.copy()

    def chapter_for_row(row: pd.Series) -> str:
        metadata = volume_metadata.get(row["archivo"], {})
        fallback = (
            metadata.get("title")
            or metadata.get("titulo")
            or row.get("volumen")
            or row["archivo"]
        )
        chapter_map, _ = chapter_info.get(row["archivo"], (None, "fallback_filename"))
        return get_chapter_for_page(int(row["pagina"]), chapter_map, str(fallback))

    df_with_chapters["capitulo"] = df_with_chapters.apply(chapter_for_row, axis=1)
    df_with_chapters["metodo_capitulo"] = df_with_chapters["archivo"].map(
        {filename: method for filename, (_, method) in chapter_info.items()}
    )
    df_with_chapters["metodo_capitulo"] = df_with_chapters["metodo_capitulo"].fillna(
        "fallback_filename"
    )
    return df_with_chapters


def extract_pages(
    corpus_files: Iterable[Path],
    corpus_config: dict[str, Any],
    extraction_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract page-level text and chapter metadata from corpus files."""
    raw_pages = extract_pages_raw(corpus_files, corpus_config)
    return detect_chapters_for_pages(raw_pages, corpus_config, extraction_config)


def expand_page_ranges(page_specs: Iterable[Any]) -> set[int]:
    """Expand page numbers and inclusive ranges into individual page numbers."""
    pages: set[int] = set()
    for index, spec in enumerate(page_specs):
        if isinstance(spec, list | tuple) and len(spec) == 2:
            pages.update(range(int(spec[0]), int(spec[1]) + 1))
        elif isinstance(spec, dict) and {"start", "end"} <= set(spec):
            pages.update(range(int(spec["start"]), int(spec["end"]) + 1))
        elif isinstance(spec, dict) and "page" in spec:
            pages.add(int(spec["page"]))
        elif isinstance(spec, dict):
            raise ValueError(
                "Malformed pages_to_exclude entry at index "
                f"{index}: {spec!r}. Expected a page number, a two-item range, "
                "or a mapping with 'start'/'end' or 'page'."
            )
        else:
            pages.add(int(spec))
    return pages


def detect_repeated_headers(
    pages_text: Iterable[str], threshold: float = 0.3
) -> set[str]:
    """Detect lines repeated in more than ``threshold`` of pages."""
    page_list = list(pages_text)
    if len(page_list) < 3:
        return set()

    line_counts: Counter[str] = Counter()
    for page in page_list:
        unique_lines = {
            line.strip()
            for line in str(page).split("\n")
            if 5 < len(line.strip()) < 100
        }
        line_counts.update(unique_lines)
    return {
        line
        for line, count in line_counts.items()
        if count >= len(page_list) * threshold
    }


def clean_page_text(
    text: str,
    headers_set: Iterable[str],
    regex_patterns: Iterable[str],
    min_line_length: int,
) -> str:
    """Remove exact headers, regex-matched noise, and short heading fragments."""
    cleaned_lines = []
    headers = normalise_config_text_list(headers_set, "headers_set")
    patterns = normalise_regex_patterns(regex_patterns)
    headers_upper = {header.upper().strip() for header in headers}
    for line in str(text).split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper() in headers_upper:
            continue
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in patterns):
            continue
        if 0 < len(stripped) < min_line_length and not stripped[0].islower():
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def normalise_config_text_list(entries: Iterable[Any], field_name: str) -> list[str]:
    """Normalise string or ``{value: ...}`` style config entries to strings."""
    normalised = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            normalised.append(entry)
            continue
        if isinstance(entry, dict) and isinstance(entry.get("value"), str):
            normalised.append(entry["value"])
            continue
        raise ValueError(
            f"Malformed {field_name} entry at index {index}: {entry!r}. "
            "Expected a string or a mapping with a string 'value' key."
        )
    return normalised


def normalise_regex_patterns(entries: Iterable[Any]) -> list[str]:
    """Normalise regex config entries to pattern strings."""
    patterns = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            patterns.append(entry)
            continue
        if isinstance(entry, dict) and isinstance(entry.get("pattern"), str):
            patterns.append(entry["pattern"])
            continue
        raise ValueError(
            "Malformed regex_patterns_to_remove entry at index "
            f"{index}: {entry!r}. Expected a string or a mapping with a "
            "string 'pattern' key."
        )
    return patterns


def remove_footnotes_from_bottom(text: str) -> tuple[str, list[str]]:
    """Remove bottom footnote lines matching the legacy notebook heuristic."""
    lines = str(text).split("\n")
    footnotes = []
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if re.match(r"^\d{1,3}\s+[A-ZÁÉÍÓÚÑ].*\.\s*$", last):
            footnotes.append(last)
            lines.pop()
        else:
            break
    footnotes.reverse()
    return "\n".join(lines), footnotes


def clean_pages(
    df_pages: pd.DataFrame,
    cleaning_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean page-level text and return cleaned pages plus extracted footnotes."""
    footnote_columns = [
        "ID_documento",
        "archivo",
        "volumen",
        "pagina",
        "capitulo",
        "nota_al_pie",
    ]
    if df_pages.empty:
        return df_pages.copy(), pd.DataFrame(columns=footnote_columns)

    threshold = cleaning_config.get("repeated_header_threshold", 0.3)
    manual_headers = set(
        normalise_config_text_list(
            cleaning_config.get("headers_footers", []), "headers_footers"
        )
    )
    regex_patterns = normalise_regex_patterns(
        cleaning_config.get("regex_patterns_to_remove", [])
    )
    min_line_length = int(cleaning_config.get("min_line_length", 30))
    min_clean_chars = int(cleaning_config.get("min_clean_page_characters", 50))

    auto_headers_per_volume = {
        file_name: detect_repeated_headers(
            df_pages.loc[df_pages["archivo"] == file_name, "texto_pagina"], threshold
        )
        for file_name in df_pages["archivo"].unique()
    }
    chapter_headers_per_volume = {
        file_name: set(
            df_pages.loc[df_pages["archivo"] == file_name, "capitulo"].dropna().unique()
        )
        for file_name in df_pages["archivo"].unique()
    }
    global_headers = manual_headers | detect_repeated_headers(
        df_pages["texto_pagina"], threshold
    )

    expanded_excludes = {
        file_name: expand_page_ranges(specs)
        for file_name, specs in cleaning_config.get("pages_to_exclude", {}).items()
    }
    keep_mask = [
        not (
            row["archivo"] in expanded_excludes
            and row["pagina"] in expanded_excludes[row["archivo"]]
        )
        for _, row in df_pages.iterrows()
    ]
    df_clean = df_pages.loc[keep_mask].copy()

    footnotes = []

    def headers_for_file(file_name: str) -> set[str]:
        combined = set(global_headers)
        combined |= auto_headers_per_volume.get(file_name, set())
        combined |= chapter_headers_per_volume.get(file_name, set())
        return combined

    cleaned_page_texts = []
    for _, row in df_clean.iterrows():
        without_footnotes, extracted = remove_footnotes_from_bottom(row["texto_pagina"])
        for note in extracted:
            footnotes.append(
                {
                    "ID_documento": row["ID_documento"],
                    "archivo": row["archivo"],
                    "volumen": row.get("volumen", ""),
                    "pagina": row["pagina"],
                    "capitulo": row.get("capitulo", ""),
                    "nota_al_pie": note,
                }
            )
        cleaned_page_texts.append(
            clean_page_text(
                without_footnotes,
                headers_for_file(row["archivo"]),
                regex_patterns,
                min_line_length,
            )
        )

    df_clean["texto_pagina"] = cleaned_page_texts
    df_clean = df_clean[df_clean["texto_pagina"].str.len() > min_clean_chars].copy()
    df_clean["n_caracteres"] = df_clean["texto_pagina"].str.len()
    return df_clean, pd.DataFrame(footnotes, columns=footnote_columns)


def load_spacy_model(segmentation_config: dict[str, Any]):
    """Load the configured spaCy model for Spanish sentence/NLP processing."""
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("spaCy is required for corpus ingestion.") from exc

    model_name = segmentation_config.get("spacy_model", "es_core_news_lg")
    exclude = segmentation_config.get("spacy_exclude", ["textcat", "custom"])
    nlp = spacy.load(model_name, exclude=exclude)
    nlp.max_length = int(segmentation_config.get("max_length", 3_000_000))
    return nlp


def segment_into_sentences(
    text: str,
    nlp: Any,
    min_length: int = 10,
    max_length: int = 2000,
) -> list[str]:
    """Segment text into sentences using a spaCy-like object."""
    doc = nlp(str(text))
    sentences = []
    for sent in doc.sents:
        sent_text = re.sub(r"\s+", " ", sent.text.strip())
        if min_length <= len(sent_text) <= max_length:
            sentences.append(sent_text)
    return sentences


def build_sentence_table(
    df_pages: pd.DataFrame,
    nlp: Any,
    segmentation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the notebook's sentence-level N0 table before token annotation."""
    segmentation_config = segmentation_config or {}
    min_length = int(segmentation_config.get("min_sentence_length", 10))
    max_length = int(segmentation_config.get("max_sentence_length", 2000))
    rows: list[dict[str, Any]] = []
    sentence_counter = 0
    for _, row in df_pages.iterrows():
        for sentence in segment_into_sentences(
            row["texto_pagina"], nlp, min_length, max_length
        ):
            sentence_counter += 1
            rows.append(
                {
                    "ID_documento": row["ID_documento"],
                    "volumen": row["volumen"],
                    "capitulo": row["capitulo"],
                    "pagina": row["pagina"],
                    "ID_oracion": f"S-{sentence_counter:06d}",
                    "oracion_texto": sentence,
                    "n_palabras": len(sentence.split()),
                    "n_caracteres": len(sentence),
                }
            )
    return pd.DataFrame(rows)


def process_nlp_batch(
    texts: list[str], nlp: Any
) -> list[tuple[list, list, list, list]]:
    """Process a batch of sentences and return tokens, lemmas, POS, and entities."""
    results = []
    for doc in nlp.pipe(texts, batch_size=200, n_process=1):
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        results.append((tokens, lemmas, pos_tags, entities))
    return results


def add_linguistic_annotations(
    df_sentences: pd.DataFrame,
    nlp: Any,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Add JSON token, lemma, POS, and NER columns to the sentence table."""
    df = df_sentences.copy()
    all_tokens: list[str] = []
    all_lemmas: list[str] = []
    all_pos: list[str] = []
    all_ner: list[str] = []

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_results = process_nlp_batch(
            df["oracion_texto"].iloc[start:end].tolist(), nlp
        )
        for tokens, lemmas, pos_tags, entities in batch_results:
            all_tokens.append(
                json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
            )
            all_lemmas.append(
                json.dumps(lemmas, ensure_ascii=False, separators=(",", ":"))
            )
            all_pos.append(
                json.dumps(pos_tags, ensure_ascii=False, separators=(",", ":"))
            )
            all_ner.append(
                json.dumps(
                    [{"text": text, "label": label} for text, label in entities],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        del batch_results
        gc.collect()

    df["tokens"] = all_tokens
    df["lemas"] = all_lemmas
    df["pos_tags"] = all_pos
    df["entidades_NER"] = all_ner
    return df


def build_n0_dataframe(df_sentences: pd.DataFrame, corpus_id: str) -> pd.DataFrame:
    """Add the corpus identifier to the notebook-compatible N0 sentence table."""
    df_n0 = df_sentences.copy()
    df_n0.insert(0, "ID_corpus", corpus_id)
    return df_n0


def document_summary(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence, word, and page counts by document/chapter."""
    return (
        df_n0.groupby(["ID_documento", "volumen", "capitulo"])
        .agg(
            n_oraciones=("ID_oracion", "count"),
            n_palabras=("n_palabras", "sum"),
            paginas=("pagina", "nunique"),
        )
        .reset_index()
    )


def summary_by_document(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence, word, chapter, and page counts by document."""
    return (
        df_n0.groupby(["ID_documento", "volumen"])
        .agg(
            n_oraciones=("ID_oracion", "count"),
            n_palabras=("n_palabras", "sum"),
            n_capitulos=("capitulo", "nunique"),
            paginas=("pagina", "nunique"),
        )
        .reset_index()
    )


def summary_by_chapter(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence, word, and page counts by document chapter."""
    return document_summary(df_n0)


def corpus_metadata(
    df_n0: pd.DataFrame,
    corpus_config: dict[str, Any],
    processed_files: Iterable[Path],
    nlp: Any | None = None,
) -> dict[str, Any]:
    """Build the N0 metadata JSON payload produced by the legacy notebook."""
    metadata = {
        "ID_corpus": corpus_config.get("id", ""),
        "nombre_corpus": corpus_config.get("name", ""),
        "comunidad_discursiva": corpus_config.get("discourse_community", ""),
        "genero_textual": corpus_config.get("textual_genre", ""),
        "fecha": corpus_config.get("date", ""),
        "idioma": corpus_config.get("language", ""),
        "timestamp": dt.datetime.now().isoformat(),
        "stats": {
            "n_documentos": int(df_n0["ID_documento"].nunique()),
            "n_oraciones": int(len(df_n0)),
            "n_palabras_total": int(df_n0["n_palabras"].sum()),
            "palabras_promedio_por_oracion": round(
                float(df_n0["n_palabras"].mean()), 1
            ),
        },
        "archivos_procesados": [path.name for path in processed_files],
    }
    if nlp is not None:
        metadata["spacy_model"] = f"{nlp.meta.get('lang')}_{nlp.meta.get('name')}"
        metadata["spacy_version"] = nlp.meta.get("version")
    return metadata


def stage_00_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return stage 00 configuration with top-level defaults merged downward."""
    stage_config = config["stage_00"].copy()
    segmentation = stage_config.get("segmentation", {}).copy()
    segmentation.setdefault("spacy_model", stage_config.get("spacy_model"))
    segmentation.setdefault("nlp_batch_size", stage_config.get("batch_size"))
    segmentation.setdefault("max_length", stage_config.get("max_length"))
    stage_config["segmentation"] = {
        key: value for key, value in segmentation.items() if value is not None
    }
    return stage_config


def stage_00_path(config: dict[str, Any], section: str, key: str) -> Path:
    """Resolve a configured stage 00 file path."""
    return project_path(stage_00_config(config)[section][key])


def ensure_parent(path: Path) -> Path:
    """Create a file path's parent directory and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def missing_previous_step_error(path: Path, previous_step: str) -> FileNotFoundError:
    """Create a clear missing-intermediate exception."""
    return FileNotFoundError(
        f"Required intermediate file not found: {path}. "
        f"Run `python scripts/00_ingest_corpus.py --step {previous_step}` first."
    )


def read_required_parquet(path: Path, previous_step: str) -> pd.DataFrame:
    """Read a required parquet file or fail with a stage-specific message."""
    if not path.exists():
        raise missing_previous_step_error(path, previous_step)
    return pd.read_parquet(path)


def read_required_csv(path: Path, previous_step: str) -> pd.DataFrame:
    """Read a required CSV file or fail with a stage-specific message."""
    if not path.exists():
        raise missing_previous_step_error(path, previous_step)
    return pd.read_csv(path)


def csv_copy_path(parquet_path: Path) -> Path:
    """Return the debug CSV path paired with a parquet path."""
    return parquet_path.with_suffix(".csv")


def debug_output_config(
    config: dict[str, Any], write_csv: bool | None = None
) -> dict[str, Any]:
    """Return stage 00 debug-output settings, applying CLI overrides."""
    debug_config = stage_00_config(config).get("debug_outputs", {}).copy()
    if write_csv:
        debug_config["write_csv_copies"] = True
    debug_config.setdefault("write_csv_copies", False)
    debug_config.setdefault("csv_preview_max_text_length", 2000)
    debug_config.setdefault("include_full_text_in_csv", True)
    return debug_config


def add_text_preview_columns(
    df: pd.DataFrame, max_text_length: int = 2000
) -> pd.DataFrame:
    """Add truncated preview columns for long text fields in debug CSVs."""
    df_csv = df.copy()
    for column in ["texto_pagina", "oracion_texto"]:
        if column in df_csv:
            preview_column = f"{column}_preview"
            df_csv[preview_column] = (
                df_csv[column].fillna("").astype(str).str.slice(0, max_text_length)
            )
    return df_csv


def write_debug_csv_copy(
    df: pd.DataFrame, csv_path: Path, debug_config: dict[str, Any]
) -> Path:
    """Write a UTF-8-SIG CSV copy with optional text previews."""
    df_csv = add_text_preview_columns(
        df, int(debug_config.get("csv_preview_max_text_length", 2000))
    )
    if not debug_config.get("include_full_text_in_csv", True):
        df_csv = df_csv.drop(
            columns=[
                column
                for column in ["texto_pagina", "oracion_texto"]
                if column in df_csv.columns
            ]
        )
    df_csv.to_csv(ensure_parent(csv_path), index=False, encoding="utf-8-sig")
    return csv_path


def write_parquet_with_optional_csv(
    df: pd.DataFrame,
    parquet_path: Path,
    debug_config: dict[str, Any] | None = None,
) -> list[Path]:
    """Write canonical parquet plus an optional debug CSV copy."""
    parquet_path = ensure_parent(parquet_path)
    df.to_parquet(parquet_path, index=False)
    outputs = [parquet_path]
    if debug_config and debug_config.get("write_csv_copies", False):
        outputs.append(
            write_debug_csv_copy(df, csv_copy_path(parquet_path), debug_config)
        )
    return outputs


def attach_outputs(result: Any, outputs: list[Path]) -> Any:
    """Attach written output paths to DataFrame-like results."""
    if isinstance(result, pd.DataFrame):
        result.attrs["outputs"] = outputs
    elif isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                item.attrs["outputs"] = outputs
                break
    return result


def output_lines(outputs: Iterable[Path]) -> list[str]:
    """Format output paths for console summaries."""
    return ["Outputs:"] + [f"  - {path}" for path in outputs]


def format_counts(series: pd.Series, label: str, limit: int = 10) -> list[str]:
    """Format value counts for compact summaries."""
    lines = [label]
    for key, value in series.head(limit).items():
        lines.append(f"  - {key}: {int(value):,}")
    return lines


def step_discover(
    config: dict[str, Any], limit_files: int | None = None
) -> pd.DataFrame:
    """Discover configured corpus files and write the discovery table."""
    stage_config = stage_00_config(config)
    files = discover_corpus_files(
        stage_config["inputs"]["corpus_dir"],
        stage_config["inputs"].get("file_extensions", [".pdf", ".txt"]),
    )
    if limit_files is not None:
        files = files[:limit_files]
    df_files = discovered_files_table(files, config["corpus"])
    output_path = stage_00_path(config, "intermediate_outputs", "discovered_files_csv")
    df_files.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    df_files.attrs["outputs"] = [output_path]
    df_files.attrs["corpus_dir"] = stage_config["inputs"]["corpus_dir"]
    df_files.attrs["extensions"] = stage_config["inputs"].get(
        "file_extensions", [".pdf", ".txt"]
    )
    df_files.attrs["limit_files"] = limit_files
    return df_files


def step_extract(
    config: dict[str, Any],
    limit_files: int | None = None,
    limit_pages: int | None = None,
    write_csv: bool | None = None,
) -> pd.DataFrame:
    """Load discovered files, extract raw page text, and write raw pages."""
    discovered_path = stage_00_path(
        config, "intermediate_outputs", "discovered_files_csv"
    )
    df_files = read_required_csv(discovered_path, "discover")
    if limit_files is not None:
        df_files = df_files.head(limit_files)
    files = paths_from_discovered_files(df_files)
    df_pages = extract_pages_raw(files, config["corpus"])
    if limit_pages is not None:
        df_pages = df_pages.head(limit_pages)
    output_path = stage_00_path(config, "intermediate_outputs", "pages_raw_parquet")
    outputs = write_parquet_with_optional_csv(
        df_pages, output_path, debug_output_config(config, write_csv)
    )
    df_pages.attrs["outputs"] = outputs
    df_pages.attrs["limit_files"] = limit_files
    df_pages.attrs["limit_pages"] = limit_pages
    return df_pages


def step_detect_chapters(
    config: dict[str, Any],
    limit_pages: int | None = None,
    write_csv: bool | None = None,
) -> pd.DataFrame:
    """Load raw pages, detect chapters, and write chaptered pages."""
    raw_path = stage_00_path(config, "intermediate_outputs", "pages_raw_parquet")
    df_raw = read_required_parquet(raw_path, "extract")
    if limit_pages is not None:
        df_raw = df_raw.head(limit_pages)
    df_pages = detect_chapters_for_pages(
        df_raw,
        config["corpus"],
        stage_00_config(config).get("extraction", {}),
    )
    output_path = stage_00_path(
        config, "intermediate_outputs", "pages_with_chapters_parquet"
    )
    outputs = write_parquet_with_optional_csv(
        df_pages, output_path, debug_output_config(config, write_csv)
    )
    df_pages.attrs["outputs"] = outputs
    df_pages.attrs["limit_pages"] = limit_pages
    return df_pages


def step_clean(
    config: dict[str, Any],
    limit_pages: int | None = None,
    write_csv: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chaptered pages, clean text, and write clean pages plus footnotes."""
    pages_path = stage_00_path(
        config, "intermediate_outputs", "pages_with_chapters_parquet"
    )
    df_pages = read_required_parquet(pages_path, "detect-chapters")
    pages_before_limit = len(df_pages)
    if limit_pages is not None:
        df_pages = df_pages.head(limit_pages)
    df_clean, footnotes = clean_pages(
        df_pages, stage_00_config(config).get("cleaning", {})
    )
    clean_path = stage_00_path(config, "intermediate_outputs", "pages_clean_parquet")
    footnotes_path = stage_00_path(config, "intermediate_outputs", "footnotes_csv")
    outputs = write_parquet_with_optional_csv(
        df_clean, clean_path, debug_output_config(config, write_csv)
    )
    footnotes.to_csv(ensure_parent(footnotes_path), index=False, encoding="utf-8-sig")
    outputs.append(footnotes_path)
    df_clean.attrs["outputs"] = outputs
    df_clean.attrs["pages_before_cleaning"] = len(df_pages)
    df_clean.attrs["pages_before_limit"] = pages_before_limit
    df_clean.attrs["characters_before_cleaning"] = int(df_pages["n_caracteres"].sum())
    df_clean.attrs["limit_pages"] = limit_pages
    return df_clean, footnotes


def filter_pages(
    df_pages: pd.DataFrame,
    document_id: str | None = None,
    page_number: int | None = None,
) -> pd.DataFrame:
    """Filter page-level tables for inspection."""
    filtered = df_pages
    if document_id:
        filtered = filtered[filtered["ID_documento"] == document_id]
    if page_number is not None:
        filtered = filtered[filtered["pagina"] == page_number]
    return filtered


def inspect_cleaned_text(
    df_pages: pd.DataFrame,
    sample_size: int = 5,
    window_size: int = 100,
    random_state: int = 42,
    document_id: str | None = None,
    page_number: int | None = None,
) -> pd.DataFrame:
    """Return notebook-like snippets from cleaned pages for manual inspection."""
    filtered = filter_pages(df_pages, document_id, page_number)
    if filtered.empty:
        return pd.DataFrame()
    sample = filtered.sample(
        n=min(sample_size, len(filtered)), random_state=random_state
    ).sort_values(["ID_documento", "pagina"])
    rows = []
    for _, row in sample.iterrows():
        text = str(row["texto_pagina"])
        rows.append(
            {
                "ID_documento": row["ID_documento"],
                "archivo": row["archivo"],
                "volumen": row["volumen"],
                "capitulo": row.get("capitulo", ""),
                "pagina": row["pagina"],
                "inicio": text[:window_size],
                "final": text[-window_size:] if len(text) > window_size else text,
                "n_caracteres": len(text),
            }
        )
    return pd.DataFrame(rows)


def inspect_footnotes(
    footnotes: pd.DataFrame,
    sample_size: int = 5,
    random_state: int = 42,
    document_id: str | None = None,
    page_number: int | None = None,
) -> pd.DataFrame:
    """Return a sample of extracted footnotes for manual inspection."""
    if footnotes.empty:
        return pd.DataFrame()
    filtered = footnotes
    if document_id and "ID_documento" in filtered:
        filtered = filtered[filtered["ID_documento"] == document_id]
    if page_number is not None:
        filtered = filtered[filtered["pagina"] == page_number]
    if filtered.empty:
        return pd.DataFrame()
    return filtered.sample(
        n=min(sample_size, len(filtered)), random_state=random_state
    ).sort_values(["archivo", "pagina"])


def step_inspect_cleaning(
    config: dict[str, Any],
    sample_size: int | None = None,
    window_size: int | None = None,
    random_state: int | None = None,
    document_id: str | None = None,
    page_number: int | None = None,
) -> pd.DataFrame:
    """Load clean pages and return a cleaned-text inspection table."""
    clean_path = stage_00_path(config, "intermediate_outputs", "pages_clean_parquet")
    df_clean = read_required_parquet(clean_path, "clean")
    inspection_config = stage_00_config(config).get("inspection", {})
    inspection = inspect_cleaned_text(
        df_clean,
        sample_size or int(inspection_config.get("sample_size", 5)),
        window_size or int(inspection_config.get("window_size", 100)),
        (
            random_state
            if random_state is not None
            else int(inspection_config.get("random_state", 42))
        ),
        document_id,
        page_number,
    )
    output_path = stage_00_path(config, "outputs", "cleaning_inspection_csv")
    inspection.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    inspection.attrs["outputs"] = [output_path]
    inspection.attrs["sample_size"] = sample_size or int(
        inspection_config.get("sample_size", 5)
    )
    inspection.attrs["window_size"] = window_size or int(
        inspection_config.get("window_size", 100)
    )
    inspection.attrs["random_state"] = (
        random_state
        if random_state is not None
        else int(inspection_config.get("random_state", 42))
    )
    inspection.attrs["document_id"] = document_id
    inspection.attrs["page_number"] = page_number
    return inspection


def step_inspect_footnotes(
    config: dict[str, Any],
    sample_size: int | None = None,
    random_state: int | None = None,
    document_id: str | None = None,
    page_number: int | None = None,
) -> pd.DataFrame:
    """Load extracted footnotes and return a footnote inspection table."""
    footnotes_path = stage_00_path(config, "intermediate_outputs", "footnotes_csv")
    footnotes = read_required_csv(footnotes_path, "clean")
    inspection_config = stage_00_config(config).get("inspection", {})
    inspection = inspect_footnotes(
        footnotes,
        sample_size or int(inspection_config.get("sample_size", 5)),
        (
            random_state
            if random_state is not None
            else int(inspection_config.get("random_state", 42))
        ),
        document_id,
        page_number,
    )
    output_path = stage_00_path(config, "outputs", "footnotes_inspection_csv")
    inspection.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    inspection.attrs["outputs"] = [output_path]
    inspection.attrs["total_footnotes"] = len(footnotes)
    inspection.attrs["sample_size"] = sample_size or int(
        inspection_config.get("sample_size", 5)
    )
    inspection.attrs["random_state"] = (
        random_state
        if random_state is not None
        else int(inspection_config.get("random_state", 42))
    )
    inspection.attrs["document_id"] = document_id
    inspection.attrs["page_number"] = page_number
    return inspection


def step_segment(
    config: dict[str, Any],
    limit_pages: int | None = None,
    write_csv: bool | None = None,
) -> pd.DataFrame:
    """Load clean pages, segment text into sentences, and write sentences."""
    clean_path = stage_00_path(config, "intermediate_outputs", "pages_clean_parquet")
    df_clean = read_required_parquet(clean_path, "clean")
    if limit_pages is not None:
        df_clean = df_clean.head(limit_pages)
    stage_config = stage_00_config(config)
    nlp = load_spacy_model(stage_config.get("segmentation", {}))
    sentences = build_sentence_table(
        df_clean, nlp, stage_config.get("segmentation", {})
    )
    output_path = stage_00_path(config, "intermediate_outputs", "sentences_parquet")
    outputs = write_parquet_with_optional_csv(
        sentences, output_path, debug_output_config(config, write_csv)
    )
    sentences.attrs["outputs"] = outputs
    sentences.attrs["pages_processed"] = len(df_clean)
    sentences.attrs["limit_pages"] = limit_pages
    return sentences


def step_annotate(
    config: dict[str, Any],
    limit_sentences: int | None = None,
    write_csv: bool | None = None,
) -> pd.DataFrame:
    """Load segmented sentences, add spaCy annotations, and write annotations."""
    sentences_path = stage_00_path(config, "intermediate_outputs", "sentences_parquet")
    sentences = read_required_parquet(sentences_path, "segment")
    if limit_sentences is not None:
        sentences = sentences.head(limit_sentences)
    stage_config = stage_00_config(config)
    nlp = load_spacy_model(stage_config.get("segmentation", {}))
    annotated = add_linguistic_annotations(
        sentences,
        nlp,
        int(stage_config.get("segmentation", {}).get("nlp_batch_size", 500)),
    )
    output_path = stage_00_path(
        config, "intermediate_outputs", "sentences_annotated_parquet"
    )
    outputs = write_parquet_with_optional_csv(
        annotated, output_path, debug_output_config(config, write_csv)
    )
    annotated.attrs["outputs"] = outputs
    annotated.attrs["limit_sentences"] = limit_sentences
    return annotated


def step_build(config: dict[str, Any], write_csv: bool | None = None) -> pd.DataFrame:
    """Build final N0 dataframe from annotated sentences."""
    annotated_path = stage_00_path(
        config, "intermediate_outputs", "sentences_annotated_parquet"
    )
    annotated = read_required_parquet(annotated_path, "annotate")
    df_n0 = build_n0_dataframe(annotated, config["corpus"]["id"])
    output_path = stage_00_path(config, "outputs", "corpus_parquet")
    outputs = write_parquet_with_optional_csv(
        df_n0, output_path, debug_output_config(config, write_csv)
    )
    df_n0.attrs["outputs"] = outputs
    return df_n0


def load_processed_files_for_metadata(config: dict[str, Any]) -> list[Path]:
    """Load processed file paths from discovery output for metadata export."""
    discovered_path = stage_00_path(
        config, "intermediate_outputs", "discovered_files_csv"
    )
    df_files = read_required_csv(discovered_path, "discover")
    return paths_from_discovered_files(df_files)


def inspect_extract_output(
    config: dict[str, Any],
    sample_size: int | None = None,
    window_size: int | None = None,
    random_state: int | None = None,
    document_id: str | None = None,
    page_number: int | None = None,
) -> pd.DataFrame:
    """Inspect raw extracted page text."""
    raw_path = stage_00_path(config, "intermediate_outputs", "pages_raw_parquet")
    df_raw = read_required_parquet(raw_path, "extract")
    inspection_config = stage_00_config(config).get("inspection", {})
    inspection = inspect_cleaned_text(
        df_raw,
        sample_size or int(inspection_config.get("sample_size", 5)),
        window_size or int(inspection_config.get("window_size", 100)),
        (
            random_state
            if random_state is not None
            else int(inspection_config.get("random_state", 42))
        ),
        document_id,
        page_number,
    )
    output_path = stage_00_path(config, "outputs", "pages_raw_inspection_csv")
    inspection.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    inspection.attrs.update({"outputs": [output_path], "source_rows": len(df_raw)})
    return inspection


def inspect_chapters_output(
    config: dict[str, Any], sample_size: int | None = None
) -> pd.DataFrame:
    """Inspect detected chapter assignments."""
    chapters_path = stage_00_path(
        config, "intermediate_outputs", "pages_with_chapters_parquet"
    )
    df_chapters = read_required_parquet(chapters_path, "detect-chapters")
    sample_size = sample_size or int(
        stage_00_config(config).get("inspection", {}).get("sample_size", 5)
    )
    columns = [
        "ID_documento",
        "archivo",
        "volumen",
        "pagina",
        "capitulo",
        "metodo_capitulo",
    ]
    inspection = df_chapters[columns].head(sample_size).copy()
    output_path = stage_00_path(config, "outputs", "chapters_inspection_csv")
    inspection.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    inspection.attrs.update({"outputs": [output_path], "source_rows": len(df_chapters)})
    return inspection


def inspect_segmentation_output(
    config: dict[str, Any],
    sample_size: int | None = None,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Inspect segmented sentence rows."""
    sentences_path = stage_00_path(config, "intermediate_outputs", "sentences_parquet")
    df_sentences = read_required_parquet(sentences_path, "segment")
    inspection_config = stage_00_config(config).get("inspection", {})
    sample_size = sample_size or int(inspection_config.get("sample_size", 5))
    window_size = window_size or int(inspection_config.get("window_size", 100))
    inspection = df_sentences.head(sample_size).copy()
    if "oracion_texto" in inspection:
        inspection["oracion_texto_preview"] = (
            inspection["oracion_texto"].fillna("").astype(str).str.slice(0, window_size)
        )
    output_path = stage_00_path(config, "outputs", "segmentation_inspection_csv")
    inspection.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    inspection.attrs.update(
        {"outputs": [output_path], "source_rows": len(df_sentences)}
    )
    return inspection


def inspect_annotations_output(
    config: dict[str, Any],
    sample_size: int | None = None,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Inspect annotated sentence rows."""
    annotated_path = stage_00_path(
        config, "intermediate_outputs", "sentences_annotated_parquet"
    )
    df_annotated = read_required_parquet(annotated_path, "annotate")
    inspection_config = stage_00_config(config).get("inspection", {})
    sample_size = sample_size or int(inspection_config.get("sample_size", 5))
    window_size = window_size or int(inspection_config.get("window_size", 100))
    columns = [
        column
        for column in [
            "ID_documento",
            "ID_oracion",
            "pagina",
            "oracion_texto",
            "tokens",
            "pos_tags",
            "entidades_NER",
        ]
        if column in df_annotated.columns
    ]
    inspection = df_annotated[columns].head(sample_size).copy()
    if "oracion_texto" in inspection:
        inspection["oracion_texto_preview"] = (
            inspection["oracion_texto"].fillna("").astype(str).str.slice(0, window_size)
        )
    output_path = stage_00_path(config, "outputs", "annotations_inspection_csv")
    inspection.to_csv(ensure_parent(output_path), index=False, encoding="utf-8-sig")
    inspection.attrs.update(
        {"outputs": [output_path], "source_rows": len(df_annotated)}
    )
    return inspection


def write_stage_00_outputs(
    df_n0: pd.DataFrame,
    metadata: dict[str, Any],
    document_summary_table: pd.DataFrame,
    chapter_summary_table: pd.DataFrame,
    footnotes: pd.DataFrame,
    outputs_config: dict[str, str],
) -> dict[str, Path]:
    """Write stage 00 outputs and return their resolved paths."""
    paths = {name: project_path(path) for name, path in outputs_config.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    df_n0.to_csv(paths["corpus_csv"], index=False, encoding="utf-8-sig")
    if "corpus_parquet" in paths:
        df_n0.to_parquet(paths["corpus_parquet"], index=False)
    paths["metadata_json"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    document_summary_table.to_csv(
        paths["document_summary_csv"], index=False, encoding="utf-8-sig"
    )
    chapter_summary_table.to_csv(
        paths["chapter_summary_csv"], index=False, encoding="utf-8-sig"
    )
    footnotes.to_csv(paths["footnotes_csv"], index=False, encoding="utf-8-sig")
    return paths


def step_export(config: dict[str, Any]) -> dict[str, Path]:
    """Load built N0/intermediates and write final CSV, metadata, and summaries."""
    n0_path = stage_00_path(config, "outputs", "corpus_parquet")
    df_n0 = read_required_parquet(n0_path, "build")
    footnotes_path = stage_00_path(config, "intermediate_outputs", "footnotes_csv")
    footnotes = (
        pd.read_csv(footnotes_path) if footnotes_path.exists() else pd.DataFrame()
    )
    metadata = corpus_metadata(
        df_n0,
        config["corpus"],
        load_processed_files_for_metadata(config),
        nlp=None,
    )
    paths = write_stage_00_outputs(
        df_n0,
        metadata,
        summary_by_document(df_n0),
        summary_by_chapter(df_n0),
        footnotes,
        stage_00_config(config)["outputs"],
    )
    paths["_row_counts"] = {
        "corpus": len(df_n0),
        "footnotes": len(footnotes),
        "summary_by_document": len(summary_by_document(df_n0)),
        "summary_by_chapter": len(summary_by_chapter(df_n0)),
    }
    paths["_summary"] = {
        "documents": int(df_n0["ID_documento"].nunique()),
        "sentences": len(df_n0),
        "words": int(df_n0["n_palabras"].sum()),
        "chapters": int(df_n0["capitulo"].nunique()),
    }
    return paths


def summarise_discovery(df_files: pd.DataFrame) -> str:
    """Return a compact discovery summary."""
    configured = (
        int(df_files["has_configured_metadata"].sum()) if not df_files.empty else 0
    )
    lines = [
        "Stage 00 / discover",
        f"Corpus directory: {df_files.attrs.get('corpus_dir', '')}",
        f"Extensions: {', '.join(df_files.attrs.get('extensions', []))}",
        f"Files discovered: {len(df_files):,}",
        f"Metadata configured: {configured}/{len(df_files)}",
        "Files:",
    ]
    lines.extend(f"  - {name}" for name in df_files["archivo"].tolist())
    missing = df_files.loc[~df_files["has_configured_metadata"], "archivo"].tolist()
    if missing:
        lines.append("Warning: files without configured metadata:")
        lines.extend(f"  - {name}" for name in missing)
    if df_files.attrs.get("limit_files") is not None:
        lines.append(f"Limit applied: files={df_files.attrs['limit_files']}")
    lines.extend(output_lines(df_files.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_extraction(df_pages: pd.DataFrame) -> str:
    """Return a compact extraction summary."""
    total_chars = int(df_pages["n_caracteres"].sum()) if not df_pages.empty else 0
    avg_chars = total_chars / len(df_pages) if len(df_pages) else 0
    documents = df_pages["ID_documento"].nunique() if not df_pages.empty else 0
    lines = [
        "Stage 00 / extract",
        f"Documents processed: {documents:,}",
        f"Pages extracted: {len(df_pages):,}",
        f"Total characters: {total_chars:,}",
        f"Average characters per page: {avg_chars:,.1f}",
    ]
    lines.extend(
        format_counts(df_pages["ID_documento"].value_counts(), "Pages by document:")
    )
    if df_pages.attrs.get("limit_files") is not None:
        lines.append(f"Limit applied: files={df_pages.attrs['limit_files']}")
    if df_pages.attrs.get("limit_pages") is not None:
        lines.append(f"Limit applied: pages={df_pages.attrs['limit_pages']}")
    lines.extend(output_lines(df_pages.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_chapter_detection(df_pages: pd.DataFrame) -> str:
    """Return a compact chapter-detection summary."""
    fallback_pages = int((df_pages.get("metodo_capitulo") == "fallback_filename").sum())
    fallback_pct = fallback_pages / len(df_pages) * 100 if len(df_pages) else 0
    documents = df_pages["ID_documento"].nunique() if not df_pages.empty else 0
    chapters = df_pages["capitulo"].nunique() if "capitulo" in df_pages else 0
    lines = [
        "Stage 00 / detect-chapters",
        f"Pages processed: {len(df_pages):,}",
        f"Documents: {documents:,}",
        f"Unique chapters: {chapters:,}",
    ]
    lines.extend(
        format_counts(df_pages["metodo_capitulo"].value_counts(), "Detection methods:")
    )
    lines.append(f"Fallback pages: {fallback_pages:,} ({fallback_pct:.1f}%)")
    lines.extend(
        format_counts(df_pages["capitulo"].value_counts(), "Top chapters:", 10)
    )
    if df_pages.attrs.get("limit_pages") is not None:
        lines.append(f"Limit applied: pages={df_pages.attrs['limit_pages']}")
    lines.extend(output_lines(df_pages.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_cleaning(result: tuple[pd.DataFrame, pd.DataFrame]) -> str:
    """Return a compact cleaning summary."""
    df_clean, footnotes = result
    before_pages = int(df_clean.attrs.get("pages_before_cleaning", len(df_clean)))
    before_chars = int(df_clean.attrs.get("characters_before_cleaning", 0))
    after_chars = int(df_clean["n_caracteres"].sum()) if not df_clean.empty else 0
    reduction = (1 - after_chars / before_chars) * 100 if before_chars else 0
    lines = [
        "Stage 00 / clean",
        f"Pages before cleaning: {before_pages:,}",
        f"Pages after cleaning: {len(df_clean):,}",
        f"Pages removed: {before_pages - len(df_clean):,}",
        f"Characters before: {before_chars:,}",
        f"Characters after: {after_chars:,}",
        f"Character reduction: {reduction:.1f}%",
        f"Footnotes extracted: {len(footnotes):,}",
    ]
    excluded = before_pages - len(df_clean)
    if excluded:
        lines.extend(
            format_counts(
                df_clean["ID_documento"].value_counts(), "Documents retained:"
            )
        )
    if df_clean.attrs.get("limit_pages") is not None:
        lines.append(f"Limit applied: pages={df_clean.attrs['limit_pages']}")
    lines.extend(output_lines(df_clean.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_cleaning_inspection(df: pd.DataFrame) -> str:
    """Return a compact cleaning-inspection summary."""
    lines = [
        "Stage 00 / inspect-cleaning",
        f"Inspected pages: {len(df):,}",
        f"Sample size used: {df.attrs.get('sample_size', '')}",
        f"Window size used: {df.attrs.get('window_size', '')}",
        f"Random state used: {df.attrs.get('random_state', '')}",
        f"Document filter: {df.attrs.get('document_id') or 'none'}",
        f"Page filter: {df.attrs.get('page_number') or 'none'}",
    ]
    lines.extend(output_lines(df.attrs.get("outputs", [])))
    for _, row in df.head(5).iterrows():
        lines.append(
            f"  - {row.get('ID_documento')} p.{row.get('pagina')} "
            f"({row.get('n_caracteres')} chars): "
            f"{str(row.get('inicio', ''))[:120]} ... {str(row.get('final', ''))[:120]}"
        )
    return "\n".join(lines)


def summarise_footnote_inspection(df: pd.DataFrame) -> str:
    """Return a compact footnote-inspection summary."""
    lines = [
        "Stage 00 / inspect-footnotes",
        f"Total footnotes available: {df.attrs.get('total_footnotes', len(df)):,}",
        f"Sampled footnotes: {len(df):,}",
        f"Document filter: {df.attrs.get('document_id') or 'none'}",
        f"Page filter: {df.attrs.get('page_number') or 'none'}",
    ]
    lines.extend(output_lines(df.attrs.get("outputs", [])))
    for _, row in df.head(5).iterrows():
        lines.append(
            f"  - {row.get('ID_documento', row.get('archivo'))} p.{row.get('pagina')}: "
            f"{str(row.get('nota_al_pie', ''))[:200]}"
        )
    return "\n".join(lines)


def summarise_segmentation(df_sentences: pd.DataFrame) -> str:
    """Return a compact segmentation summary."""
    pages = int(df_sentences.attrs.get("pages_processed", 0))
    avg_per_page = len(df_sentences) / pages if pages else 0
    avg_words = df_sentences["n_palabras"].mean() if not df_sentences.empty else 0
    min_chars = int(df_sentences["n_caracteres"].min()) if not df_sentences.empty else 0
    max_chars = int(df_sentences["n_caracteres"].max()) if not df_sentences.empty else 0
    lines = [
        "Stage 00 / segment",
        f"Pages processed: {pages:,}",
        f"Sentences generated: {len(df_sentences):,}",
        f"Average sentences per page: {avg_per_page:.1f}",
        f"Average words per sentence: {avg_words:.1f}",
        f"Sentence length range: {min_chars}-{max_chars} chars",
    ]
    lines.extend(
        format_counts(
            df_sentences["ID_documento"].value_counts(), "Sentences by document:"
        )
    )
    if df_sentences.attrs.get("limit_pages") is not None:
        lines.append(f"Limit applied: pages={df_sentences.attrs['limit_pages']}")
    lines.extend(output_lines(df_sentences.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_annotations(df: pd.DataFrame) -> str:
    """Return a compact linguistic-annotation summary."""
    token_counts = (
        df["tokens"].apply(lambda value: len(json.loads(value)))
        if "tokens" in df
        else pd.Series(dtype=int)
    )
    pos_counter: Counter[str] = Counter()
    ner_rows = 0
    for pos_json in df.get("pos_tags", []):
        pos_counter.update(json.loads(pos_json))
    for ner_json in df.get("entidades_NER", []):
        if json.loads(ner_json):
            ner_rows += 1
    total_tokens = int(token_counts.sum()) if len(token_counts) else 0
    average_tokens = token_counts.mean() if len(token_counts) else 0
    lines = [
        "Stage 00 / annotate",
        f"Sentences annotated: {len(df):,}",
        f"Tokens generated: {total_tokens:,}",
        f"Average tokens per sentence: {average_tokens:.1f}",
        f"Rows with NER entities: {ner_rows:,}",
        "Top POS tags:",
    ]
    lines.extend(f"  - {tag}: {count:,}" for tag, count in pos_counter.most_common(10))
    if df.attrs.get("limit_sentences") is not None:
        lines.append(f"Limit applied: sentences={df.attrs['limit_sentences']}")
    lines.extend(output_lines(df.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_n0_build(df_n0: pd.DataFrame) -> str:
    """Return a compact final N0 build summary."""
    chapters = df_n0["capitulo"].nunique() if not df_n0.empty else 0
    words = int(df_n0["n_palabras"].sum()) if not df_n0.empty else 0
    average_words = df_n0["n_palabras"].mean() if not df_n0.empty else 0
    lines = [
        "Stage 00 / build",
        f"Rows in final N0 dataframe: {len(df_n0):,}",
        f"Corpus ID: {df_n0['ID_corpus'].iloc[0] if not df_n0.empty else ''}",
        f"Documents: {df_n0['ID_documento'].nunique() if not df_n0.empty else 0:,}",
        f"Chapters: {chapters:,}",
        f"Total words: {words:,}",
        f"Average words per sentence: {average_words:.1f}",
    ]
    lines.extend(output_lines(df_n0.attrs.get("outputs", [])))
    return "\n".join(lines)


def summarise_export(paths: dict[str, Any]) -> str:
    """Return a compact export summary."""
    lines = ["Stage 00 / export", "Exported files:"]
    for key, path in paths.items():
        if key.startswith("_"):
            continue
        lines.append(f"  - {key}: {path}")
    row_counts = paths.get("_row_counts", {})
    if row_counts:
        lines.append("Row counts:")
        lines.extend(f"  - {key}: {value:,}" for key, value in row_counts.items())
    summary = paths.get("_summary", {})
    if summary:
        lines.append("Summary:")
        lines.extend(f"  - {key}: {value:,}" for key, value in summary.items())
    return "\n".join(lines)


def summarise_generic_inspection(step: str, df: pd.DataFrame) -> str:
    """Return a compact summary for optional inspection steps."""
    lines = [
        f"Stage 00 / {step}",
        f"Source rows: {df.attrs.get('source_rows', len(df)):,}",
        f"Inspection rows: {len(df):,}",
    ]
    lines.extend(output_lines(df.attrs.get("outputs", [])))
    preview_cols = [
        column
        for column in ["ID_documento", "pagina", "capitulo", "oracion_texto_preview"]
        if column in df.columns
    ]
    if preview_cols and not df.empty:
        lines.append(df[preview_cols].head(5).to_string(index=False))
    return "\n".join(lines)


def summarise_stage_00_result(step: str, result: Any) -> str:
    """Dispatch summary formatting for a stage 00 step result."""
    if step == "discover":
        return summarise_discovery(result)
    if step == "extract":
        return summarise_extraction(result)
    if step == "detect-chapters":
        return summarise_chapter_detection(result)
    if step == "clean":
        return summarise_cleaning(result)
    if step == "inspect-cleaning":
        return summarise_cleaning_inspection(result)
    if step == "inspect-footnotes":
        return summarise_footnote_inspection(result)
    if step == "segment":
        return summarise_segmentation(result)
    if step == "annotate":
        return summarise_annotations(result)
    if step == "build":
        return summarise_n0_build(result)
    if step == "export":
        return summarise_export(result)
    if step.startswith("inspect-"):
        return summarise_generic_inspection(step, result)
    return f"Stage 00 / {step}\nComplete"


def run_stage_00_step(
    config: dict[str, Any],
    step: str,
    sample_size: int | None = None,
    window_size: int | None = None,
    random_state: int | None = None,
    document_id: str | None = None,
    page_number: int | None = None,
    limit_files: int | None = None,
    limit_pages: int | None = None,
    limit_sentences: int | None = None,
    write_csv: bool | None = None,
) -> Any:
    """Run one inspectable stage 00 step."""
    if step == "discover":
        return step_discover(config, limit_files=limit_files)
    if step == "extract":
        return step_extract(config, limit_files, limit_pages, write_csv)
    if step == "detect-chapters":
        return step_detect_chapters(config, limit_pages, write_csv)
    if step == "clean":
        return step_clean(config, limit_pages, write_csv)
    if step == "inspect-extract":
        return inspect_extract_output(
            config, sample_size, window_size, random_state, document_id, page_number
        )
    if step == "inspect-chapters":
        return inspect_chapters_output(config, sample_size)
    if step == "inspect-cleaning":
        return step_inspect_cleaning(
            config, sample_size, window_size, random_state, document_id, page_number
        )
    if step == "inspect-footnotes":
        return step_inspect_footnotes(
            config, sample_size, random_state, document_id, page_number
        )
    if step == "inspect-segmentation":
        return inspect_segmentation_output(config, sample_size, window_size)
    if step == "inspect-annotations":
        return inspect_annotations_output(config, sample_size, window_size)
    if step == "segment":
        return step_segment(config, limit_pages, write_csv)
    if step == "annotate":
        return step_annotate(config, limit_sentences, write_csv)
    if step == "build":
        return step_build(config, write_csv)
    if step == "export":
        return step_export(config)
    if step == "all":
        return run_stage_00_all(
            config, limit_files, limit_pages, limit_sentences, write_csv
        )
    raise ValueError(f"Unknown stage 00 step: {step}")


def run_stage_00_all(
    config: dict[str, Any],
    limit_files: int | None = None,
    limit_pages: int | None = None,
    limit_sentences: int | None = None,
    write_csv: bool | None = None,
) -> pd.DataFrame:
    """Run all executable stage 00 steps in order."""
    step_discover(config, limit_files)
    step_extract(config, limit_files, limit_pages, write_csv)
    step_detect_chapters(config, limit_pages, write_csv)
    step_clean(config, limit_pages, write_csv)
    step_segment(config, limit_pages, write_csv)
    step_annotate(config, limit_sentences, write_csv)
    df_n0 = step_build(config, write_csv)
    step_export(config)
    return df_n0


def run_stage_00(config: dict[str, Any]) -> pd.DataFrame:
    """Run corpus ingestion from configured raw files through N0 exports."""
    stage_config = stage_00_config(config)
    corpus_config = config["corpus"]
    files = discover_corpus_files(
        stage_config["inputs"]["corpus_dir"],
        stage_config["inputs"].get("file_extensions", [".pdf", ".txt"]),
    )
    pages = extract_pages(files, corpus_config, stage_config.get("extraction", {}))
    cleaned_pages, footnotes = clean_pages(pages, stage_config.get("cleaning", {}))
    nlp = load_spacy_model(stage_config.get("segmentation", {}))
    sentences = build_sentence_table(
        cleaned_pages, nlp, stage_config.get("segmentation", {})
    )
    annotated = add_linguistic_annotations(
        sentences,
        nlp,
        int(stage_config.get("segmentation", {}).get("nlp_batch_size", 500)),
    )
    df_n0 = build_n0_dataframe(annotated, corpus_config["id"])
    metadata = corpus_metadata(df_n0, corpus_config, files, nlp)
    write_stage_00_outputs(
        df_n0,
        metadata,
        summary_by_document(df_n0),
        summary_by_chapter(df_n0),
        footnotes,
        stage_config["outputs"],
    )
    return df_n0
