"""Visualisation utilities for the AI-MELT research pipeline."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - exercised only in lean environments
    sns = None

from ai_melt.paths import project_path
from ai_melt.primary_metaphors import load_approach_results

STAGE_00_VISUALISATION_STEPS = [
    "load",
    "corpus-overview",
    "document-distribution",
    "chapter-distribution",
    "sentence-lengths",
    "named-entities",
    "pos-distribution",
    "word-counts",
    "footnotes",
    "export-summary",
]
STAGE_00_VISUALISATION_EXECUTION_STEPS = STAGE_00_VISUALISATION_STEPS.copy()

if sns is not None:
    sns.set_style("whitegrid")
else:
    plt.style.use("ggplot")


def color_palette(name: str, n_colors: int) -> list[Any]:
    """Return a seaborn palette when available, otherwise a matplotlib palette."""
    if sns is not None:
        return list(sns.color_palette(name, n_colors))
    cmap = plt.get_cmap("tab20")
    return [cmap(index / max(n_colors, 1)) for index in range(n_colors)]


def draw_heatmap(data: pd.DataFrame, ax: Any) -> None:
    """Draw an annotated heatmap with seaborn when available."""
    if sns is not None:
        sns.heatmap(data, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
        return
    image = ax.imshow(data.values, cmap="YlOrRd")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(
        range(len(data.columns)), labels=data.columns, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(data.index)), labels=data.index)
    for row_index, row in enumerate(data.values):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, int(value), ha="center", va="center")


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert #RRGGBB colors to Plotly-compatible rgba strings."""
    if not isinstance(color, str) or not re.match(r"^#[0-9A-Fa-f]{6}$", color):
        return color
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def slugify(value: str, max_length: int = 30) -> str:
    """Return a filesystem-friendly slug for plot filenames."""
    return re.sub(r"[^a-zA-Z0-9]", "_", value[:max_length]).lower().strip("_")


def safe_json_loads(value: Any) -> list[Any]:
    """Parse JSON strings safely, returning an empty list for invalid values."""
    try:
        return json.loads(value) if isinstance(value, str) else []
    except (json.JSONDecodeError, TypeError):
        return []


def extract_entities_for_subset(df_subset: pd.DataFrame) -> list[tuple[str, str]]:
    """Extract NER tuples from a corpus subset."""
    entities = []
    for ner_json in df_subset.get("entidades_NER", []):
        for entity in safe_json_loads(ner_json):
            entities.append((entity.get("text", ""), entity.get("label", "")))
    return entities


def extract_pos_for_subset(df_subset: pd.DataFrame) -> list[str]:
    """Extract POS tags from a corpus subset."""
    pos_tags = []
    for pos_json in df_subset.get("pos_tags", []):
        pos_tags.extend(safe_json_loads(pos_json))
    return pos_tags


def extract_content_lemmas_for_subset(df_subset: pd.DataFrame) -> list[str]:
    """Extract content lemmas used by the legacy wordcloud visualisations."""
    lemmas = []
    for lemmas_json, pos_json in zip(
        df_subset.get("lemas", []), df_subset.get("pos_tags", []), strict=False
    ):
        for lemma, pos in zip(
            safe_json_loads(lemmas_json), safe_json_loads(pos_json), strict=False
        ):
            if pos in {"NOUN", "VERB", "ADJ"} and len(str(lemma)) > 2:
                lemmas.append(str(lemma).lower())
    return lemmas


def volume_color_map(df_n0: pd.DataFrame) -> dict[str, Any]:
    """Build a stable color map for volumes."""
    volumes = sorted(df_n0["volumen"].dropna().unique()) if "volumen" in df_n0 else []
    colors = color_palette("husl", max(len(volumes), 1))
    return {volume: colors[index] for index, volume in enumerate(volumes)}


def save_current_figure(path: Path) -> Path:
    """Save and close the current matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def stage_00_viz_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Stage 00 visualisation config section."""
    return config["stage_00_visualisation"]


def stage_00_viz_input_path(viz_config: dict[str, Any]) -> Path:
    """Return the configured N0 corpus CSV path."""
    inputs = viz_config.get("inputs", {})
    return project_path(inputs.get("corpus_csv", viz_config.get("input")))


def stage_00_footnotes_input_path(config: dict[str, Any]) -> Path:
    """Return the configured footnotes table path, preferring exported output."""
    viz_config = stage_00_viz_config(config)
    inputs = viz_config.get("inputs", {})
    if "footnotes_csv" in inputs:
        return project_path(inputs["footnotes_csv"])
    stage_outputs = config.get("stage_00", {}).get("outputs", {})
    if "footnotes_csv" in stage_outputs:
        return project_path(stage_outputs["footnotes_csv"])
    stage_intermediate = config.get("stage_00", {}).get("intermediate_outputs", {})
    return project_path(
        stage_intermediate.get("footnotes_csv", "outputs/tables/n0_footnotes.csv")
    )


def stage_00_viz_output_dirs(viz_config: dict[str, Any]) -> dict[str, Path]:
    """Return configured visualisation output directories."""
    outputs = viz_config.get("outputs", {})
    return {
        "figures": project_path(outputs.get("figures_dir", "outputs/figures")),
        "tables": project_path(outputs.get("tables_dir", "outputs/tables")),
        "html": project_path(outputs.get("html_dir", "outputs/html")),
    }


def stage_00_table_path(viz_config: dict[str, Any], key: str) -> Path:
    """Return a configured Stage 00 visualisation table path."""
    dirs = stage_00_viz_output_dirs(viz_config)
    return dirs["tables"] / viz_config["table_names"][key]


def stage_00_figure_path(
    viz_config: dict[str, Any], key: str, slug: str | None = None
) -> Path:
    """Return a configured Stage 00 visualisation figure path."""
    dirs = stage_00_viz_output_dirs(viz_config)
    name = viz_config["figure_names"][key]
    if slug is not None:
        name = name.format(slug=slug)
    return dirs["figures"] / name


def write_stage_00_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a Stage 00 visualisation summary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_stage_00_visualisation_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load the configured N0 corpus CSV."""
    path = stage_00_viz_input_path(stage_00_viz_config(config))
    return pd.read_csv(path)


def stage_00_volumes(df_n0: pd.DataFrame) -> list[str]:
    """Return the notebook-compatible sorted volume list."""
    if "volumen" not in df_n0:
        return ["corpus"]
    return sorted(df_n0["volumen"].dropna().unique())


def corpus_overview_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise corpus-level counts shown by the legacy notebook."""
    words = int(df_n0["n_palabras"].sum()) if "n_palabras" in df_n0 else 0
    sentences = len(df_n0)
    return pd.DataFrame(
        [
            {
                "total_documents": (
                    int(df_n0["ID_documento"].nunique())
                    if "ID_documento" in df_n0
                    else int(df_n0["volumen"].nunique()) if "volumen" in df_n0 else 0
                ),
                "total_chapters": (
                    int(df_n0["capitulo"].nunique()) if "capitulo" in df_n0 else 0
                ),
                "total_sentences": sentences,
                "total_words": words,
                "average_words_per_sentence": (words / sentences if sentences else 0),
            }
        ]
    )


def document_distribution_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence and word counts by document/volume."""
    group_cols = [col for col in ["ID_documento", "volumen"] if col in df_n0]
    if not group_cols:
        group_cols = ["_document"]
        df_n0 = df_n0.assign(_document="corpus")
    return (
        df_n0.groupby(group_cols, dropna=False)
        .agg(
            n_oraciones=("ID_oracion", "count"),
            n_palabras=("n_palabras", "sum"),
        )
        .reset_index()
        .sort_values("n_oraciones", ascending=False)
    )


def chapter_distribution_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence and word counts by volume and chapter."""
    return (
        df_n0.groupby(["volumen", "capitulo"], dropna=False)
        .agg(
            n_oraciones=("ID_oracion", "count"),
            n_palabras=("n_palabras", "sum"),
            primera_pagina=("pagina", "min"),
        )
        .reset_index()
    )


def sentence_length_summary_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence length distributions for corpus and each volume."""
    rows = []
    subsets = [("CORPUS COMPLETO", df_n0)]
    if "volumen" in df_n0:
        subsets.extend(
            (str(volume), df_n0[df_n0["volumen"] == volume])
            for volume in stage_00_volumes(df_n0)
        )
    for label, subset in subsets:
        lengths = subset["n_palabras"]
        rows.append(
            {
                "scope": label,
                "sentences": len(subset),
                "min_words": int(lengths.min()) if not lengths.empty else 0,
                "mean_words": float(lengths.mean()) if not lengths.empty else 0,
                "median_words": float(lengths.median()) if not lengths.empty else 0,
                "max_words": int(lengths.max()) if not lengths.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def word_count_summary_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Summarise word counts by corpus, document, and chapter."""
    rows = [
        {
            "scope": "corpus",
            "items": 1,
            "total_words": int(df_n0["n_palabras"].sum()),
            "average_words": float(df_n0["n_palabras"].sum()),
        }
    ]
    if "ID_documento" in df_n0:
        by_doc = df_n0.groupby("ID_documento")["n_palabras"].sum()
        rows.append(
            {
                "scope": "document",
                "items": int(len(by_doc)),
                "total_words": int(by_doc.sum()),
                "average_words": float(by_doc.mean()) if not by_doc.empty else 0,
            }
        )
    if "capitulo" in df_n0:
        by_chapter = df_n0.groupby(["volumen", "capitulo"])["n_palabras"].sum()
        rows.append(
            {
                "scope": "chapter",
                "items": int(len(by_chapter)),
                "total_words": int(by_chapter.sum()),
                "average_words": (
                    float(by_chapter.mean()) if not by_chapter.empty else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_corpus_size_by_volume(
    df_n0: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Plot sentence and word counts by volume and chapter."""
    config = config or {}
    names = config.get("figure_names", {})
    output_dir = project_path(output_dir)
    colors = volume_color_map(df_n0)
    volumes = sorted(df_n0["volumen"].dropna().unique())
    written = []

    vol_stats = (
        df_n0.groupby("volumen")
        .agg(n_oraciones=("ID_oracion", "count"), n_palabras=("n_palabras", "sum"))
        .reindex(volumes)
    )
    fig, ax = plt.subplots(figsize=(12, max(4, len(volumes) * 0.8)))
    ax.barh(
        [volume[:40] for volume in volumes],
        vol_stats["n_oraciones"].values,
        color=[colors[volume] for volume in volumes],
    )
    ax.set_xlabel("Oraciones")
    ax.set_title("Oraciones por volumen")
    ax.tick_params(axis="y", labelsize=8)
    written.append(
        save_current_figure(
            output_dir
            / names.get("sentences_by_volume", "viz_oraciones_por_volumen.png")
        )
    )

    fig, ax = plt.subplots(figsize=(12, max(4, len(volumes) * 0.8)))
    ax.barh(
        [volume[:40] for volume in volumes],
        vol_stats["n_palabras"].values,
        color=[colors[volume] for volume in volumes],
    )
    ax.set_xlabel("Palabras")
    ax.set_title("Palabras por volumen")
    ax.tick_params(axis="y", labelsize=8)
    written.append(
        save_current_figure(
            output_dir / names.get("words_by_volume", "viz_palabras_por_volumen.png")
        )
    )

    cap_stats = (
        df_n0.groupby(["volumen", "capitulo"])
        .agg(
            n_oraciones=("ID_oracion", "count"),
            n_palabras=("n_palabras", "sum"),
            primera_pagina=("pagina", "min"),
        )
        .reset_index()
    )
    top_chapters = int(config.get("top_chapters", 30))
    display_caps = cap_stats.sort_values("n_oraciones").tail(top_chapters)
    fig, ax = plt.subplots(figsize=(14, max(8, len(display_caps) * 0.45)))
    labels = [
        f"{row['capitulo'][:40]}  [{row['volumen'][:20]}]"
        for _, row in display_caps.iterrows()
    ]
    ax.barh(
        labels,
        display_caps["n_oraciones"].values,
        color=[
            colors.get(row["volumen"], "#888") for _, row in display_caps.iterrows()
        ],
    )
    ax.set_xlabel("Oraciones")
    ax.set_title(
        f"Oraciones por capítulo — todos los volúmenes (top {len(display_caps)})"
    )
    ax.tick_params(axis="y", labelsize=7)
    written.append(
        save_current_figure(
            output_dir
            / names.get(
                "sentences_by_chapter_global",
                "viz_oraciones_por_capitulo_global.png",
            )
        )
    )

    display_words = cap_stats.sort_values("n_palabras").tail(top_chapters)
    fig, ax = plt.subplots(figsize=(14, max(8, len(display_words) * 0.45)))
    labels = [
        f"{row['capitulo'][:40]}  [{row['volumen'][:20]}]"
        for _, row in display_words.iterrows()
    ]
    ax.barh(
        labels,
        display_words["n_palabras"].values,
        color=[
            colors.get(row["volumen"], "#888") for _, row in display_words.iterrows()
        ],
    )
    ax.set_xlabel("Palabras")
    ax.set_title(
        f"Palabras por capítulo — todos los volúmenes (top {len(display_words)})"
    )
    ax.tick_params(axis="y", labelsize=7)
    written.append(
        save_current_figure(
            output_dir
            / names.get(
                "words_by_chapter_global", "viz_palabras_por_capitulo_global.png"
            )
        )
    )
    return written


def plot_sentence_length_distributions(
    df_n0: pd.DataFrame, output_dir: str | Path, config: dict[str, Any] | None = None
) -> list[Path]:
    """Plot corpus-level and volume-level sentence length distributions."""
    config = config or {}
    names = config.get("figure_names", {})
    output_dir = project_path(output_dir)
    colors = volume_color_map(df_n0)
    written = []

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df_n0["n_palabras"], bins=50, color="#3B8BD4", edgecolor="white", alpha=0.8)
    median = df_n0["n_palabras"].median()
    ax.axvline(median, color="red", linestyle="--", label=f"Mediana: {median:.0f}")
    ax.set_title("Distribución de longitud de oraciones — CORPUS COMPLETO")
    ax.set_xlabel("Palabras por oración")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    written.append(
        save_current_figure(
            output_dir / names.get("sentence_length_corpus", "viz_longitud_corpus.png")
        )
    )

    volumes = sorted(df_n0["volumen"].dropna().unique())
    if len(volumes) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        data_by_volume = [
            df_n0.loc[df_n0["volumen"] == volume, "n_palabras"] for volume in volumes
        ]
        boxplot = ax.boxplot(
            data_by_volume,
            labels=[volume[:25] for volume in volumes],
            patch_artist=True,
        )
        for patch, color in zip(boxplot["boxes"], colors.values(), strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Volumen")
        ax.set_ylabel("Palabras por oración")
        ax.set_title("Distribución de longitud de oraciones por volumen")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        written.append(
            save_current_figure(
                output_dir
                / names.get("sentence_length_by_volume", "viz_longitud_por_volumen.png")
            )
        )
    return written


def plot_named_entities(
    df_subset: pd.DataFrame,
    output_dir: str | Path,
    label: str = "CORPUS COMPLETO",
    color: str = "#D85A30",
    slug: str = "corpus",
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Plot top named entities and NER type distribution."""
    config = config or {}
    names = config.get("figure_names", {})
    output_dir = project_path(output_dir)
    entities = extract_entities_for_subset(df_subset)
    if not entities:
        return []
    df_entities = pd.DataFrame(entities, columns=["text", "label"])
    written = []

    top = df_entities["text"].value_counts().head(int(config.get("top_entities", 20)))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top.index[::-1], top.values[::-1], color=color)
    ax.set_xlabel("Frecuencia")
    ax.set_title(f"Top 20 entidades nombradas — {label}")
    written.append(
        save_current_figure(
            output_dir
            / names.get("named_entities_top", "viz_NER_top20_{slug}.png").format(
                slug=slug
            )
        )
    )

    type_counts = df_entities["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        colors=color_palette("husl", len(type_counts)),
        startangle=90,
    )
    ax.set_title(f"Distribución por tipo NER — {label}")
    written.append(
        save_current_figure(
            output_dir
            / names.get("named_entities_types", "viz_NER_tipos_{slug}.png").format(
                slug=slug
            )
        )
    )
    return written


def plot_pos_distribution(
    df_subset: pd.DataFrame,
    output_dir: str | Path,
    label: str = "CORPUS COMPLETO",
    color: str = "#3B8BD4",
    slug: str = "corpus",
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Plot POS distribution for a corpus subset."""
    config = config or {}
    names = config.get("figure_names", {})
    pos_top = pd.DataFrame(
        Counter(extract_pos_for_subset(df_subset)).most_common(
            int(config.get("top_pos", 12))
        ),
        columns=["POS", "Freq"],
    )
    if pos_top.empty:
        return []
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pos_top["POS"], pos_top["Freq"], color=color, edgecolor="white")
    ax.set_title(f"Distribución POS — {label}")
    ax.set_xlabel("Categoría gramatical")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    return [
        save_current_figure(
            project_path(output_dir)
            / names.get("pos_distribution", "viz_pos_{slug}.png").format(slug=slug)
        )
    ]


def plot_corpus_wordcloud(
    df_subset: pd.DataFrame,
    output_dir: str | Path,
    slug: str = "corpus",
    label: str = "CORPUS COMPLETO",
    max_words: int = 150,
    width: int = 1200,
    height: int = 500,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Plot a wordcloud for content lemmas."""
    try:
        from wordcloud import WordCloud
    except ImportError:  # pragma: no cover - optional runtime
        return []

    config = config or {}
    names = config.get("figure_names", {})
    lemmas = extract_content_lemmas_for_subset(df_subset)
    if not lemmas:
        return []
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        max_words=max_words,
        colormap="viridis",
        collocations=False,
    ).generate(" ".join(lemmas))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Wordcloud — {label}", fontsize=14)
    return [
        save_current_figure(
            project_path(output_dir)
            / names.get("wordcloud", "viz_wordcloud_{slug}.png").format(slug=slug)
        )
    ]


def plot_document_distribution_step(
    df_n0: pd.DataFrame, viz_config: dict[str, Any]
) -> list[Path]:
    """Generate notebook volume-level sentence and word count figures."""
    colors = volume_color_map(df_n0)
    volumes = stage_00_volumes(df_n0)
    vol_stats = (
        df_n0.groupby("volumen")
        .agg(n_oraciones=("ID_oracion", "count"), n_palabras=("n_palabras", "sum"))
        .reindex(volumes)
    )
    written = []

    fig, ax = plt.subplots(figsize=tuple(viz_config["figure_sizes"]["volume_bars"]))
    ax.barh(
        [volume[:40] for volume in volumes],
        vol_stats["n_oraciones"].values,
        color=[colors[volume] for volume in volumes],
    )
    ax.set_xlabel("Oraciones")
    ax.set_title("Oraciones por volumen")
    ax.tick_params(axis="y", labelsize=8)
    written.append(
        save_current_figure(stage_00_figure_path(viz_config, "sentences_by_volume"))
    )

    fig, ax = plt.subplots(figsize=tuple(viz_config["figure_sizes"]["volume_bars"]))
    ax.barh(
        [volume[:40] for volume in volumes],
        vol_stats["n_palabras"].values,
        color=[colors[volume] for volume in volumes],
    )
    ax.set_xlabel("Palabras")
    ax.set_title("Palabras por volumen")
    ax.tick_params(axis="y", labelsize=8)
    written.append(
        save_current_figure(stage_00_figure_path(viz_config, "words_by_volume"))
    )
    return written


def plot_chapter_distribution_step(
    df_n0: pd.DataFrame, viz_config: dict[str, Any]
) -> list[Path]:
    """Generate notebook global and per-volume chapter distribution figures."""
    colors = volume_color_map(df_n0)
    cap_stats = chapter_distribution_table(df_n0)
    top_chapters = int(viz_config.get("top_chapters", 30))
    chapter_order = viz_config.get("chapter_order", "aparicion")
    written = []

    display_caps = cap_stats.sort_values("n_oraciones", ascending=True).tail(
        top_chapters
    )
    fig, ax = plt.subplots(figsize=(14, max(8, len(display_caps) * 0.45)))
    labels = [
        f"{row['capitulo'][:40]}  [{row['volumen'][:20]}]"
        for _, row in display_caps.iterrows()
    ]
    ax.barh(
        labels,
        display_caps["n_oraciones"].values,
        color=[
            colors.get(row["volumen"], "#888") for _, row in display_caps.iterrows()
        ],
    )
    ax.set_xlabel("Oraciones")
    ax.set_title(
        f"Oraciones por capitulo - todos los volumenes (top {len(display_caps)})"
    )
    ax.tick_params(axis="y", labelsize=7)
    written.append(
        save_current_figure(
            stage_00_figure_path(viz_config, "sentences_by_chapter_global")
        )
    )

    display_words = cap_stats.sort_values("n_palabras", ascending=True).tail(
        top_chapters
    )
    fig, ax = plt.subplots(figsize=(14, max(8, len(display_words) * 0.45)))
    labels = [
        f"{row['capitulo'][:40]}  [{row['volumen'][:20]}]"
        for _, row in display_words.iterrows()
    ]
    ax.barh(
        labels,
        display_words["n_palabras"].values,
        color=[
            colors.get(row["volumen"], "#888") for _, row in display_words.iterrows()
        ],
    )
    ax.set_xlabel("Palabras")
    ax.set_title(
        f"Palabras por capitulo - todos los volumenes (top {len(display_words)})"
    )
    ax.tick_params(axis="y", labelsize=7)
    written.append(
        save_current_figure(stage_00_figure_path(viz_config, "words_by_chapter_global"))
    )

    for volume in stage_00_volumes(df_n0):
        volume_data = cap_stats[cap_stats["volumen"] == volume].copy()
        if volume_data.empty:
            continue
        if chapter_order == "magnitud":
            sentences_data = volume_data.sort_values("n_oraciones", ascending=True)
            words_data = volume_data.sort_values("n_palabras", ascending=True)
        else:
            sentences_data = volume_data.sort_values("primera_pagina", ascending=False)
            words_data = volume_data.sort_values("primera_pagina", ascending=False)
        slug = slugify(str(volume))
        color = colors.get(volume, "#888")
        fig, ax = plt.subplots(figsize=(12, max(4, len(volume_data) * 0.5)))
        ax.barh(
            [chapter[:50] for chapter in sentences_data["capitulo"]],
            sentences_data["n_oraciones"].values,
            color=color,
        )
        ax.set_xlabel("Oraciones")
        ax.set_title(
            f"Oraciones por capitulo - {str(volume)[:50]} (orden: {chapter_order})"
        )
        ax.tick_params(axis="y", labelsize=7)
        written.append(
            save_current_figure(
                stage_00_figure_path(viz_config, "sentences_by_chapter_volume", slug)
            )
        )

        fig, ax = plt.subplots(figsize=(12, max(4, len(volume_data) * 0.5)))
        ax.barh(
            [chapter[:50] for chapter in words_data["capitulo"]],
            words_data["n_palabras"].values,
            color=color,
        )
        ax.set_xlabel("Palabras")
        ax.set_title(
            f"Palabras por capitulo - {str(volume)[:50]} (orden: {chapter_order})"
        )
        ax.tick_params(axis="y", labelsize=7)
        written.append(
            save_current_figure(
                stage_00_figure_path(viz_config, "words_by_chapter_volume", slug)
            )
        )
    return written


def plot_sentence_lengths_step(
    df_n0: pd.DataFrame, viz_config: dict[str, Any]
) -> list[Path]:
    """Generate notebook sentence-length histogram and boxplot figures."""
    colors = volume_color_map(df_n0)
    bins = viz_config.get("histogram_bins", {})
    written = []

    fig, ax = plt.subplots(figsize=tuple(viz_config["figure_sizes"]["sentence_hist"]))
    ax.hist(
        df_n0["n_palabras"],
        bins=int(bins.get("corpus_sentence_lengths", 50)),
        color="#3B8BD4",
        edgecolor="white",
        alpha=0.8,
    )
    median = df_n0["n_palabras"].median()
    ax.axvline(median, color="red", linestyle="--", label=f"Mediana: {median:.0f}")
    ax.set_title("Distribucion de longitud de oraciones - CORPUS COMPLETO")
    ax.set_xlabel("Palabras por oracion")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    written.append(
        save_current_figure(stage_00_figure_path(viz_config, "sentence_length_corpus"))
    )

    volumes = stage_00_volumes(df_n0)
    for volume in volumes:
        data = df_n0[df_n0["volumen"] == volume]["n_palabras"]
        fig, ax = plt.subplots(
            figsize=tuple(viz_config["figure_sizes"]["sentence_hist"])
        )
        ax.hist(
            data,
            bins=int(bins.get("volume_sentence_lengths", 40)),
            color=colors[volume],
            edgecolor="white",
            alpha=0.8,
        )
        median_volume = data.median()
        ax.axvline(
            median_volume,
            color="red",
            linestyle="--",
            label=f"Mediana: {median_volume:.0f}",
        )
        ax.set_title(f"Distribucion de longitud de oraciones - {str(volume)[:50]}")
        ax.set_xlabel("Palabras por oracion")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=9)
        written.append(
            save_current_figure(
                stage_00_figure_path(
                    viz_config, "sentence_length_volume", slugify(str(volume))
                )
            )
        )

    if len(volumes) > 1:
        fig, ax = plt.subplots(
            figsize=tuple(viz_config["figure_sizes"]["sentence_boxplot"])
        )
        data_by_volume = [
            df_n0.loc[df_n0["volumen"] == volume, "n_palabras"].values
            for volume in volumes
        ]
        boxplot = ax.boxplot(
            data_by_volume,
            labels=[volume[:25] for volume in volumes],
            patch_artist=True,
        )
        for patch, color in zip(boxplot["boxes"], colors.values(), strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Volumen")
        ax.set_ylabel("Palabras por oracion")
        ax.set_title("Distribucion de longitud de oraciones por volumen")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        written.append(
            save_current_figure(
                stage_00_figure_path(viz_config, "sentence_length_by_volume")
            )
        )
    return written


def named_entities_summary_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Build NER frequency summaries for corpus and each volume."""
    rows = []
    subsets = [("corpus", "CORPUS COMPLETO", df_n0)]
    subsets.extend(
        (slugify(str(volume)), str(volume), df_n0[df_n0["volumen"] == volume])
        for volume in stage_00_volumes(df_n0)
    )
    for slug, label, subset in subsets:
        entities = extract_entities_for_subset(subset)
        if not entities:
            continue
        counts = Counter(entity for entity, _ in entities)
        labels = Counter(entity_label for _, entity_label in entities)
        for entity, count in counts.most_common():
            rows.append(
                {
                    "scope_slug": slug,
                    "scope_label": label,
                    "summary_type": "entity",
                    "value": entity,
                    "count": count,
                }
            )
        for entity_label, count in labels.most_common():
            rows.append(
                {
                    "scope_slug": slug,
                    "scope_label": label,
                    "summary_type": "entity_type",
                    "value": entity_label,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def plot_named_entities_step(
    df_n0: pd.DataFrame, viz_config: dict[str, Any]
) -> list[Path]:
    """Generate notebook NER figures for corpus and each volume."""
    output_dir = stage_00_viz_output_dirs(viz_config)["figures"]
    written = []
    colors = volume_color_map(df_n0)
    written.extend(
        plot_named_entities(
            df_n0,
            output_dir,
            label="CORPUS COMPLETO",
            color="#D85A30",
            slug="corpus",
            config=viz_config,
        )
    )
    for volume in stage_00_volumes(df_n0):
        written.extend(
            plot_named_entities(
                df_n0[df_n0["volumen"] == volume],
                output_dir,
                label=str(volume)[:50],
                color=colors.get(volume, "#888"),
                slug=slugify(str(volume)),
                config=viz_config,
            )
        )
    return written


def pos_distribution_summary_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Build POS frequency summaries for corpus and each volume."""
    rows = []
    subsets = [("corpus", "CORPUS COMPLETO", df_n0)]
    subsets.extend(
        (slugify(str(volume)), str(volume), df_n0[df_n0["volumen"] == volume])
        for volume in stage_00_volumes(df_n0)
    )
    for slug, label, subset in subsets:
        pos_counts = Counter(extract_pos_for_subset(subset))
        for pos, count in pos_counts.most_common():
            rows.append(
                {
                    "scope_slug": slug,
                    "scope_label": label,
                    "pos": pos,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def plot_pos_distribution_step(
    df_n0: pd.DataFrame, viz_config: dict[str, Any]
) -> list[Path]:
    """Generate notebook POS figures for corpus and each volume."""
    output_dir = stage_00_viz_output_dirs(viz_config)["figures"]
    written = []
    colors = volume_color_map(df_n0)
    written.extend(
        plot_pos_distribution(
            df_n0,
            output_dir,
            label="CORPUS COMPLETO",
            color="#3B8BD4",
            slug="corpus",
            config=viz_config,
        )
    )
    for volume in stage_00_volumes(df_n0):
        written.extend(
            plot_pos_distribution(
                df_n0[df_n0["volumen"] == volume],
                output_dir,
                label=str(volume)[:50],
                color=colors.get(volume, "#888"),
                slug=slugify(str(volume)),
                config=viz_config,
            )
        )
    return written


def content_lemma_frequency_table(df_n0: pd.DataFrame) -> pd.DataFrame:
    """Build content-lemma frequencies used by the notebook wordclouds."""
    rows = []
    subsets = [("corpus", "CORPUS COMPLETO", df_n0)]
    subsets.extend(
        (slugify(str(volume)), str(volume), df_n0[df_n0["volumen"] == volume])
        for volume in stage_00_volumes(df_n0)
    )
    for slug, label, subset in subsets:
        lemma_counts = Counter(extract_content_lemmas_for_subset(subset))
        for lemma, count in lemma_counts.most_common():
            rows.append(
                {
                    "scope_slug": slug,
                    "scope_label": label,
                    "lemma": lemma,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def plot_word_counts_step(
    df_n0: pd.DataFrame, viz_config: dict[str, Any]
) -> list[Path]:
    """Generate notebook wordcloud figures for corpus and each volume."""
    output_dir = stage_00_viz_output_dirs(viz_config)["figures"]
    wordcloud_cfg = viz_config.get("wordcloud", {})
    written = []
    written.extend(
        plot_corpus_wordcloud(
            df_n0,
            output_dir,
            slug="corpus",
            label="CORPUS COMPLETO",
            max_words=int(wordcloud_cfg.get("corpus_max_words", 150)),
            width=int(wordcloud_cfg.get("corpus_width", 1200)),
            height=int(wordcloud_cfg.get("corpus_height", 500)),
            config=viz_config,
        )
    )
    for volume in stage_00_volumes(df_n0):
        subset = df_n0[df_n0["volumen"] == volume]
        lemmas = extract_content_lemmas_for_subset(subset)
        if len(lemmas) <= int(wordcloud_cfg.get("min_volume_lemmas", 10)):
            continue
        written.extend(
            plot_corpus_wordcloud(
                subset,
                output_dir,
                slug=slugify(str(volume)),
                label=str(volume)[:50],
                max_words=int(wordcloud_cfg.get("volume_max_words", 80)),
                width=int(wordcloud_cfg.get("volume_width", 800)),
                height=int(wordcloud_cfg.get("volume_height", 400)),
                config=viz_config,
            )
        )
    return written


def footnotes_summary_table(footnotes: pd.DataFrame) -> pd.DataFrame:
    """Summarise available Stage 00 footnotes by document and page."""
    if footnotes.empty:
        return pd.DataFrame(columns=["ID_documento", "archivo", "pagina", "footnotes"])
    group_cols = [
        column
        for column in ["ID_documento", "archivo", "volumen", "pagina"]
        if column in footnotes
    ]
    if not group_cols:
        return pd.DataFrame([{"footnotes": len(footnotes)}])
    return (
        footnotes.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="footnotes")
        .sort_values("footnotes", ascending=False)
    )


def export_summary_table(paths: list[Path]) -> pd.DataFrame:
    """Build a final listing of generated Stage 00 visualisation outputs."""
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def run_stage_00_visualisation_step(
    config: dict[str, Any], step: str
) -> dict[str, Any]:
    """Run one inspectable Stage 00 visualisation step."""
    viz_config = stage_00_viz_config(config)
    input_path = stage_00_viz_input_path(viz_config)
    if step == "load":
        df_n0 = load_stage_00_visualisation_data(config)
        return {
            "input": input_path,
            "_summary": {
                "rows": len(df_n0),
                "columns": list(df_n0.columns),
                "documents": (
                    int(df_n0["ID_documento"].nunique())
                    if "ID_documento" in df_n0
                    else 0
                ),
                "chapters": (
                    int(df_n0["capitulo"].nunique()) if "capitulo" in df_n0 else 0
                ),
                "total_words": (
                    int(df_n0["n_palabras"].sum()) if "n_palabras" in df_n0 else 0
                ),
            },
        }

    df_n0 = load_stage_00_visualisation_data(config)
    if step == "corpus-overview":
        table = corpus_overview_table(df_n0)
        path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "corpus_overview_csv")
        )
        return {"table": path, "_summary": table.iloc[0].to_dict()}
    if step == "document-distribution":
        table = document_distribution_table(df_n0)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "document_distribution_csv")
        )
        figures = plot_document_distribution_step(df_n0, viz_config)
        return {
            "table": table_path,
            "figures": figures,
            "_summary": {
                "documents": len(table),
                "total_sentences": int(table["n_oraciones"].sum()),
                "total_words": int(table["n_palabras"].sum()),
            },
        }
    if step == "chapter-distribution":
        table = chapter_distribution_table(df_n0)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "chapter_distribution_csv")
        )
        figures = plot_chapter_distribution_step(df_n0, viz_config)
        return {
            "table": table_path,
            "figures": figures,
            "_summary": {
                "chapters": len(table),
                "top_sentence_chapter": (
                    table.sort_values("n_oraciones", ascending=False)
                    .head(1)
                    .to_dict("records")
                ),
                "top_word_chapter": (
                    table.sort_values("n_palabras", ascending=False)
                    .head(1)
                    .to_dict("records")
                ),
            },
        }
    if step == "sentence-lengths":
        table = sentence_length_summary_table(df_n0)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "sentence_lengths_csv")
        )
        figures = plot_sentence_lengths_step(df_n0, viz_config)
        corpus_row = table[table["scope"] == "CORPUS COMPLETO"].iloc[0].to_dict()
        return {"table": table_path, "figures": figures, "_summary": corpus_row}
    if step == "named-entities":
        table = named_entities_summary_table(df_n0)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "named_entities_csv")
        )
        figures = plot_named_entities_step(df_n0, viz_config)
        return {
            "table": table_path,
            "figures": figures,
            "_summary": {"rows": len(table), "figures": len(figures)},
        }
    if step == "pos-distribution":
        table = pos_distribution_summary_table(df_n0)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "pos_distribution_csv")
        )
        figures = plot_pos_distribution_step(df_n0, viz_config)
        return {
            "table": table_path,
            "figures": figures,
            "_summary": {"rows": len(table), "figures": len(figures)},
        }
    if step == "word-counts":
        word_table = word_count_summary_table(df_n0)
        word_path = write_stage_00_table(
            word_table, stage_00_table_path(viz_config, "word_counts_csv")
        )
        lemma_table = content_lemma_frequency_table(df_n0)
        lemma_path = write_stage_00_table(
            lemma_table, stage_00_table_path(viz_config, "content_lemmas_csv")
        )
        figures = plot_word_counts_step(df_n0, viz_config)
        return {
            "word_table": word_path,
            "lemma_table": lemma_path,
            "figures": figures,
            "_summary": {
                "total_words": int(df_n0["n_palabras"].sum()),
                "avg_words_by_document": (
                    float(df_n0.groupby("ID_documento")["n_palabras"].sum().mean())
                    if "ID_documento" in df_n0
                    else 0
                ),
                "avg_words_by_chapter": (
                    float(
                        df_n0.groupby(["volumen", "capitulo"])["n_palabras"]
                        .sum()
                        .mean()
                    )
                    if {"volumen", "capitulo"}.issubset(df_n0.columns)
                    else 0
                ),
                "figures": len(figures),
            },
        }
    if step == "footnotes":
        footnote_path = stage_00_footnotes_input_path(config)
        footnotes = (
            pd.read_csv(footnote_path) if footnote_path.exists() else pd.DataFrame()
        )
        table = footnotes_summary_table(footnotes)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "footnotes_summary_csv")
        )
        return {
            "input": footnote_path,
            "table": table_path,
            "_summary": {
                "footnotes": len(footnotes),
                "groups": len(table),
                "input_exists": footnote_path.exists(),
            },
        }
    if step == "export-summary":
        outputs = []
        for directory in stage_00_viz_output_dirs(viz_config).values():
            if directory.exists():
                outputs.extend(
                    path
                    for path in directory.iterdir()
                    if path.is_file()
                    and path.name.startswith(("viz_", "n0_viz_", "n0_"))
                )
        table = export_summary_table(outputs)
        table_path = write_stage_00_table(
            table, stage_00_table_path(viz_config, "export_summary_csv")
        )
        return {
            "table": table_path,
            "_summary": {"outputs_listed": len(table)},
        }
    raise ValueError(f"Unknown Stage 00 visualisation step: {step}")


def visualise_corpus(df_n0: pd.DataFrame, config: dict[str, Any]) -> list[Path]:
    """Generate the main N0 corpus visualisations."""
    output_dir = config["outputs"]["figures_dir"]
    written = []
    written.extend(plot_corpus_size_by_volume(df_n0, output_dir, config))
    written.extend(plot_sentence_length_distributions(df_n0, output_dir, config))
    colors = volume_color_map(df_n0)
    written.extend(plot_named_entities(df_n0, output_dir, config=config))
    written.extend(plot_pos_distribution(df_n0, output_dir, config=config))
    wordcloud_cfg = config.get("wordcloud", {})
    written.extend(
        plot_corpus_wordcloud(
            df_n0,
            output_dir,
            max_words=int(wordcloud_cfg.get("corpus_max_words", 150)),
            config=config,
        )
    )
    for volume in sorted(df_n0["volumen"].dropna().unique()):
        subset = df_n0[df_n0["volumen"] == volume]
        slug = slugify(volume)
        color = colors.get(volume, "#888")
        written.extend(
            plot_named_entities(subset, output_dir, volume[:50], color, slug, config)
        )
        written.extend(
            plot_pos_distribution(subset, output_dir, volume[:50], color, slug, config)
        )
    return written


def stage_00_visualisation_output_paths(result: dict[str, Any]) -> list[Path]:
    """Extract output paths from a Stage 00 visualisation step result."""
    outputs = []
    for key, value in result.items():
        if key.startswith("_") or key == "input":
            continue
        if isinstance(value, Path):
            outputs.append(value)
        elif isinstance(value, list | tuple):
            outputs.extend(path for path in value if isinstance(path, Path))
    return outputs


def summarise_stage_00_visualisation_result(step: str, result: dict[str, Any]) -> str:
    """Return useful console output for a Stage 00 visualisation step."""
    summary = result.get("_summary", {})
    outputs = stage_00_visualisation_output_paths(result)
    lines = [f"Stage 00 visualisation step: {step}"]
    if step == "load":
        lines.extend(
            [
                f"Input file: {result['input']}",
                f"Rows loaded: {summary.get('rows', 0):,}",
                f"Columns loaded: {', '.join(summary.get('columns', []))}",
                f"Documents: {summary.get('documents', 0):,}",
                f"Chapters: {summary.get('chapters', 0):,}",
                f"Total words: {summary.get('total_words', 0):,}",
                "Output status: "
                f"{'available' if result['input'].exists() else 'missing'}",
            ]
        )
    elif step == "corpus-overview":
        lines.extend(
            [
                f"Total documents: {int(summary.get('total_documents', 0)):,}",
                f"Total sentences: {int(summary.get('total_sentences', 0)):,}",
                f"Total words: {int(summary.get('total_words', 0)):,}",
                "Average words per sentence: "
                f"{summary.get('average_words_per_sentence', 0):.2f}",
            ]
        )
    elif step == "document-distribution":
        lines.extend(
            [
                f"Documents included: {summary.get('documents', 0):,}",
                f"Sentences per document total: {summary.get('total_sentences', 0):,}",
                f"Words per document total: {summary.get('total_words', 0):,}",
            ]
        )
    elif step == "chapter-distribution":
        lines.append(f"Number of chapters: {summary.get('chapters', 0):,}")
        if summary.get("top_sentence_chapter"):
            lines.append(
                "Top chapter by sentence count: "
                f"{summary['top_sentence_chapter'][0]}"
            )
        if summary.get("top_word_chapter"):
            lines.append(f"Top chapter by word count: {summary['top_word_chapter'][0]}")
    elif step == "sentence-lengths":
        lines.extend(
            [
                f"Minimum sentence length: {summary.get('min_words', 0):,}",
                f"Mean sentence length: {summary.get('mean_words', 0):.2f}",
                f"Maximum sentence length: {summary.get('max_words', 0):,}",
            ]
        )
    elif step == "word-counts":
        lines.extend(
            [
                f"Total words: {summary.get('total_words', 0):,}",
                "Average words by document: "
                f"{summary.get('avg_words_by_document', 0):.2f}",
                "Average words by chapter: "
                f"{summary.get('avg_words_by_chapter', 0):.2f}",
            ]
        )
    elif step == "footnotes":
        lines.extend(
            [
                f"Footnote file used: {result['input']}",
                f"Number of footnotes: {summary.get('footnotes', 0):,}",
                f"Footnote document/page groups: {summary.get('groups', 0):,}",
            ]
        )
    else:
        for key, value in summary.items():
            lines.append(f"{key}: {value}")

    if outputs:
        lines.append("Outputs written:")
        lines.extend(f"  - {path}" for path in outputs)
    else:
        lines.append("Outputs written: none")
    return "\n".join(lines)


def concat_metaphor_results(
    results_by_approach: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate non-empty metaphor result tables."""
    frames = [df for df in results_by_approach.values() if not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nonempty_domain_approaches(
    results_by_approach: dict[str, pd.DataFrame],
) -> list[str]:
    """Return approaches with at least one source-domain value."""
    return [
        approach
        for approach, df in results_by_approach.items()
        if "dominio_fuente" in df
        and df["dominio_fuente"].notna().any()
        and (df["dominio_fuente"] != "").any()
    ]


def plot_metaphors_by_chapter(
    results_by_approach: dict[str, pd.DataFrame],
    df_n0: pd.DataFrame,
    output_dir: str | Path,
    colors: dict[str, str],
    config: dict[str, Any],
) -> list[Path]:
    """Plot metaphor counts by chapter for each approach."""
    if not results_by_approach:
        return []
    approaches = list(results_by_approach)
    n_columns = min(len(approaches), 2)
    n_rows = (len(approaches) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for index, approach in enumerate(approaches):
        ax = axes[index]
        df_joined = results_by_approach[approach].merge(
            df_n0[["ID_oracion", "capitulo"]].drop_duplicates(),
            on="ID_oracion",
            how="left",
        )
        counts = df_joined["capitulo"].value_counts().sort_index()
        ax.bar(
            counts.index.astype(str), counts.values, color=colors.get(approach, "#888")
        )
        ax.set_xlabel("Capítulo")
        ax.set_ylabel("Número de metáforas")
        ax.set_title(f"Metáforas por capítulo — {approach}")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for axis in axes[len(approaches) :]:
        axis.set_visible(False)
    name = config["figure_names"].get(
        "metaphors_by_chapter", "viz_metaforas_por_capitulo.png"
    )
    return [save_current_figure(project_path(output_dir) / name)]


def plot_top_domains(
    results_by_approach: dict[str, pd.DataFrame],
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot aggregate source and target domain frequencies."""
    df_all = concat_metaphor_results(results_by_approach)
    if df_all.empty or "dominio_fuente" not in df_all:
        return []
    df_all = df_all[df_all["dominio_fuente"].notna() & (df_all["dominio_fuente"] != "")]
    if df_all.empty:
        return []
    top_n = int(config.get("top_domains", 20))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    top_source = df_all["dominio_fuente"].value_counts().head(top_n)
    axes[0].barh(top_source.index[::-1], top_source.values[::-1], color="#1D9E75")
    axes[0].set_xlabel("Frecuencia")
    axes[0].set_title(f"Top {top_n} dominios fuente")
    top_target = df_all["dominio_meta"].value_counts().head(top_n)
    axes[1].barh(top_target.index[::-1], top_target.values[::-1], color="#7F77DD")
    axes[1].set_xlabel("Frecuencia")
    axes[1].set_title(f"Top {top_n} dominios meta")
    name = config["figure_names"].get("top_domains", "viz_top_dominios.png")
    return [save_current_figure(project_path(output_dir) / name)]


def plot_domain_heatmap(
    results_by_approach: dict[str, pd.DataFrame],
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot aggregate source-domain by target-domain heatmap."""
    df_all = concat_metaphor_results(results_by_approach)
    if df_all.empty:
        return []
    df_all = df_all[df_all["dominio_fuente"].notna() & (df_all["dominio_fuente"] != "")]
    if df_all.empty:
        return []
    top_n = int(config.get("heatmap_top_n", 15))
    top_source = df_all["dominio_fuente"].value_counts().head(top_n).index
    top_target = df_all["dominio_meta"].value_counts().head(top_n).index
    df_heat = df_all[
        df_all["dominio_fuente"].isin(top_source)
        & df_all["dominio_meta"].isin(top_target)
    ]
    cross = pd.crosstab(df_heat["dominio_fuente"], df_heat["dominio_meta"])
    if cross.empty:
        return []
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_heatmap(cross, ax)
    ax.set_title(f"Heatmap: Dominio fuente x Dominio meta (Top {top_n})")
    ax.set_xlabel("Dominio meta")
    ax.set_ylabel("Dominio fuente")
    name = config["figure_names"].get("domain_heatmap", "viz_heatmap_dominios.png")
    return [save_current_figure(project_path(output_dir) / name)]


def plot_focus_pos(
    results_by_approach: dict[str, pd.DataFrame],
    output_dir: str | Path,
    colors: dict[str, str],
    config: dict[str, Any],
) -> list[Path]:
    """Plot focus POS distribution per approach."""
    approaches = [
        approach
        for approach, df in results_by_approach.items()
        if "foco_part_of_speech" in df
        and df["foco_part_of_speech"].notna().any()
        and (df["foco_part_of_speech"] != "").any()
    ]
    if not approaches:
        return []
    n_columns = min(len(approaches), 4)
    n_rows = (len(approaches) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    pos_colors = {
        "VERB": "#3B8BD4",
        "NOUN": "#1D9E75",
        "ADJ": "#D85A30",
        "ADV": "#7F77DD",
        "PROPN": "#E8A838",
    }
    for index, approach in enumerate(approaches):
        counts = results_by_approach[approach]["foco_part_of_speech"].value_counts()
        axes[index].pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[
                pos_colors.get(pos, colors.get(approach, "#888"))
                for pos in counts.index
            ],
            startangle=90,
        )
        axes[index].set_title(f"POS del foco — {approach}")
    for axis in axes[len(approaches) :]:
        axis.set_visible(False)
    name = config["figure_names"].get("focus_pos", "viz_pos_foco.png")
    return [save_current_figure(project_path(output_dir) / name)]


def build_sankey_per_approach(
    df_sub: pd.DataFrame,
    approach_label: str,
    color: str = "#888888",
    top_n: int = 20,
) -> go.Figure | None:
    """Build Sankey: target domain -> conceptual metaphor -> source domain."""
    df_clean = df_sub[
        df_sub["dominio_fuente"].notna()
        & (df_sub["dominio_fuente"] != "")
        & df_sub["dominio_meta"].notna()
        & (df_sub["dominio_meta"] != "")
        & df_sub["metafora_conceptual"].notna()
        & (df_sub["metafora_conceptual"] != "")
    ]
    if df_clean.empty:
        return None
    top_concepts = df_clean["metafora_conceptual"].value_counts().head(top_n).index
    df_filtered = df_clean[df_clean["metafora_conceptual"].isin(top_concepts)]
    targets = df_filtered["dominio_meta"].unique().tolist()
    concepts = df_filtered["metafora_conceptual"].unique().tolist()
    sources = df_filtered["dominio_fuente"].unique().tolist()
    labels = (
        [f"[META] {target}" for target in targets]
        + concepts
        + [f"[FTE] {source}" for source in sources]
    )
    node_index = {label: index for index, label in enumerate(labels)}
    link_sources = []
    link_targets = []
    values = []
    for concept in concepts:
        df_concept = df_filtered[df_filtered["metafora_conceptual"] == concept]
        for target in df_concept["dominio_meta"].unique():
            link_sources.append(node_index[f"[META] {target}"])
            link_targets.append(node_index[concept])
            values.append(len(df_concept[df_concept["dominio_meta"] == target]))
        for source in df_concept["dominio_fuente"].unique():
            link_sources.append(node_index[concept])
            link_targets.append(node_index[f"[FTE] {source}"])
            values.append(len(df_concept[df_concept["dominio_fuente"] == source]))
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 10,
                    "thickness": 15,
                    "line": {"color": "black", "width": 0.5},
                    "label": labels,
                },
                link={
                    "source": link_sources,
                    "target": link_targets,
                    "value": values,
                    "color": hex_to_rgba(color, 0.53),
                },
            )
        ]
    )
    fig.update_layout(
        title=(
            f"Sankey — enfoque {approach_label} "
            f"(meta -> metáfora conceptual -> fuente, top {top_n})"
        ),
        font_size=10,
        height=700,
    )
    return fig


def build_sankey_consolidated(
    results_by_approach: dict[str, pd.DataFrame],
    approaches: list[str],
    colors: dict[str, str],
    top_n: int = 25,
) -> go.Figure | None:
    """Build consolidated Sankey: target domain -> source domain."""
    frames = [
        results_by_approach[approach].copy()
        for approach in approaches
        if approach in results_by_approach
        and not results_by_approach[approach].empty
        and "dominio_fuente" in results_by_approach[approach]
    ]
    if not frames:
        return None
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[
        df_all["dominio_fuente"].notna()
        & (df_all["dominio_fuente"] != "")
        & df_all["dominio_meta"].notna()
        & (df_all["dominio_meta"] != "")
    ]
    if df_all.empty:
        return None
    group = (
        df_all.groupby(["dominio_meta", "dominio_fuente", "enfoque"])
        .size()
        .reset_index(name="count")
    )
    pair_totals = (
        group.groupby(["dominio_meta", "dominio_fuente"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    top_pairs = set(
        zip(pair_totals["dominio_meta"], pair_totals["dominio_fuente"], strict=False)
    )
    filtered = group[
        group.apply(
            lambda row: (row["dominio_meta"], row["dominio_fuente"]) in top_pairs,
            axis=1,
        )
    ]
    dominant = {
        pair: sub.loc[sub["count"].idxmax(), "enfoque"]
        for pair, sub in filtered.groupby(["dominio_meta", "dominio_fuente"])
    }
    targets = sorted({target for target, _ in top_pairs})
    sources = sorted({source for _, source in top_pairs})
    labels = [f"[META] {target}" for target in targets] + [
        f"[FTE] {source}" for source in sources
    ]
    node_index = {label: index for index, label in enumerate(labels)}
    link_sources = []
    link_targets = []
    values = []
    link_colors = []
    for target, source in top_pairs:
        total = pair_totals[
            (pair_totals["dominio_meta"] == target)
            & (pair_totals["dominio_fuente"] == source)
        ]["count"].iloc[0]
        link_sources.append(node_index[f"[META] {target}"])
        link_targets.append(node_index[f"[FTE] {source}"])
        values.append(int(total))
        link_colors.append(
            hex_to_rgba(colors.get(dominant[(target, source)], "#888888"), 0.67)
        )
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 10,
                    "thickness": 18,
                    "line": {"color": "black", "width": 0.5},
                    "label": labels,
                },
                link={
                    "source": link_sources,
                    "target": link_targets,
                    "value": values,
                    "color": link_colors,
                },
            )
        ]
    )
    fig.update_layout(
        title=f"Sankey CONSOLIDADO — dominio META -> dominio FUENTE (top {top_n})",
        font_size=10,
        height=800,
    )
    return fig


def visualise_primary_metaphors(
    df_n0: pd.DataFrame,
    results_by_approach: dict[str, pd.DataFrame],
    config: dict[str, Any],
    colors: dict[str, str],
    epistemic_correspondences: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate the main N1 primary-metaphor visualisations."""
    figures_dir = config["outputs"]["figures_dir"]
    html_dir = project_path(config["outputs"]["html_dir"])
    html_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written.extend(
        plot_metaphors_by_chapter(
            results_by_approach, df_n0, figures_dir, colors, config
        )
    )
    written.extend(plot_top_domains(results_by_approach, figures_dir, config))
    written.extend(plot_domain_heatmap(results_by_approach, figures_dir, config))
    written.extend(plot_focus_pos(results_by_approach, figures_dir, colors, config))

    if epistemic_correspondences is not None and not epistemic_correspondences.empty:
        df_epi = epistemic_correspondences[
            epistemic_correspondences["enfoque"].isin(results_by_approach)
        ]
        if not df_epi.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            counts = (
                df_epi.groupby(["tipo_inferencia", "enfoque"])
                .size()
                .unstack(fill_value=0)
            )
            counts.plot(
                kind="bar",
                ax=ax,
                color=[colors.get(col, "#888") for col in counts.columns],
            )
            ax.set_xlabel("Tipo de inferencia")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Correspondencias epistémicas por tipo y enfoque")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Enfoque")
            name = config["figure_names"].get(
                "epistemic_correspondences", "viz_correspondencias_epistemicas.png"
            )
            written.append(save_current_figure(project_path(figures_dir) / name))

    sankey_top = int(config.get("sankey_top_n", 20))
    for approach, df in results_by_approach.items():
        fig = build_sankey_per_approach(
            df, approach, colors.get(approach, "#888888"), sankey_top
        )
        if fig is not None:
            path = html_dir / config["figure_names"].get(
                "sankey_by_approach", "viz_sankey_{approach}.html"
            ).format(approach=approach)
            pio.write_html(fig, path, include_plotlyjs="cdn")
            written.append(path)

    fig = build_sankey_consolidated(
        results_by_approach,
        list(results_by_approach),
        colors,
        int(config.get("consolidated_sankey_top_n", 25)),
    )
    if fig is not None:
        path = html_dir / config["figure_names"].get(
            "sankey_consolidated", "viz_sankey_consolidado.html"
        )
        pio.write_html(fig, path, include_plotlyjs="cdn")
        written.append(path)
    return written


def run_stage_00_visualisation(config: dict[str, Any]) -> list[Path]:
    """Load configured N0 data and generate corpus visualisations."""
    written = []
    for step in STAGE_00_VISUALISATION_EXECUTION_STEPS:
        result = run_stage_00_visualisation_step(config, step)
        written.extend(stage_00_visualisation_output_paths(result))
    return written


def run_stage_01_visualisation(config: dict[str, Any]) -> list[Path]:
    """Load configured N1 data and generate primary metaphor visualisations."""
    viz_config = config["stage_01_visualisation"]
    stage_config = config["stage_01"]
    df_n0 = pd.read_csv(project_path(viz_config["inputs"]["corpus_csv"]))
    results = load_approach_results(
        viz_config.get("approaches", []), viz_config["inputs"]["metaphors_pattern"]
    )
    epistemic_path = project_path(viz_config["inputs"]["epistemic_correspondences_csv"])
    epistemic = pd.read_csv(epistemic_path) if epistemic_path.exists() else None
    return visualise_primary_metaphors(
        df_n0,
        results,
        viz_config,
        stage_config.get("approaches", {}).get("colors", {}),
        epistemic,
    )
