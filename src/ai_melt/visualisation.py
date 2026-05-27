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
        width=1200,
        height=500,
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
    viz_config = config["stage_00_visualisation"]
    df_n0 = pd.read_csv(project_path(viz_config["input"]))
    return visualise_corpus(df_n0, viz_config)


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
