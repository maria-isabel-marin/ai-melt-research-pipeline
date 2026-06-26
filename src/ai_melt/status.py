"""Local pipeline status helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ai_melt.paths import ROOT_DIR, project_path

STAGE_00_NAME = "stage_00"
STAGE_00_STATUS_PATH = Path("outputs/logs/stage_00_status.json")
STAGE_00_VISUALISATION_NAME = "stage_00_visualisation"
STAGE_00_VISUALISATION_STATUS_PATH = Path(
    "outputs/logs/stage_00_visualisation_status.json"
)
STAGE_00_STEP_ORDER = [
    "discover",
    "extract",
    "detect-chapters",
    "clean",
    "inspect-cleaning",
    "inspect-footnotes",
    "segment",
    "annotate",
    "build",
    "export",
]
STAGE_00_VALIDATION_STEPS = {"inspect-cleaning", "inspect-footnotes"}
STAGE_00_VISUALISATION_STEP_ORDER = [
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
STAGE_00_VISUALISATION_STEP_TYPES = {
    "load": "diagnostic",
    "corpus-overview": "table",
    "document-distribution": "visualisation",
    "chapter-distribution": "visualisation",
    "sentence-lengths": "visualisation",
    "named-entities": "visualisation",
    "pos-distribution": "visualisation",
    "word-counts": "visualisation",
    "footnotes": "diagnostic",
    "export-summary": "table",
}
STAGE_01_STEP_ORDER = [
    "config",
    "load-data",
    "design-prompt",
    "approach-a-claude",
    "approach-b-openai",
    "export-approach-results",
    "load-results",
    "compare-approaches",
    "consolidate-results",
    "human-evaluation-and-summary",
]
STAGE_01_STEP_TYPES = {
    "config": "diagnostic",
    "load-data": "processing",
    "design-prompt": "diagnostic",
    "approach-a-claude": "api",
    "approach-b-openai": "api",
    "export-approach-results": "export",
    "load-results": "processing",
    "compare-approaches": "analysis",
    "consolidate-results": "processing",
    "human-evaluation-and-summary": "validation",
}
STAGE_01_VISUALISATION_STEP_ORDER = [
    "load",
    "metaphors-by-chapter-and-approach",
    "top-domains-aggregated",
    "top-source-domains-by-approach",
    "top-target-domains-by-approach",
    "source-target-heatmap",
    "focus-pos-by-approach",
    "conceptual-metaphor-wordcloud",
    "epistemic-correspondences",
    "sankey-by-approach",
    "sankey-consolidated",
    "approach-concordance-matrix",
    "summary",
]
STAGE_01_VISUALISATION_STEP_TYPES = {
    step: "visualisation" for step in STAGE_01_VISUALISATION_STEP_ORDER
}
STAGE_01_VISUALISATION_STEP_TYPES["load"] = "diagnostic"
STAGE_01_VISUALISATION_STEP_TYPES["summary"] = "table"


def _root(root: Path | None = None) -> Path:
    return root or ROOT_DIR


def status_path(root: Path | None = None) -> Path:
    """Return the local Stage 00 status path."""
    return project_path(STAGE_00_STATUS_PATH, _root(root))


def visualisation_status_path(root: Path | None = None) -> Path:
    """Return the local Stage 00 visualisation status path."""
    return project_path(STAGE_00_VISUALISATION_STATUS_PATH, _root(root))


def load_stage_status(path: str | Path | None = None) -> dict[str, Any]:
    """Load a stage status JSON file, returning an empty structure if absent."""
    resolved = Path(path) if path is not None else status_path()
    if not resolved.exists():
        return {"stages": {}}
    with resolved.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else {"stages": {}}


def save_stage_status(status: dict[str, Any], path: str | Path | None = None) -> Path:
    """Persist a stage status JSON file."""
    resolved = Path(path) if path is not None else status_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as file:
        json.dump(status, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return resolved


def reset_stage_status(path: str | Path | None = None) -> bool:
    """Delete the local status file. Return True when a file was removed."""
    resolved = Path(path) if path is not None else status_path()
    if resolved.exists():
        resolved.unlink()
        return True
    return False


def stage_01_status_path(config: dict[str, Any], root: Path | None = None) -> Path:
    """Return the configured Stage 01 processing status path."""
    return project_path(config["stage_01"]["status"]["file"], _root(root))


def update_named_stage_status(
    stage_name: str,
    step: str,
    result: Any,
    path: str | Path,
    command: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a successful step for any named local stage."""
    status = load_stage_status(path)
    stage = status.setdefault("stages", {}).setdefault(stage_name, {"steps": {}})
    stage["stage"] = stage_name
    stage["updated_at"] = datetime.now(timezone.utc).isoformat()
    stage.setdefault("steps", {})[step] = {
        "stage": stage_name,
        "step": step,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outputs": [str(output) for output in _result_outputs(result)],
        "summary": _result_summary(result),
        "command": command,
        "parameters": _json_safe(parameters or {}),
    }
    save_stage_status(status, path)
    return status


def _named_status_report(
    stage_name: str,
    step_order: list[str],
    step_types: dict[str, str],
    expected: dict[str, list[Path]],
    path: str | Path,
    command_script: str,
    title: str,
) -> dict[str, Any]:
    status = load_stage_status(path)
    records = status.get("stages", {}).get(stage_name, {}).get("steps", {})
    rows = []
    for step in step_order:
        record = records.get(step, {})
        expected_paths = expected.get(step, [])
        missing = [output for output in expected_paths if not output.exists()]
        complete = bool(expected_paths) and not missing
        if record.get("status") == "completed" and not missing:
            complete = True
        rows.append(
            {
                "step": step,
                "type": step_types[step],
                "status": "completed" if complete else "pending",
                "recorded_status": record.get("status", "pending"),
                "missing_outputs": missing,
                "record": record,
            }
        )
    next_step = get_next_recommended_step_from_rows(rows)
    return {
        "rows": rows,
        "next_step": next_step,
        "command_script": command_script,
        "title": title,
    }


def configured_stage_01_outputs(
    config: dict[str, Any], root: Path | None = None
) -> dict[str, list[Path]]:
    """Map Stage 01 processing steps to expected output paths."""
    stage = config["stage_01"]
    base = _root(root)

    def path(section: str, key: str, **values: str) -> Path:
        return project_path(stage[section][key].format(**values), base)

    return {
        "config": [path("outputs", "config_summary_json")],
        "load-data": [path("intermediate_outputs", "work_parquet")],
        "design-prompt": [path("intermediate_outputs", "prompt_preview_txt")],
        "approach-a-claude": [
            path("intermediate_outputs", "raw_results_pattern", approach="claude"),
            path("intermediate_outputs", "parsed_results_pattern", approach="claude"),
        ],
        "approach-b-openai": [
            path("intermediate_outputs", "raw_results_pattern", approach="openai"),
            path("intermediate_outputs", "parsed_results_pattern", approach="openai"),
        ],
        "export-approach-results": [
            path("outputs", "metaphors_pattern", approach="claude"),
            path("outputs", "metaphors_pattern", approach="openai"),
        ],
        "load-results": [path("intermediate_outputs", "loaded_results_parquet")],
        "compare-approaches": [
            path("outputs", "approach_comparison_parquet"),
            path("outputs", "kappa_matrix_csv"),
        ],
        "consolidate-results": [path("outputs", "primary_metaphors_parquet")],
        "human-evaluation-and-summary": [
            path("outputs", "evaluation_sample_csv"),
            path("outputs", "final_summary_json"),
        ],
    }


def get_stage_01_status_report(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return the Stage 01 processing status report."""
    resolved = path or stage_01_status_path(config, root)
    return _named_status_report(
        "stage_01",
        STAGE_01_STEP_ORDER,
        STAGE_01_STEP_TYPES,
        configured_stage_01_outputs(config, root),
        resolved,
        "scripts/01_process_primary_metaphors.py",
        "Stage 01 processing status",
    )


def next_stage_01_command(config: dict[str, Any]) -> str:
    """Return the next recommended Stage 01 processing command."""
    step = get_stage_01_status_report(config)["next_step"]
    if step is None:
        return "Stage 01 processing is complete."
    return f"python scripts/01_process_primary_metaphors.py --step {step}"


def stage_01_visualisation_status_path(
    config: dict[str, Any], root: Path | None = None
) -> Path:
    """Return the configured Stage 01 visualisation status path."""
    return project_path(config["stage_01_visualisation"]["status"]["file"], _root(root))


def configured_stage_01_visualisation_outputs(
    config: dict[str, Any], root: Path | None = None
) -> dict[str, list[Path]]:
    """Map Stage 01 visualisation steps to expected output paths."""
    viz = config["stage_01_visualisation"]
    base = _root(root)
    figures = project_path(viz["outputs"]["figures_dir"], base)
    tables = project_path(viz["outputs"]["tables_dir"], base)
    html = project_path(viz["outputs"]["html_dir"], base)

    def table(key: str) -> Path:
        return tables / viz["table_names"][key]

    def figure(key: str) -> Path:
        return figures / viz["figure_names"][key]

    return {
        "load": [table("load_summary")],
        "metaphors-by-chapter-and-approach": [
            table("metaphors_by_chapter"),
            figure("metaphors_by_chapter"),
        ],
        "top-domains-aggregated": [
            table("top_domains_aggregated"),
            figure("top_domains"),
        ],
        "top-source-domains-by-approach": [
            table("source_domains_by_approach"),
            figure("source_domains_by_approach"),
        ],
        "top-target-domains-by-approach": [
            table("target_domains_by_approach"),
            figure("target_domains_by_approach"),
        ],
        "source-target-heatmap": [
            table("source_target_heatmap"),
            figure("domain_heatmap"),
        ],
        "focus-pos-by-approach": [table("focus_pos"), figure("focus_pos")],
        "conceptual-metaphor-wordcloud": [
            table("conceptual_metaphors"),
            figure("conceptual_metaphor_wordcloud"),
        ],
        "epistemic-correspondences": [
            table("epistemic_correspondences"),
            figure("epistemic_correspondences"),
        ],
        "sankey-by-approach": [
            table("sankey_by_approach"),
            *[
                html
                / viz["figure_names"]["sankey_by_approach"].format(approach=approach)
                for approach in viz["approaches"]
            ],
        ],
        "sankey-consolidated": [
            table("sankey_consolidated"),
            html / viz["figure_names"]["sankey_consolidated"],
        ],
        "approach-concordance-matrix": [
            table("concordance_matrix"),
            figure("agreement_heatmap"),
        ],
        "summary": [table("summary")],
    }


def get_stage_01_visualisation_status_report(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return the Stage 01 visualisation status report."""
    resolved = path or stage_01_visualisation_status_path(config, root)
    return _named_status_report(
        "stage_01_visualisation",
        STAGE_01_VISUALISATION_STEP_ORDER,
        STAGE_01_VISUALISATION_STEP_TYPES,
        configured_stage_01_visualisation_outputs(config, root),
        resolved,
        "scripts/01_visualise_primary_metaphors.py",
        "Stage 01 visualisation status",
    )


def next_stage_01_visualisation_command(config: dict[str, Any]) -> str:
    """Return the next recommended Stage 01 visualisation command."""
    step = get_stage_01_visualisation_status_report(config)["next_step"]
    if step is None:
        return "Stage 01 visualisation is complete."
    return f"python scripts/01_visualise_primary_metaphors.py --step {step}"


def configured_stage_00_outputs(
    config: dict[str, Any], root: Path | None = None
) -> dict[str, list[Path]]:
    """Map Stage 00 steps to their configured expected output paths."""
    stage_config = config["stage_00"]
    intermediate = stage_config.get("intermediate_outputs", {})
    outputs = stage_config.get("outputs", {})
    base = _root(root)

    def paths(section: dict[str, str], *keys: str) -> list[Path]:
        return [project_path(section[key], base) for key in keys if key in section]

    return {
        "discover": paths(intermediate, "discovered_files_csv"),
        "extract": paths(intermediate, "pages_raw_parquet"),
        "detect-chapters": paths(intermediate, "pages_with_chapters_parquet"),
        "clean": paths(intermediate, "pages_clean_parquet", "footnotes_csv"),
        "inspect-cleaning": paths(outputs, "cleaning_inspection_csv"),
        "inspect-footnotes": paths(outputs, "footnotes_inspection_csv"),
        "segment": paths(intermediate, "sentences_parquet"),
        "annotate": paths(intermediate, "sentences_annotated_parquet"),
        "build": paths(outputs, "corpus_parquet"),
        "export": paths(
            outputs,
            "corpus_csv",
            "corpus_parquet",
            "metadata_json",
            "footnotes_csv",
            "document_summary_csv",
            "chapter_summary_csv",
        ),
    }


def configured_stage_00_visualisation_outputs(
    config: dict[str, Any], root: Path | None = None
) -> dict[str, list[Path]]:
    """Map Stage 00 visualisation steps to configured expected output paths."""
    viz_config = config["stage_00_visualisation"]
    outputs = viz_config.get("outputs", {})
    figure_names = viz_config.get("figure_names", {})
    table_names = viz_config.get("table_names", {})
    base = _root(root)
    figures_dir = project_path(outputs.get("figures_dir", "outputs/figures"), base)
    tables_dir = project_path(outputs.get("tables_dir", "outputs/tables"), base)
    input_path = project_path(
        viz_config.get("inputs", {}).get("corpus_csv", viz_config.get("input")),
        base,
    )

    def figure(name: str) -> Path:
        return figures_dir / figure_names[name]

    def table(name: str) -> Path:
        return tables_dir / table_names[name]

    expected = {
        "load": [input_path],
        "corpus-overview": [table("corpus_overview_csv")],
        "document-distribution": [
            table("document_distribution_csv"),
            figure("sentences_by_volume"),
            figure("words_by_volume"),
        ],
        "chapter-distribution": [
            table("chapter_distribution_csv"),
            figure("sentences_by_chapter_global"),
            figure("words_by_chapter_global"),
        ],
        "sentence-lengths": [
            table("sentence_lengths_csv"),
            figure("sentence_length_corpus"),
        ],
        "named-entities": [
            table("named_entities_csv"),
            figure_names.get("named_entities_top", "viz_NER_top20_{slug}.png"),
            figure_names.get("named_entities_types", "viz_NER_tipos_{slug}.png"),
        ],
        "pos-distribution": [
            table("pos_distribution_csv"),
            figure_names.get("pos_distribution", "viz_pos_{slug}.png"),
        ],
        "word-counts": [
            table("word_counts_csv"),
            table("content_lemmas_csv"),
            figure_names.get("wordcloud", "viz_wordcloud_{slug}.png"),
        ],
        "footnotes": [table("footnotes_summary_csv")],
        "export-summary": [table("export_summary_csv")],
    }

    if input_path.exists():
        try:
            df = pd.read_csv(input_path)
            volumes = sorted(df["volumen"].dropna().unique()) if "volumen" in df else []
            for volume in volumes:
                slug = _slugify(str(volume))
                expected["chapter-distribution"].extend(
                    [
                        figures_dir
                        / figure_names.get(
                            "sentences_by_chapter_volume",
                            "viz_oraciones_{slug}.png",
                        ).format(slug=slug),
                        figures_dir
                        / figure_names.get(
                            "words_by_chapter_volume", "viz_palabras_{slug}.png"
                        ).format(slug=slug),
                    ]
                )
                expected["sentence-lengths"].append(
                    figures_dir
                    / figure_names.get(
                        "sentence_length_volume", "viz_longitud_{slug}.png"
                    ).format(slug=slug)
                )
                expected["named-entities"].extend(
                    [
                        figures_dir
                        / figure_names.get(
                            "named_entities_top", "viz_NER_top20_{slug}.png"
                        ).format(slug=slug),
                        figures_dir
                        / figure_names.get(
                            "named_entities_types", "viz_NER_tipos_{slug}.png"
                        ).format(slug=slug),
                    ]
                )
                expected["pos-distribution"].append(
                    figures_dir
                    / figure_names.get("pos_distribution", "viz_pos_{slug}.png").format(
                        slug=slug
                    )
                )
                expected["word-counts"].append(
                    figures_dir
                    / figure_names.get("wordcloud", "viz_wordcloud_{slug}.png").format(
                        slug=slug
                    )
                )
            if len(volumes) > 1:
                expected["sentence-lengths"].append(figure("sentence_length_by_volume"))
        except Exception:
            pass

    for step, paths in expected.items():
        expected[step] = [
            path if isinstance(path, Path) else figures_dir / path.format(slug="corpus")
            for path in paths
        ]
    return expected


def _slugify(value: str, max_length: int = 30) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]", "_", value[:max_length]).lower().strip("_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _result_attrs(result: Any) -> dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        return dict(result.attrs)
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return dict(item.attrs)
    return {}


def _result_outputs(result: Any) -> list[Path]:
    if isinstance(result, dict):
        outputs = []
        for key, value in result.items():
            if str(key).startswith("_"):
                continue
            if isinstance(value, Path):
                outputs.append(value)
            elif isinstance(value, list | tuple):
                outputs.extend(path for path in value if isinstance(path, Path))
        return outputs
    return [Path(path) for path in _result_attrs(result).get("outputs", [])]


def _result_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        return {"rows": len(result), **_json_safe(result.attrs)}
    if isinstance(result, tuple):
        summary: dict[str, Any] = {}
        for index, item in enumerate(result):
            if isinstance(item, pd.DataFrame):
                key = "rows" if index == 0 else f"rows_{index + 1}"
                summary[key] = len(item)
        summary.update(_json_safe(_result_attrs(result)))
        return summary
    if isinstance(result, dict):
        summary = result.get("_summary", {}).copy()
        summary.update(
            {
                f"{key}_rows": value
                for key, value in result.get("_row_counts", {}).items()
            }
        )
        return _json_safe(summary)
    return {}


def update_stage_status(
    step: str,
    result: Any,
    command: str | None = None,
    parameters: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Mark a Stage 00 step as completed after a successful run."""
    status = load_stage_status(path)
    stages = status.setdefault("stages", {})
    stage = stages.setdefault(STAGE_00_NAME, {"steps": {}})
    stage["stage"] = STAGE_00_NAME
    stage["updated_at"] = datetime.now(timezone.utc).isoformat()
    stage.setdefault("steps", {})[step] = {
        "stage": STAGE_00_NAME,
        "step": step,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outputs": [str(path) for path in _result_outputs(result)],
        "summary": _result_summary(result),
        "command": command,
        "parameters": _json_safe(parameters or {}),
    }
    save_stage_status(status, path)
    return status


def update_stage_00_visualisation_status(
    step: str,
    result: Any,
    command: str | None = None,
    parameters: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Mark a Stage 00 visualisation step as completed after a successful run."""
    resolved_path = path if path is not None else visualisation_status_path()
    status = load_stage_status(resolved_path)
    stages = status.setdefault("stages", {})
    stage = stages.setdefault(STAGE_00_VISUALISATION_NAME, {"steps": {}})
    stage["stage"] = STAGE_00_VISUALISATION_NAME
    stage["updated_at"] = datetime.now(timezone.utc).isoformat()
    stage.setdefault("steps", {})[step] = {
        "stage": STAGE_00_VISUALISATION_NAME,
        "step": step,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outputs": [str(path) for path in _result_outputs(result)],
        "summary": _result_summary(result),
        "command": command,
        "parameters": _json_safe(parameters or {}),
    }
    save_stage_status(status, resolved_path)
    return status


def get_stage_00_status_report(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build a Stage 00 status report from JSON state and the filesystem."""
    status = load_stage_status(path)
    steps_status = status.get("stages", {}).get(STAGE_00_NAME, {}).get("steps", {})
    expected = configured_stage_00_outputs(config, root)
    rows = []
    for step in STAGE_00_STEP_ORDER:
        record = steps_status.get(step, {})
        expected_paths = expected.get(step, [])
        missing_outputs = [path for path in expected_paths if not path.exists()]
        recorded_completed = record.get("status") == "completed"
        outputs_exist = bool(expected_paths) and not missing_outputs
        completed = (recorded_completed or outputs_exist) and not missing_outputs
        rows.append(
            {
                "step": step,
                "type": (
                    "validation" if step in STAGE_00_VALIDATION_STEPS else "processing"
                ),
                "recorded_status": record.get("status", "pending"),
                "status": "completed" if completed else "pending",
                "timestamp": record.get("timestamp"),
                "expected_outputs": expected_paths,
                "missing_outputs": missing_outputs,
                "record": record,
            }
        )
    next_step = get_next_recommended_step_from_rows(rows)
    return {"rows": rows, "next_step": next_step}


def get_stage_00_visualisation_status_report(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build a Stage 00 visualisation status report from JSON and outputs."""
    resolved_path = path if path is not None else visualisation_status_path(root)
    status = load_stage_status(resolved_path)
    steps_status = (
        status.get("stages", {}).get(STAGE_00_VISUALISATION_NAME, {}).get("steps", {})
    )
    expected = configured_stage_00_visualisation_outputs(config, root)
    rows = []
    for step in STAGE_00_VISUALISATION_STEP_ORDER:
        record = steps_status.get(step, {})
        expected_paths = expected.get(step, [])
        missing_outputs = [path for path in expected_paths if not path.exists()]
        recorded_completed = record.get("status") == "completed"
        outputs_exist = bool(expected_paths) and not missing_outputs
        completed = (recorded_completed or outputs_exist) and not missing_outputs
        rows.append(
            {
                "step": step,
                "type": STAGE_00_VISUALISATION_STEP_TYPES[step],
                "recorded_status": record.get("status", "pending"),
                "status": "completed" if completed else "pending",
                "timestamp": record.get("timestamp"),
                "expected_outputs": expected_paths,
                "missing_outputs": missing_outputs,
                "record": record,
            }
        )
    next_step = get_next_recommended_step_from_rows(rows)
    return {"rows": rows, "next_step": next_step}


def get_next_recommended_step_from_rows(rows: list[dict[str, Any]]) -> str | None:
    """Return the first pending Stage 00 step in canonical order."""
    for row in rows:
        if row["status"] != "completed":
            return str(row["step"])
    return None


def get_next_recommended_step(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> str | None:
    """Return the next recommended Stage 00 step."""
    return get_stage_00_status_report(config, path, root)["next_step"]


def next_recommended_command(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> str:
    """Return the next recommended CLI command."""
    step = get_next_recommended_step(config, path, root)
    if step is None:
        return "Stage 00 is complete."
    return f"python scripts/00_ingest_corpus.py --step {step}"


def next_visualisation_recommended_command(
    config: dict[str, Any],
    path: str | Path | None = None,
    root: Path | None = None,
) -> str:
    """Return the next recommended Stage 00 visualisation CLI command."""
    step = get_stage_00_visualisation_status_report(config, path, root)["next_step"]
    if step is None:
        return "Stage 00 visualisation is complete."
    return f"python scripts/00_visualise_corpus.py --step {step}"


def _format_paths(paths: list[Path]) -> str:
    if not paths:
        return "-"
    return "; ".join(str(path) for path in paths)


def format_status_report(report: dict[str, Any]) -> str:
    """Format a readable Stage 00 status table."""
    rows = report["rows"]
    lines = ["Stage 00 status", ""]
    header = (
        f"{'Step':<18} {'Type':<10} {'Status':<10} {'Recorded':<10} Missing outputs"
    )
    lines.extend([header, "-" * len(header)])
    for row in rows:
        lines.append(
            f"{row['step']:<18} {row['type']:<10} {row['status']:<10} "
            f"{row['recorded_status']:<10} {_format_paths(row['missing_outputs'])}"
        )
    next_step = report["next_step"]
    lines.append("")
    if next_step is None:
        lines.append("Next recommended step: none (Stage 00 complete)")
    else:
        lines.append(f"Next recommended step: {next_step}")
        lines.append(
            "Next recommended command: "
            f"python scripts/00_ingest_corpus.py --step {next_step}"
        )
    missing_recorded = [
        row
        for row in rows
        if row["recorded_status"] == "completed" and row["missing_outputs"]
    ]
    if missing_recorded:
        lines.append("")
        lines.append("Warnings:")
        for row in missing_recorded:
            lines.append(
                f"  - {row['step']} is recorded as completed, "
                f"but missing: {_format_paths(row['missing_outputs'])}"
            )
    return "\n".join(lines)


def format_visualisation_status_report(report: dict[str, Any]) -> str:
    """Format a readable Stage 00 visualisation status table."""
    rows = report["rows"]
    lines = ["Stage 00 visualisation status", ""]
    header = (
        f"{'Step':<24} {'Type':<14} {'Status':<10} {'Recorded':<10} Missing outputs"
    )
    lines.extend([header, "-" * len(header)])
    for row in rows:
        lines.append(
            f"{row['step']:<24} {row['type']:<14} {row['status']:<10} "
            f"{row['recorded_status']:<10} {_format_paths(row['missing_outputs'])}"
        )
    next_step = report["next_step"]
    lines.append("")
    if next_step is None:
        lines.append("Next recommended step: none (Stage 00 visualisation complete)")
    else:
        lines.append(f"Next recommended step: {next_step}")
        lines.append(
            "Next recommended command: "
            f"python scripts/00_visualise_corpus.py --step {next_step}"
        )
    missing_recorded = [
        row
        for row in rows
        if row["recorded_status"] == "completed" and row["missing_outputs"]
    ]
    if missing_recorded:
        lines.append("")
        lines.append("Warnings:")
        for row in missing_recorded:
            lines.append(
                f"  - {row['step']} is recorded as completed, "
                f"but missing: {_format_paths(row['missing_outputs'])}"
            )
    return "\n".join(lines)


def format_named_status_report(report: dict[str, Any]) -> str:
    """Format a generic named-stage status report."""
    lines = [report["title"], ""]
    header = (
        f"{'Step':<36} {'Type':<12} {'Status':<10} {'Recorded':<10} Missing outputs"
    )
    lines.extend([header, "-" * len(header)])
    for row in report["rows"]:
        lines.append(
            f"{row['step']:<36} {row['type']:<12} {row['status']:<10} "
            f"{row['recorded_status']:<10} {_format_paths(row['missing_outputs'])}"
        )
    lines.append("")
    step = report["next_step"]
    if step is None:
        lines.append("Next recommended step: none (complete)")
    else:
        lines.append(f"Next recommended step: {step}")
        lines.append(
            "Next recommended command: "
            f"python {report['command_script']} --step {step}"
        )
    return "\n".join(lines)
