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


def _root(root: Path | None = None) -> Path:
    return root or ROOT_DIR


def status_path(root: Path | None = None) -> Path:
    """Return the local Stage 00 status path."""
    return project_path(STAGE_00_STATUS_PATH, _root(root))


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
        return [
            path
            for key, path in result.items()
            if not str(key).startswith("_") and isinstance(path, Path)
        ]
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
