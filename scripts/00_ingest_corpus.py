import argparse
import sys
import warnings

from ai_melt.config import load_config
from ai_melt.corpus import run_stage_00_step, summarise_stage_00_result
from ai_melt.paths import ensure_project_directories
from ai_melt.status import (
    format_status_report,
    get_stage_00_status_report,
    next_recommended_command,
    reset_stage_status,
    update_stage_status,
)

STEPS = [
    "discover",
    "extract",
    "inspect-extract",
    "detect-chapters",
    "inspect-chapters",
    "clean",
    "inspect-cleaning",
    "inspect-footnotes",
    "segment",
    "inspect-segmentation",
    "annotate",
    "inspect-annotations",
    "build",
    "export",
    "all",
]

ALL_EXECUTION_STEPS = [
    "discover",
    "extract",
    "detect-chapters",
    "clean",
    "segment",
    "annotate",
    "build",
    "export",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run stage 00 corpus ingestion one inspectable step at a time. "
            f"Available steps: {', '.join(STEPS)}."
        )
    )
    parser.add_argument("--step", choices=STEPS, default="all")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write CSV debug copies alongside canonical parquet outputs.",
    )
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--limit-sentences", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deprecated alias for --limit-files.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--page-number", type=int, default=None)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print local Stage 00 progress and expected-output status.",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="Print only the next recommended Stage 00 command.",
    )
    parser.add_argument(
        "--reset-status",
        action="store_true",
        help="Delete the local Stage 00 status file.",
    )
    return parser


def status_parameters(args: argparse.Namespace) -> dict:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"status", "next", "reset_status"} and value is not None
    }


def run_and_print_step(
    config: dict, step: str, args: argparse.Namespace, command: str | None = None
):
    result = run_stage_00_step(
        config,
        step,
        sample_size=args.sample_size,
        window_size=args.window_size,
        random_state=args.random_state,
        document_id=args.document_id,
        page_number=args.page_number,
        limit_files=args.limit_files,
        limit_pages=args.limit_pages,
        limit_sentences=args.limit_sentences,
        write_csv=args.write_csv,
    )
    print(summarise_stage_00_result(step, result))
    print()
    update_stage_status(
        step,
        result,
        command=command,
        parameters=status_parameters(args),
    )
    return result


def print_final_all_summary(config: dict) -> None:
    result = run_stage_00_step(config, "build")
    pages_raw = config["stage_00"]["intermediate_outputs"]["pages_raw_parquet"]
    pages_clean = config["stage_00"]["intermediate_outputs"]["pages_clean_parquet"]
    import pandas as pd

    from ai_melt.paths import project_path

    raw_count = len(pd.read_parquet(project_path(pages_raw)))
    clean_count = len(pd.read_parquet(project_path(pages_clean)))
    outputs = result.attrs.get("outputs", [])
    print("Stage 00 complete")
    print(f"Documents: {result['ID_documento'].nunique():,}")
    print(f"Pages extracted: {raw_count:,}")
    print(f"Pages after cleaning: {clean_count:,}")
    print(f"Sentences: {len(result):,}")
    print(f"Words: {int(result['n_palabras'].sum()):,}")
    print("Outputs written:")
    for path in outputs:
        print(f"  - {path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_argv = argv if argv is not None else sys.argv[1:]
    if args.limit is not None:
        warnings.warn(
            "--limit is deprecated and now maps to --limit-files. "
            "Use --limit-files, --limit-pages, or --limit-sentences instead.",
            stacklevel=2,
        )
        if args.limit_files is None:
            args.limit_files = args.limit

    ensure_project_directories()
    config = load_config()
    if args.reset_status:
        removed = reset_stage_status()
        print("Stage 00 status reset." if removed else "No Stage 00 status file found.")
        return
    if args.status:
        print(format_status_report(get_stage_00_status_report(config)))
        return
    if args.next:
        print(next_recommended_command(config))
        return

    command = "python scripts/00_ingest_corpus.py " + " ".join(raw_argv)
    if args.step == "all":
        for step in ALL_EXECUTION_STEPS:
            run_and_print_step(
                config,
                step,
                args,
                command=f"python scripts/00_ingest_corpus.py --step {step}",
            )
        print_final_all_summary(config)
    else:
        run_and_print_step(config, args.step, args, command=command)


if __name__ == "__main__":
    main()
