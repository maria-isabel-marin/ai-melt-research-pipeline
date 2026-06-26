import argparse
import sys

from ai_melt.config import load_config
from ai_melt.paths import ensure_project_directories
from ai_melt.primary_metaphors import (
    STAGE_01_APPROACHES,
    STAGE_01_STEPS,
    run_stage_01_step,
    summarise_stage_01_result,
)
from ai_melt.status import (
    format_named_status_report,
    get_stage_01_status_report,
    next_stage_01_command,
    reset_stage_status,
    stage_01_status_path,
    update_named_stage_status,
)

STEPS = [*STAGE_01_STEPS, "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 01 primary-metaphor processing step by step. "
            f"Available steps: {', '.join(STEPS)}."
        )
    )
    parser.add_argument("--step", choices=STEPS, default="all")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--limit-rows", type=int, dest="limit_sentences")
    parser.add_argument("--limit-sentences", type=int)
    parser.add_argument("--document-id")
    parser.add_argument("--page-number", type=int)
    parser.add_argument("--approach", choices=STAGE_01_APPROACHES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--reset-status", action="store_true")
    return parser


def run_step(config: dict, step: str, args: argparse.Namespace, command: str) -> dict:
    result = run_stage_01_step(
        config,
        step,
        write_csv=args.write_csv,
        sample_size=args.sample_size,
        random_state=args.random_state,
        limit_sentences=args.limit_sentences,
        document_id=args.document_id,
        page_number=args.page_number,
        force=args.force,
    )
    print(summarise_stage_01_result(step, result))
    print()
    update_named_stage_status(
        "stage_01",
        step,
        result,
        stage_01_status_path(config),
        command,
        vars(args),
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_argv = argv if argv is not None else sys.argv[1:]
    ensure_project_directories()
    config = load_config()
    status_path = stage_01_status_path(config)
    if args.reset_status:
        removed = reset_stage_status(status_path)
        print("Stage 01 status reset." if removed else "No Stage 01 status file found.")
        return
    if args.status:
        print(format_named_status_report(get_stage_01_status_report(config)))
        return
    if args.next:
        print(next_stage_01_command(config))
        return
    if args.approach:
        expected = {
            "claude": "approach-a-claude",
            "openai": "approach-b-openai",
        }[args.approach]
        if args.step not in {"all", expected}:
            parser = build_parser()
            parser.error(
                f"--approach {args.approach} only applies to --step {expected}"
            )
    command = "python scripts/01_process_primary_metaphors.py " + " ".join(raw_argv)
    steps = STAGE_01_STEPS if args.step == "all" else [args.step]
    for step in steps:
        try:
            run_step(
                config,
                step,
                args,
                (
                    f"python scripts/01_process_primary_metaphors.py --step {step}"
                    if args.step == "all"
                    else command
                ),
            )
        except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
