import argparse
import sys

from ai_melt.config import load_config
from ai_melt.paths import ensure_project_directories
from ai_melt.status import (
    format_named_status_report,
    get_stage_01_visualisation_status_report,
    next_stage_01_visualisation_command,
    reset_stage_status,
    stage_01_visualisation_status_path,
    update_named_stage_status,
)
from ai_melt.visualisation import (
    STAGE_01_VISUALISATION_STEPS,
    run_stage_01_visualisation_step,
    summarise_stage_01_visualisation_result,
)

STEPS = [*STAGE_01_VISUALISATION_STEPS, "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 01 primary-metaphor visualisation step by step. "
            f"Available steps: {', '.join(STEPS)}."
        )
    )
    parser.add_argument("--step", choices=STEPS, default="all")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--reset-status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_argv = argv if argv is not None else sys.argv[1:]
    ensure_project_directories()
    config = load_config()
    status_path = stage_01_visualisation_status_path(config)
    if args.reset_status:
        removed = reset_stage_status(status_path)
        print(
            "Stage 01 visualisation status reset."
            if removed
            else "No Stage 01 visualisation status file found."
        )
        return
    if args.status:
        print(
            format_named_status_report(get_stage_01_visualisation_status_report(config))
        )
        return
    if args.next:
        print(next_stage_01_visualisation_command(config))
        return
    command = "python scripts/01_visualise_primary_metaphors.py " + " ".join(raw_argv)
    steps = STAGE_01_VISUALISATION_STEPS if args.step == "all" else [args.step]
    for step in steps:
        try:
            result = run_stage_01_visualisation_step(config, step)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        print(summarise_stage_01_visualisation_result(step, result))
        print()
        update_named_stage_status(
            "stage_01_visualisation",
            step,
            result,
            status_path,
            (
                f"python scripts/01_visualise_primary_metaphors.py --step {step}"
                if args.step == "all"
                else command
            ),
            vars(args),
        )


if __name__ == "__main__":
    main()
