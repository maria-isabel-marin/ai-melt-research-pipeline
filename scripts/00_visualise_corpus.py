import argparse
import sys

from ai_melt.config import load_config
from ai_melt.paths import ensure_project_directories
from ai_melt.status import (
    format_visualisation_status_report,
    get_stage_00_visualisation_status_report,
    next_visualisation_recommended_command,
    reset_stage_status,
    update_stage_00_visualisation_status,
    visualisation_status_path,
)
from ai_melt.visualisation import (
    STAGE_00_VISUALISATION_EXECUTION_STEPS,
    STAGE_00_VISUALISATION_STEPS,
    run_stage_00_visualisation_step,
    stage_00_visualisation_output_paths,
    summarise_stage_00_visualisation_result,
)

STEPS = [*STAGE_00_VISUALISATION_STEPS, "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 00 corpus visualisation one inspectable step at a time. "
            f"Available steps: {', '.join(STEPS)}."
        )
    )
    parser.add_argument("--step", choices=STEPS, default="all")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print local Stage 00 visualisation progress and output status.",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="Print only the next recommended Stage 00 visualisation command.",
    )
    parser.add_argument(
        "--reset-status",
        action="store_true",
        help="Delete the local Stage 00 visualisation status file.",
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
) -> dict:
    result = run_stage_00_visualisation_step(config, step)
    print(summarise_stage_00_visualisation_result(step, result))
    print()
    update_stage_00_visualisation_status(
        step,
        result,
        command=command,
        parameters=status_parameters(args),
    )
    return result


def print_final_all_summary(results: list[dict]) -> None:
    outputs = []
    for result in results:
        outputs.extend(stage_00_visualisation_output_paths(result))
    print("Stage 00 visualisation complete")
    print(f"Outputs written: {len(outputs):,}")
    for path in outputs:
        print(f"  - {path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_argv = argv if argv is not None else sys.argv[1:]
    ensure_project_directories()
    config = load_config()
    status_file = visualisation_status_path()

    if args.reset_status:
        removed = reset_stage_status(status_file)
        print(
            "Stage 00 visualisation status reset."
            if removed
            else "No Stage 00 visualisation status file found."
        )
        return
    if args.status:
        print(
            format_visualisation_status_report(
                get_stage_00_visualisation_status_report(config)
            )
        )
        return
    if args.next:
        print(next_visualisation_recommended_command(config))
        return

    command = "python scripts/00_visualise_corpus.py " + " ".join(raw_argv)
    if args.step == "all":
        results = []
        for step in STAGE_00_VISUALISATION_EXECUTION_STEPS:
            results.append(
                run_and_print_step(
                    config,
                    step,
                    args,
                    command=f"python scripts/00_visualise_corpus.py --step {step}",
                )
            )
        print_final_all_summary(results)
    else:
        run_and_print_step(config, args.step, args, command=command)


if __name__ == "__main__":
    main()
