from ai_melt.config import load_config
from ai_melt.paths import ensure_project_directories
from ai_melt.visualisation import run_stage_00_visualisation


def main() -> None:
    ensure_project_directories()
    written = run_stage_00_visualisation(load_config())
    print(f"Stage 00 visualisation complete: {len(written)} files")


if __name__ == "__main__":
    main()
