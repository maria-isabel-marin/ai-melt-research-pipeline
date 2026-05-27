from ai_melt.paths import ensure_project_directories


def run_pipeline() -> None:
    """Initialise the AI-MELT research pipeline scaffold."""
    ensure_project_directories()
    print("AI-MELT research pipeline initialised.")


if __name__ == "__main__":
    run_pipeline()
