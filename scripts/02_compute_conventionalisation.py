from ai_melt.paths import ensure_project_directories


def main() -> None:
    ensure_project_directories()
    print("Stage 02: conventionalisation analysis")


if __name__ == "__main__":
    main()
