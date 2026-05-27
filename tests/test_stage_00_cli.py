import importlib.util
from pathlib import Path


def load_stage_00_script():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "00_ingest_corpus.py"
    )
    spec = importlib.util.spec_from_file_location("stage_00_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_00_cli_parser_accepts_requested_options() -> None:
    module = load_stage_00_script()
    args = module.build_parser().parse_args(
        [
            "--step",
            "inspect-cleaning",
            "--sample-size",
            "2",
            "--window-size",
            "50",
            "--random-state",
            "7",
            "--document-id",
            "DOC-1",
            "--page-number",
            "3",
            "--limit-files",
            "4",
            "--limit-pages",
            "5",
            "--limit-sentences",
            "6",
            "--write-csv",
        ]
    )
    assert args.step == "inspect-cleaning"
    assert args.sample_size == 2
    assert args.window_size == 50
    assert args.random_state == 7
    assert args.document_id == "DOC-1"
    assert args.page_number == 3
    assert args.limit_files == 4
    assert args.limit_pages == 5
    assert args.limit_sentences == 6
    assert args.write_csv is True
