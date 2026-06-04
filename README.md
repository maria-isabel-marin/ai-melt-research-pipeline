# AI-MELT Research Pipeline

Research pipeline for the computational operationalisation of the Metaphor Field-Loop Theory (MELT) model.

This repository contains the Python implementation of the AI-MELT research pipeline developed as part of a doctoral dissertation on conceptual metaphors, metaphor regimes, and cultural narratives.

## Pipeline stages

1. Corpus ingestion
2. Primary metaphor processing
3. Conventionalisation analysis
4. Metaphor scenario construction
5. Metaphor regime analysis
6. Cultural narrative synthesis

## Repository structure

- `src/ai_melt/`: reusable Python package.
- `scripts/`: executable scripts for each pipeline stage.
- `notebooks/legacy/`: original exploratory notebooks.
- `notebooks/reports/`: lightweight notebooks for analysis and reporting.
- `config/`: project configuration.
- `data/`: local data folders, not versioned.
- `outputs/`: generated outputs, not versioned by default.
- `tests/`: minimal tests.

`config/settings.yaml` contains the default CEV corpus configuration used by the
current thesis pipeline, including volume metadata, stage inputs/outputs,
cleaning rules, processing parameters, and visualisation filenames. The
`notebooks/legacy/` directory stores the original AI-generated notebooks as
migration references; the executable layer is now the Python package plus
scripts.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

Install the project in editable mode:

```bash
pip install -e ".[dev]"
```

Install Git hooks:

```bash
pre-commit install
nbstripout --install
```

## Run tests

```bash
pytest
```

## Run stages 00 and 01

Place private corpus PDFs/TXTs under `data/raw/corpus/`. Generated tables are
written under `data/interim`, `data/processed`, and `outputs/`; private data and
generated outputs should stay out of Git.

Run stage 00 processing:

```bash
python scripts/00_ingest_corpus.py
```

Stage 00 can also be run step by step, mirroring the original N0 notebook while
keeping each operation inspectable from the command line:

```bash
python scripts/00_ingest_corpus.py --step discover
python scripts/00_ingest_corpus.py --step extract
python scripts/00_ingest_corpus.py --step detect-chapters
python scripts/00_ingest_corpus.py --step clean
python scripts/00_ingest_corpus.py --step inspect-cleaning
python scripts/00_ingest_corpus.py --step inspect-footnotes
python scripts/00_ingest_corpus.py --step segment
python scripts/00_ingest_corpus.py --step annotate
python scripts/00_ingest_corpus.py --step build
python scripts/00_ingest_corpus.py --step export
```

Use `--step all` to run the complete N0 pipeline. Inspection and debugging
commands accept filters and sampling options:

```bash
python scripts/00_ingest_corpus.py --step inspect-cleaning --sample-size 3 --window-size 250
python scripts/00_ingest_corpus.py --step inspect-cleaning --document-id DOC-doc --page-number 12
python scripts/00_ingest_corpus.py --step extract --limit-files 1
```

Stage 00 also keeps a local progress/status file at
`outputs/logs/stage_00_status.json`. The file is ignored by Git and is updated
after each successful step. Use it to resume a step-by-step run:

```bash
python scripts/00_ingest_corpus.py --status
python scripts/00_ingest_corpus.py --next
python scripts/00_ingest_corpus.py --reset-status
```

`--status` checks both the local status file and configured expected outputs, so
it warns when a step is recorded as complete but its output files are missing.
Inspection steps are shown as recommended validation steps, not strict
dependencies for later processing.

Intermediate N0 files are written to `data/interim/`, including discovered
files, raw pages, chaptered pages, cleaned pages, segmented sentences, and
annotated sentences. Final N0 tables are written to `data/processed/` and
`outputs/tables/`.

For validation against the legacy notebooks, keep parquet as the canonical
pipeline format and add CSV inspection copies only when needed:

```bash
python scripts/00_ingest_corpus.py --step discover
python scripts/00_ingest_corpus.py --step extract --limit-pages 5 --write-csv
python scripts/00_ingest_corpus.py --step detect-chapters --write-csv
python scripts/00_ingest_corpus.py --step clean --write-csv
python scripts/00_ingest_corpus.py --step inspect-cleaning --sample-size 5 --window-size 700
python scripts/00_ingest_corpus.py --step segment --limit-pages 10 --write-csv
python scripts/00_ingest_corpus.py --step annotate --limit-sentences 100 --write-csv
python scripts/00_ingest_corpus.py --step build --write-csv
python scripts/00_ingest_corpus.py --step export
```

`--limit-files`, `--limit-pages`, and `--limit-sentences` are explicit debugging
limits for document-level, page-level, and sentence-level steps. `--write-csv`
can also be enabled by default via `stage_00.debug_outputs.write_csv_copies` in
`config/settings.yaml`.

Run stage 00 visualisation:

```bash
python scripts/00_visualise_corpus.py
```

Stage 00 visualisation can also be validated step by step from the command
line. The migrated steps follow the central cells of
`notebooks/legacy/00_corpus_ingestion_viz.ipynb`: corpus overview,
document/volume distribution, chapter distribution, sentence lengths,
named entities, POS tags, word counts/wordclouds, footnotes, and final export
summary.

```bash
python scripts/00_visualise_corpus.py --step load
python scripts/00_visualise_corpus.py --step corpus-overview
python scripts/00_visualise_corpus.py --step document-distribution
python scripts/00_visualise_corpus.py --step chapter-distribution
python scripts/00_visualise_corpus.py --step sentence-lengths
python scripts/00_visualise_corpus.py --step named-entities
python scripts/00_visualise_corpus.py --step pos-distribution
python scripts/00_visualise_corpus.py --step word-counts
python scripts/00_visualise_corpus.py --step footnotes
python scripts/00_visualise_corpus.py --step export-summary
```

Use `--step all` to regenerate the migrated N0 visual outputs. Tables are
written under `outputs/tables/`, static figures under `outputs/figures/`, and
interactive outputs, when present, under `outputs/html/`. The visualisation
status file is local-only at `outputs/logs/stage_00_visualisation_status.json`
and is updated after each successful step:

```bash
python scripts/00_visualise_corpus.py --status
python scripts/00_visualise_corpus.py --next
python scripts/00_visualise_corpus.py --reset-status
```

`--status` checks both the run record and the configured output files, so a
step is treated as complete when its expected files already exist even if it was
generated before the status file was created.

Run stage 01 processing:

```bash
python scripts/01_process_primary_metaphors.py
```

Run stage 01 visualisation:

```bash
python scripts/01_visualise_primary_metaphors.py
```

Stage 01 defaults to the `claude` and `openai` approaches from the legacy
notebook. Configure API keys in the environment before running those approaches,
or edit `config/settings.yaml` to change the active approaches.

## Run the pipeline scaffold

```bash
python -m ai_melt.pipeline
```
