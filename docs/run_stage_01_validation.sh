# Stage 01 processing validation commands

source .venv/Scripts/activate
pip install -e ".[dev]"
pytest

# Stage 00 prerequisite
python scripts/00_ingest_corpus.py --status
find data/processed -maxdepth 1 -type f

# Credentials: create .env from the example, then replace placeholders.
test -f .env || cp .env.example .env

python scripts/01_process_primary_metaphors.py --help
python scripts/01_process_primary_metaphors.py --status
python scripts/01_process_primary_metaphors.py --reset-status

python scripts/01_process_primary_metaphors.py --step config
python scripts/01_process_primary_metaphors.py --step load-data --write-csv
python scripts/01_process_primary_metaphors.py --step design-prompt

# These two commands require real keys in .env and may incur API costs.
python scripts/01_process_primary_metaphors.py --step approach-a-claude
python scripts/01_process_primary_metaphors.py --step approach-b-openai

python scripts/01_process_primary_metaphors.py --step export-approach-results --write-csv
python scripts/01_process_primary_metaphors.py --step load-results --write-csv
python scripts/01_process_primary_metaphors.py --step compare-approaches --write-csv
python scripts/01_process_primary_metaphors.py --step consolidate-results --write-csv
python scripts/01_process_primary_metaphors.py --step human-evaluation-and-summary

python scripts/01_process_primary_metaphors.py --status
python scripts/01_process_primary_metaphors.py --next
find data/interim -maxdepth 1 -type f
find data/processed -maxdepth 1 -type f
find outputs/tables -maxdepth 1 -type f

pytest
git status --short
git status --short data outputs notebooks/legacy .env

# Suggested source-only staging
git add .env.example .gitignore README.md config/settings.yaml
git add scripts/01_process_primary_metaphors.py
git add src/ai_melt/primary_metaphors.py src/ai_melt/status.py
git add tests docs/run_stage_01_validation.sh
git commit -m "feat: add inspectable stage 01 processing pipeline"
