# Stage 00 visualisation validation commands

# 0. Activate environment
source .venv/Scripts/activate

# 1. Reinstall project in editable mode if needed
pip install -e ".[dev]"

# 2. Run tests
pytest

# 3. Verify that Stage 00 processing outputs exist
python scripts/00_ingest_corpus.py --status

# 4. List required N0 processing outputs
find data/processed -type f
find outputs/tables -type f

# 5. Verify Stage 00 visualisation CLI help
python scripts/00_visualise_corpus.py --help

# 6. Check Stage 00 visualisation status
python scripts/00_visualise_corpus.py --status

# 7. Reset Stage 00 visualisation status if you want to validate from scratch
python scripts/00_visualise_corpus.py --reset-status

# 8. Check status after reset
python scripts/00_visualise_corpus.py --status

# 9. Load N0 corpus and print corpus diagnostics
python scripts/00_visualise_corpus.py --step load

# 10. Check status after load
python scripts/00_visualise_corpus.py --status

# 11. Generate corpus overview outputs
python scripts/00_visualise_corpus.py --step corpus-overview

# 12. Generate document distribution outputs
python scripts/00_visualise_corpus.py --step document-distribution

# 13. Generate chapter distribution outputs
python scripts/00_visualise_corpus.py --step chapter-distribution

# 14. Generate sentence length outputs
python scripts/00_visualise_corpus.py --step sentence-lengths

# 15. Generate named entity outputs
python scripts/00_visualise_corpus.py --step named-entities

# 16. Generate POS distribution outputs
python scripts/00_visualise_corpus.py --step pos-distribution

# 17. Generate word count outputs
python scripts/00_visualise_corpus.py --step word-counts

# 18. Generate footnote visualisation outputs
python scripts/00_visualise_corpus.py --step footnotes

# 19. Generate final visualisation summary outputs
python scripts/00_visualise_corpus.py --step export-summary

# 20. Check final visualisation status
python scripts/00_visualise_corpus.py --status

# 21. Check next recommended visualisation step
python scripts/00_visualise_corpus.py --next

# 22. List generated visualisation files
find outputs/tables -type f
find outputs/figures -type f
find outputs/html -type f

# 23. Open generated outputs in Windows Explorer
explorer.exe outputs\\tables
explorer.exe outputs\\figures
explorer.exe outputs\\html

# 24. Optional: run the full Stage 00 visualisation pipeline again
python scripts/00_visualise_corpus.py --reset-status
python scripts/00_visualise_corpus.py --step all
python scripts/00_visualise_corpus.py --status

# 25. Create folder for legacy notebook outputs if you want to compare manually
mkdir -p outputs/legacy_n0_visualisation
explorer.exe outputs\\legacy_n0_visualisation

# 26. Open current outputs and legacy outputs for comparison
explorer.exe outputs\\tables
explorer.exe outputs\\figures
explorer.exe outputs\\legacy_n0_visualisation

# 27. Final tests
pytest

# 28. Check Git status
git status --short
git status --short data outputs notebooks/legacy
git status --short outputs/logs

# 29. If everything is correct, stage source-code changes only
git add README.md
git add config/settings.yaml
git add scripts/00_visualise_corpus.py
git add src/ai_melt/visualisation.py
git add src/ai_melt/status.py
git add tests
git add .gitignore
git add docs/run_stage_00_visualisation_validation.sh

# 30. Commit visualisation pipeline changes
git commit -m "feat: add inspectable stage 00 visualisation pipeline"

# 31. Push branch
git push