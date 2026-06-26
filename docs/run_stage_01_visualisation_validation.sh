# Stage 01 visualisation validation commands

source .venv/Scripts/activate
pip install -e ".[dev]"
pytest

python scripts/01_process_primary_metaphors.py --status
find data/processed -maxdepth 1 -type f

python scripts/01_visualise_primary_metaphors.py --help
python scripts/01_visualise_primary_metaphors.py --status
python scripts/01_visualise_primary_metaphors.py --reset-status

python scripts/01_visualise_primary_metaphors.py --step load
python scripts/01_visualise_primary_metaphors.py --step metaphors-by-chapter-and-approach
python scripts/01_visualise_primary_metaphors.py --step top-domains-aggregated
python scripts/01_visualise_primary_metaphors.py --step top-source-domains-by-approach
python scripts/01_visualise_primary_metaphors.py --step top-target-domains-by-approach
python scripts/01_visualise_primary_metaphors.py --step source-target-heatmap
python scripts/01_visualise_primary_metaphors.py --step focus-pos-by-approach
python scripts/01_visualise_primary_metaphors.py --step conceptual-metaphor-wordcloud
python scripts/01_visualise_primary_metaphors.py --step epistemic-correspondences
python scripts/01_visualise_primary_metaphors.py --step sankey-by-approach
python scripts/01_visualise_primary_metaphors.py --step sankey-consolidated
python scripts/01_visualise_primary_metaphors.py --step approach-concordance-matrix
python scripts/01_visualise_primary_metaphors.py --step summary

python scripts/01_visualise_primary_metaphors.py --status
python scripts/01_visualise_primary_metaphors.py --next
find outputs/tables -maxdepth 1 -type f
find outputs/figures -maxdepth 1 -type f
find outputs/html -maxdepth 1 -type f

explorer.exe outputs\\tables
explorer.exe outputs\\figures
explorer.exe outputs\\html

pytest
git status --short
git status --short data outputs notebooks/legacy

# Suggested source-only staging
git add README.md config/settings.yaml scripts/01_visualise_primary_metaphors.py
git add src/ai_melt/visualisation.py src/ai_melt/status.py tests
git add docs/run_stage_01_visualisation_validation.sh
git commit -m "feat: add inspectable stage 01 visualisation pipeline"
