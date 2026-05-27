# Stage 00 validation commands

# 0. Activate environment
source .venv/Scripts/activate

# 1. Install spaCy Spanish model if needed
python -m spacy download es_core_news_lg

# 2. Verify spaCy model
python - <<'PY'
import spacy
nlp = spacy.load("es_core_news_lg")
print("Loaded:", nlp.meta["lang"], nlp.meta["name"], nlp.meta["version"])
PY

# 3. Run tests
pytest

# 4. Verify CLI help
python scripts/00_ingest_corpus.py --help

# 5. Discover corpus files
python scripts/00_ingest_corpus.py --step discover --write-csv

# 6. Extract first pages for validation
python scripts/00_ingest_corpus.py --step extract --limit-pages 5 --write-csv

# 7. Inspect raw extraction, if available
python scripts/00_ingest_corpus.py --step inspect-extract --sample-size 5 --window-size 700

# 8. Extract full corpus
python scripts/00_ingest_corpus.py --step extract --write-csv

# 9. Detect chapters
python scripts/00_ingest_corpus.py --step detect-chapters --write-csv

# 10. Inspect chapter detection, if available
python scripts/00_ingest_corpus.py --step inspect-chapters --sample-size 10

# 11. Clean extracted text
python scripts/00_ingest_corpus.py --step clean --write-csv

# 12. Inspect cleaning
python scripts/00_ingest_corpus.py --step inspect-cleaning --sample-size 5 --window-size 700

# 13. Inspect footnotes
python scripts/00_ingest_corpus.py --step inspect-footnotes --sample-size 10

# 14. Segment a small sample
python scripts/00_ingest_corpus.py --step segment --limit-pages 10 --write-csv

# 15. Inspect segmentation, if available
python scripts/00_ingest_corpus.py --step inspect-segmentation --sample-size 10 --window-size 500

# 16. Segment full corpus
python scripts/00_ingest_corpus.py --step segment --write-csv

# 17. Annotate a small sample
python scripts/00_ingest_corpus.py --step annotate --limit-sentences 100 --write-csv

# 18. Inspect annotations, if available
python scripts/00_ingest_corpus.py --step inspect-annotations --sample-size 10

# 19. Annotate full corpus
python scripts/00_ingest_corpus.py --step annotate --write-csv

# 20. Build final N0 dataframe
python scripts/00_ingest_corpus.py --step build --write-csv

# 21. Export corpus, metadata, footnotes, and summaries
python scripts/00_ingest_corpus.py --step export

# 22. List generated files
find data/interim -type f
find data/processed -type f
find outputs/tables -type f

# 23. Open generated outputs in Windows Explorer
explorer.exe data\\interim
explorer.exe data\\processed
explorer.exe outputs\\tables

# 24. Final tests
pytest

# 25. Check Git status
git status --short
git status --short data outputs notebooks/legacy