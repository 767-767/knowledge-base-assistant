# Sci-RAG

Scientific-paper RAG prototype with PDF/TXT/DOCX ingestion, canonical Markdown
table chunks, Chroma retrieval, DeepSeek generation, and optional RAGAS
evaluation.

## Quick start

1. Use the pinned dependencies in `requirements.txt` with a supported Python
   environment.
2. Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` only when using
   generation or the Gradio UI.  Parsing and unit tests do not require an API
   key.
3. Run offline checks with `./venv/bin/python test_setup.py` and
   `./venv/bin/python -m unittest discover -s tests -v`.
4. Launch the UI with `./venv/bin/python app.py`.
5. Run the RAGAS evaluator only when external model calls are authorized:
   `./venv/bin/python evaluation/evaluate.py`.

Importing `app.py` is side-effect free.  Models, ChromaDB, the OpenAI client,
and Gradio are initialized only by `create_runtime()`/the UI entrypoint.

For questions that explicitly name a table, row, and column (for example,
`Table 2` + `DrugR*` + `Target property F1 score`), the application parses the
Markdown table cell deterministically. This prevents a similarly worded Table
1 narrative from overriding the requested Table 2 value. A fresh database is
recommended when validating the new indexing metadata; the existing local
`chroma_db/` is intentionally ignored and is not rebuilt automatically.
