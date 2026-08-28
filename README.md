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

## Phase 1 UI validation

The one-command validation launcher keeps the repository database untouched. It
creates a temporary empty ChromaDB, imports the default paper from
`../2602.08213v1.pdf`, and starts the normal UI:

```bash
bash scripts/launch_phase1_ui_test.sh
```

The launcher reads the local `.env` for `DEEPSEEK_API_KEY`. It does not copy the
key into the temporary directory. After the browser checks, press `Ctrl+C` in
the terminal; the temporary database is removed automatically.

To test only compatibility with the existing local index, use:

```bash
bash scripts/launch_phase1_ui_test.sh --existing
```

To use a different PDF in the fresh-index test:

```bash
bash scripts/launch_phase1_ui_test.sh /absolute/path/to/paper.pdf
```

Acceptance checks for the default paper:

- the upload, chat, outline, and quiz tabs are present;
- Table 2 + `DrugR*` + Overall Optimization Score returns `0.2060`;
- Table 2 + `DrugR*` + Target property F1 score returns `0.3404`;
- Table 1 + `DrugR` + Overall Optimization Score returns `0.2712`;
- the answer does not substitute Table 1 for an explicit Table 2 question;
- a normal narrative question, outline generation, quiz generation, and a
  second document upload still complete normally.
