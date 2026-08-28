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

## Multi-paper benchmark (Phase 2)

The benchmark manifest is in `evaluation/benchmark/`. It currently contains
five papers and 53 cases (11 existing DrugR cases plus independently curated
cases for four additional papers). Each external PDF is represented by its
SHA-256 without storing the PDF in Git. Validate the manifest offline with:

```bash
./venv/bin/python evaluation/validate_benchmark.py
```

To verify the seed PDF and the four new PDFs against recorded hashes, pass both
external directories (the flag may be repeated):

```bash
./venv/bin/python evaluation/validate_benchmark.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --verify-files --require-complete
```

Add future PDFs outside the repository, then add their metadata and cases to the
manifest/JSONL files before implementing or comparing new retrieval methods.

The current Phase 2 parser regression suite is offline and does not rebuild the
database:

```bash
./venv/bin/python -m unittest discover -s tests -v
```

The suite covers caption placement, grouped/unit table headers, PDF markup
boundaries, and layout-table false positives. A local PDF smoke check confirms
that the same ingestion path can recover table numbers and deterministic cells;
it does not prove retrieval or answer-generation quality.

For a no-API, no-Chroma multi-paper retrieval baseline, run:

```bash
./venv/bin/python evaluation/benchmark_retrieval.py \
  --papers-dir /Users/qinleqi/Desktop \
  --papers-dir /Users/qinleqi/Desktop/sci-rag-benchmark-papers \
  --top-k 1,3,5,10
```

This BM25-lite diagnostic ranks one global in-memory index and reports document,
reference-context, page, table-number, and deterministic required-fact coverage
proxies. Add `--show-failures` to print every case that is not fully covered and
its missing facts. Cross-language fact matches are accepted only through
case-level aliases that the validator checks against human gold contexts. It is
the fixed comparison baseline for Hybrid/RRF; it does not call DeepSeek or prove
answer correctness.

For a local-only Hybrid/RRF comparison (the embedding model must already be
cached), use `--retriever hybrid` with `HF_HUB_OFFLINE=1`. The diagnostic does
not change the app's default dense retrieval path or the existing ChromaDB.

## Experimental Hybrid runtime

The application still defaults to the original Chroma dense retrieval. To test
the experimental BM25 + dense Reciprocal Rank Fusion path for one process, run:

```bash
SCI_RAG_RETRIEVAL_MODE=hybrid ./venv/bin/python app.py
```

The first Hybrid question reads the current collection once and builds an
in-memory BM25 snapshot. Later questions reuse it, and a document upload through
this runtime invalidates it. `SCI_RAG_HYBRID_CANDIDATE_K` controls candidates
from each ranking (default `50`), `SCI_RAG_HYBRID_RRF_K` controls the RRF
constant (default `60`), and `SCI_RAG_CONTEXT_K` still caps generation context.
No new dependency or database rebuild is required.

Explicit `Table N` handling runs after fusion: all structured table chunks are
still checked, another table cannot replace the requested one, and a resolvable
row/column question still uses deterministic cell lookup without calling the
generation model. Generic quantity phrases such as “how many samples” do not
trigger an all-table scan unless the question explicitly refers to a table.
Hybrid alone remains an opt-in retrieval experiment and is not the default.

## Experimental local cross-encoder

An additional local-only reranker can be enabled after caching the pinned model
revision. The download is about 1.1 GB and is stored in the Hugging Face cache,
not this repository:

```bash
./venv/bin/hf download BAAI/bge-reranker-base \
  config.json model.safetensors sentencepiece.bpe.model \
  special_tokens_map.json tokenizer.json tokenizer_config.json \
  --revision 2cfc18c --max-workers 4
```

Run the UI with Hybrid plus the reranker:

```bash
HF_HUB_OFFLINE=1 \
SCI_RAG_RETRIEVAL_MODE=hybrid \
SCI_RAG_RERANKER_MODEL=BAAI/bge-reranker-base \
SCI_RAG_RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70 \
./venv/bin/python app.py
```

The model is loaded only during explicit runtime creation and always uses
`local_files_only=True`; a missing cache fails instead of downloading. The
cross-encoder ranking is conservatively fused with the original Hybrid order,
then the existing explicit-table filtering and deterministic cell lookup run as
before. Dense remains the default, and Hybrid without the reranker is unchanged.

On the fixed 5-paper/53-case offline benchmark, Hybrid + cross-encoder + RRF
raised top-10 full required-fact coverage from `0.547` to `0.698`; fact macro and
micro reached `0.785/0.776`. CPU reranking averaged `2.73 s` per 50 candidates
with a `3.31 s` p95 and about `2.20 GB` process peak RSS on the test machine.
These are retrieval-context proxies, not generated-answer accuracy. See
`PHASE2_RERANKER_HANDOFF.md` for the gate, regressions, and limitations.
